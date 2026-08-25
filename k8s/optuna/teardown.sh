#!/bin/sh
set -eu
# Teardown the optuna/mlflow stack and the kind cluster, then return the host
# to ZERO project footprint (MAIN_AGENT_CONTRACT "Ledger & artifact hygiene").
# LAW: kind cluster teardown destroys in-cluster optuna studies (RDBStorage rode
# postgres); this script must leave ZERO project containers/images/volumes --
# mlflow included.
# Idempotent by design -- every docker target is removed best-effort
# (`|| true`), so re-running against an already-clean host exits 0.
# Finished Jobs self-delete via ttlSecondsAfterFinished; this script removes
# everything else (and the cluster) on demand.

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"

# --- 1) in-cluster objects + kind cluster -------------------------------
kubectl delete -f "$(dirname "$0")/optuna-init.yaml" 2>/dev/null || true
kubectl delete -f "$(dirname "$0")"/mlflow.yaml 2>/dev/null || true
kubectl delete -f "$(dirname "$0")"/postgres.yaml 2>/dev/null || true
kubectl delete -f "$(dirname "$0")"/configmap.yaml 2>/dev/null || true
kubectl delete -f "$(dirname "$0")"/secret.yaml 2>/dev/null || true
kind delete cluster --name broadway 2>/dev/null || true

# --- 2) no stray optuna persistence on the host filesystem --------------
strays=''
if cd "$REPO_ROOT"; then
    strays=$(find data -maxdepth 3 \
        \( -name '*.db' -o -iname '*optuna*' -o -name '*.sqlite*' \) 2>/dev/null || true)
fi
if [ -n "$strays" ]; then
    echo "TEARDOWN REFUSED: stray optuna/db artifacts under repo/data:" >&2
    echo "$strays" >&2
    exit 1
fi

# --- 3..5) host docker: containers, images, volumes ---------------------
if command -v docker >/dev/null 2>&1; then
    # (i) project containers -- matched on BOTH the container name AND its
    # image repository (broadway* / mlflow* / optuna / view-pg / *postgres*),
    # so an oddly-named container riding a project image still dies here.
    proj_ctrs=$(docker ps -a --format '{{.Names}}\t{{.Image}}' 2>/dev/null \
        | grep -Ei 'broadway|mlflow|optuna|view-pg|postgres' \
        | cut -f1 || true)
    for c in $proj_ctrs; do
        docker rm -f "$c" >/dev/null 2>&1 || true
    done

    # (ii) ALL project image tags across BOTH census vocabularies:
    #   CI-minted:     broadway-base*, broadway-optuna-worker*, mlflow-server*
    #   manifest-dem.: broadway:broadway-*, broadway-mlflow*, broadway-postgres*
    #   judged:        postgres:16-alpine -- project-pulled (sole recorded
    #     consumer was broadway-view-pg; kind carries its own internal postgres;
    #     re-pull cost trivial if ever needed otherwise). Tolerate-not-exist.
    for img in $(docker images --format '{{.Repository}}:{{.Tag}}' 2>/dev/null \
            | grep -E '^(broadway|mlflow-server|postgres:16-alpine)' || true); do
        docker rmi -f "$img" >/dev/null 2>&1 || true
    done

    # (iii) anonymous/dangling volumes: the f60f69… census volume plus any
    # used_by=0 anonymous siblings (volume prune covers those).
    for v in $(docker volume ls --format '{{.Name}}' 2>/dev/null \
            | grep -E '^f60f69' || true); do
        docker volume rm "$v" >/dev/null 2>&1 || true
    done
    docker volume prune -f >/dev/null 2>&1 || true

    # Zero-footprint assertion: fail loudly if anything survived.
    residual_imgs=$(docker images --format '{{.Repository}}:{{.Tag}}' 2>/dev/null \
        | grep -E '^(broadway|mlflow-server|postgres:16-alpine)' || true)
    residual_ctrs=$(docker ps -a --format '{{.Names}}\t{{.Image}}' 2>/dev/null \
        | grep -Ei 'broadway|mlflow|optuna|view-pg|postgres' || true)
    if [ -n "$residual_imgs" ] || [ -n "$residual_ctrs" ]; then
        echo "TEARDOWN INCOMPLETE -- residual project docker state:" >&2
        [ -n "$residual_imgs" ] && echo "images:     $residual_imgs" >&2
        [ -n "$residual_ctrs" ] && echo "containers: $residual_ctrs" >&2
        exit 1
    fi
else
    echo "docker not present on host; skipping container/image/volume cleanup"
fi

echo "teardown complete"
