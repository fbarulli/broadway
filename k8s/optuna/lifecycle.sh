#!/bin/sh
# On-demand lifecycle for the optuna/mlflow kind stack.
#
#   train  -> activate the cluster (restore the last snapshot if present),
#             run the HPO worker jobs to completion, snapshot the results,
#             then tear the cluster down.
#   up     -> create the cluster (if absent), load images, restore state.
#   dump   -> snapshot optuna + mlflow DBs and the mlflow artifacts (no teardown).
#   down   -> dump, then delete the cluster.
#
# The cluster is only alive while training runs. Durable state lives in
# $BACKUP_DIR (gitignored): optuna.sql.gz, mlflow.sql.gz, mlflow-artifacts.tar.gz.
# The next `train` restores them, so studies resume across cycles.
set -e

DIR=$(dirname "$0")
ROOT=$(cd "$DIR/../.." && pwd)
NAME=broadway
BACKUP_DIR=${OPTUNA_BACKUP_DIR:-$ROOT/data/optuna-backup}
OPTUNA_DUMP=$BACKUP_DIR/optuna.sql.gz
MLFLOW_DUMP=$BACKUP_DIR/mlflow.sql.gz
ARTIFACTS_TAR=$BACKUP_DIR/mlflow-artifacts.tar.gz

DB_USER=$(awk '/DB_USER:/{print $2; exit}' "$DIR/secret.yaml")
DB_PASSWORD=$(awk '/DB_PASSWORD:/{print $2; exit}' "$DIR/secret.yaml")

mkdir -p "$BACKUP_DIR"

log() { echo "[lifecycle] $*"; }

cluster_up() {
  if ! kind get clusters 2>/dev/null | grep -qx "$NAME"; then
    log "creating cluster $NAME"
    kind create cluster --config "$DIR/kind-config.yaml" --name "$NAME"
  fi
  log "loading images"
  kind load docker-image broadway-base broadway-optuna-worker mlflow-server --name "$NAME"
  log "applying manifests"
  kubectl apply -f "$DIR/secret.yaml" -f "$DIR/configmap.yaml" \
    -f "$DIR/postgres.yaml" -f "$DIR/mlflow.yaml" -f "$DIR/optuna-init.yaml" >/dev/null
}

restore_state() {
  [ -f "$OPTUNA_DUMP" ] || { log "no snapshot to restore"; return; }
  log "waiting for postgres + mlflow"
  # Wait on the DEPLOYMENTS (they exist immediately after apply; a pod
  # selector wait errors when pods haven't been created yet).
  kubectl wait --for=condition=available deployment/postgres --timeout=180s >/dev/null
  kubectl wait --for=condition=available deployment/mlflow --timeout=180s >/dev/null
  PG=$(kubectl get pods -l app=postgres -o name | head -1)
  MF=$(kubectl get pods -l app=mlflow -o name | head -1)
  log "restoring optuna DB"
  PGPASSWORD=$DB_PASSWORD kubectl exec -i "$PG" -- sh -c \
    "gunzip -c | PGPASSWORD='$DB_PASSWORD' pg_restore -U '$DB_USER' -d optuna --clean --if-exists" < "$OPTUNA_DUMP"
  if [ -f "$MLFLOW_DUMP" ]; then
    log "restoring mlflow DB"
    PGPASSWORD=$DB_PASSWORD kubectl exec -i "$PG" -- sh -c \
      "gunzip -c | PGPASSWORD='$DB_PASSWORD' pg_restore -U '$DB_USER' -d mlflow --clean --if-exists" < "$MLFLOW_DUMP"
  fi
  if [ -f "$ARTIFACTS_TAR" ]; then
    log "restoring mlflow artifacts"
    kubectl exec -i "$MF" -- sh -c "tar xzf - -C /mlflow" < "$ARTIFACTS_TAR"
  fi
}

run_hpo() {
  log "starting worker jobs (rendered from the HPO spec — single source of truth)"
  # The renderer is the ONLY generator of Job manifests; the image tag is
  # injectable (CD sets OPTUNA_WORKER_IMAGE to the built registry tag).
  uv run python "$DIR/render_worker_jobs.py" \
    --image "${OPTUNA_WORKER_IMAGE:-broadway-optuna-worker:latest}" \
    | kubectl apply -f - >/dev/null
  # Wait for every job the renderer created (no hardcoded model list).
  JOBS=$(kubectl get jobs -o name | grep '^job.batch/optuna-' || true)
  for job in $JOBS; do
    log "waiting for $job"
    kubectl wait --for=condition=complete "$job" --timeout=1200s >/dev/null \
      || { log "$job FAILED"; kubectl logs "$job" | tail -20; return 1; }
  done
  log "all workers complete"
}

dump_state() {
  log "dumping state"
  PG=$(kubectl get pods -l app=postgres -o name | head -1)
  # Re-resolve the mlflow pod right before use: it can briefly be mid-rollout.
  for _ in 1 2 3; do
    MF=$(kubectl get pods -l app=mlflow -o name 2>/dev/null | head -1)
    if kubectl exec "$MF" -- true 2>/dev/null; then break; fi
    sleep 5
  done
  kubectl exec "$PG" -- sh -c "PGPASSWORD='$DB_PASSWORD' pg_dump -U '$DB_USER' -Fc optuna" \
    | gzip > "$OPTUNA_DUMP"
  kubectl exec "$PG" -- sh -c "PGPASSWORD='$DB_PASSWORD' pg_dump -U '$DB_USER' -Fc mlflow" \
    | gzip > "$MLFLOW_DUMP"
  kubectl exec "$MF" -- sh -c "tar czf - -C /mlflow ." > "$ARTIFACTS_TAR"
  log "snapshot written to $BACKUP_DIR"
}

cluster_down() {
  log "deleting cluster $NAME"
  kind delete cluster --name "$NAME"
}

view_local() {
  # VIEW ONLY — no Kubernetes. Restore the dumped snapshot into a throwaway
  # docker postgres and run mlflow locally; open the UI at localhost:5000.
  # Stop when done: `docker rm -f broadway-view-pg` + kill the mlflow process.
  # Throwaway postgres creds/port are env-overridable (dev-only defaults).
  PG_NAME=${VIEW_PG_NAME:-broadway-view-pg}
  VIEW_PORT=${MLFLOW_VIEW_PORT:-5000}
  VIEW_PG_PORT=${VIEW_PG_PORT:-15433}
  VIEW_PG_USER=${VIEW_PG_USER:-view}
  VIEW_PG_PASSWORD=${VIEW_PG_PASSWORD:-view}
  if ! [ -f "$OPTUNA_DUMP" ] || ! [ -f "$MLFLOW_DUMP" ]; then
    echo "[lifecycle] no snapshot in $BACKUP_DIR" >&2
    exit 1
  fi
  docker rm -f "$PG_NAME" >/dev/null 2>&1 || true
  docker run -d --rm --name "$PG_NAME" \
    -e POSTGRES_USER="$VIEW_PG_USER" -e POSTGRES_PASSWORD="$VIEW_PG_PASSWORD" -e POSTGRES_DB=mlflow \
    -p "$VIEW_PG_PORT:5432" postgres:16-alpine >/dev/null
  log "waiting for view postgres"
  for _ in $(seq 1 30); do
    docker exec "$PG_NAME" pg_isready -U view -q 2>/dev/null && break
    sleep 2
  done
  docker exec "$PG_NAME" createdb -U "$VIEW_PG_USER" optuna
  log "restoring dumps"
  gunzip -c "$OPTUNA_DUMP" | docker exec -i "$PG_NAME" pg_restore -U "$VIEW_PG_USER" -d optuna --no-owner
  gunzip -c "$MLFLOW_DUMP" | docker exec -i "$PG_NAME" pg_restore -U "$VIEW_PG_USER" -d mlflow --no-owner
  ART_DIR=$BACKUP_DIR/artifacts-view
  if [ -s "$ARTIFACTS_TAR" ]; then
    rm -rf "$ART_DIR"; mkdir -p "$ART_DIR"
    tar xzf "$ARTIFACTS_TAR" -C "$ART_DIR"
  fi
  log "starting local mlflow (artifact links may 404 — pod paths in the dump)"
  (cd "$ROOT" && uv run mlflow server \
    --backend-store-uri "postgresql+psycopg2://$VIEW_PG_USER:$VIEW_PG_PASSWORD@127.0.0.1:$VIEW_PG_PORT/mlflow" \
    --default-artifact-root "$ART_DIR" \
    --host 127.0.0.1 --port "$VIEW_PORT" >"$BACKUP_DIR/mlflow-view.log" 2>&1 &)
  URL="http://127.0.0.1:$VIEW_PORT"
  for _ in $(seq 1 30); do
    if curl -s -o /dev/null --max-time 5 "$URL/"; then break; fi
    sleep 3
  done
  echo "[lifecycle] mlflow UI (all dumped results): $URL"
  echo   echo "[lifecycle] stop: docker rm -f '$PG_NAME'; kill \$(pgrep -f 'mlflow server.*$VIEW_PG_PORT')"
  if command -v xdg-open >/dev/null 2>&1; then xdg-open "$URL" >/dev/null 2>&1 || true
  elif command -v open >/dev/null 2>&1; then open "$URL" >/dev/null 2>&1 || true
  fi
}

case "${1:-}" in
  train) cluster_up; restore_state; run_hpo; dump_state; cluster_down ;;
  up)    cluster_up; restore_state ;;

  view)  view_local ;;
  dump)  dump_state ;;
  down)  dump_state; cluster_down ;;
  *)
    echo "usage: $0 {train|up|view|dump|down}" >&2
    exit 2
    ;;
esac
