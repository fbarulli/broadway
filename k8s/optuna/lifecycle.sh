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
  kubectl wait --for=condition=ready pod -l app=postgres --timeout=180s >/dev/null
  kubectl wait --for=condition=ready pod -l app=mlflow --timeout=180s >/dev/null
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
  log "starting worker jobs"
  kubectl apply -f "$DIR/worker-jobs.yaml" >/dev/null
  for m in ols lgbm xgb; do
    log "waiting for optuna-$m"
    kubectl wait --for=condition=complete "job/optuna-$m" --timeout=1200s >/dev/null \
      || { log "optuna-$m FAILED"; kubectl logs "job/optuna-$m" | tail -20; return 1; }
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

case "${1:-}" in
  train) cluster_up; restore_state; run_hpo; dump_state; cluster_down ;;
  up)    cluster_up; restore_state ;;
  dump)  dump_state ;;
  down)  dump_state; cluster_down ;;
  *)
    echo "usage: $0 {train|up|dump|down}" >&2
    exit 2
    ;;
esac
