#!/bin/sh
set -e
# Teardown the optuna/mlflow stack and the kind cluster.
# Finished Jobs self-delete via ttlSecondsAfterFinished; this script removes
# everything else (and the cluster) on demand.
kubectl delete -f "$(dirname "$0")/optuna-init.yaml" 2>/dev/null || true
kubectl delete -f "$(dirname "$0")/mlflow.yaml" 2>/dev/null || true
kubectl delete -f "$(dirname "$0")/postgres.yaml" 2>/dev/null || true
kubectl delete -f "$(dirname "$0")/configmap.yaml" 2>/dev/null || true
kubectl delete -f "$(dirname "$0")/secret.yaml" 2>/dev/null || true
kind delete cluster --name broadway 2>/dev/null || true
echo "teardown complete"
