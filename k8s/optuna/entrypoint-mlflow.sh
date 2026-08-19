#!/bin/sh
set -e
# Read the mounted ConfigMap/Secret FILES (no env vars), wait for postgres,
# log the resolved endpoint, and start the tracking server. MLflow uses its
# OWN database (databases.mlflow) so its Alembic revisions never clash with
# optuna's shared alembic_version table.
python - <<'PY'
import socket
from pathlib import Path

import yaml

cfg = yaml.safe_load(Path("/etc/broadway/config.yaml").read_text())
db = cfg["databases"]["mlflow"]
secret = {k: Path(f"/etc/broadway/secret/{k}").read_text().strip()
          for k in ("DB_USER", "DB_PASSWORD", "DB_NAME")}
url = (f"{db['driver']}://{secret['DB_USER']}:{secret['DB_PASSWORD']}"
       f"@{db['host']}:{db['port']}/{db['name']}")
Path("/tmp/backend_uri").write_text(url)
Path("/tmp/allowed_hosts").write_text(cfg["mlflow"]["allowed_hosts"])
print(f"[mlflow] hostname={socket.gethostname()} "
      f"ip={socket.gethostbyname(socket.gethostname())} "
      f"backend={db['host']}:{db['port']}/{db['name']} "
      f"tracking_uri={cfg['mlflow']['tracking_uri']}")
PY
BACKEND_URI="$(cat /tmp/backend_uri)"
MLFLOW_SERVER_ALLOWED_HOSTS="$(cat /tmp/allowed_hosts)"
export MLFLOW_SERVER_ALLOWED_HOSTS

python - <<'PY'
import time

import sqlalchemy

url = open("/tmp/backend_uri").read().strip()
for attempt in range(30):
    try:
        sqlalchemy.create_engine(url).connect().close()
        print("[mlflow] postgres ready")
        break
    except Exception:  # noqa: BLE001 — retry loop; last attempt re-raises
        if attempt == 29:
            raise
        time.sleep(2)
PY

# Single uvicorn worker: mlflow 3.15 defaults to 4 workers, which multiplies
# memory (~2Gi+) and OOM-kills tight limits. --workers 1 keeps the tracking
# UI lean (true minimal resources) and stable.
exec mlflow server --backend-store-uri "$BACKEND_URI" \
    --default-artifact-root /mlflow/artifacts \
    --host 0.0.0.0 --port 5000 --workers 1
