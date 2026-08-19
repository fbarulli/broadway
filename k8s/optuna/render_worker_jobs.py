"""Render the optuna worker Jobs for the HPO spec's models (dry-run / CI).

The HPO spec (configs/experiments/mlflow.yaml -> `hpo.models`) is the single
source for WHICH models get worker Jobs; the registry provides the display
names the worker CLI accepts. CI runs this with a CI image tag and asserts the
output is valid Kubernetes with the right image/env/model args — without a
cluster. The deployed manifests (worker-jobs.yaml) mirror this structure.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

from broadway.config.schema import HPOConfig
from broadway.training.models.registry import display_name

ROOT = Path(__file__).resolve().parents[2]


def render_jobs(hpo: HPOConfig, image: str) -> list[dict]:
    """One Job per HPO model: mirrored pod spec with env + model args."""
    jobs = []
    for model in hpo.models:
        name = display_name(model.name)
        jobs.append({
            "apiVersion": "batch/v1",
            "kind": "Job",
            "metadata": {"name": f"optuna-{name}"},
            "spec": {
                "backoffLimit": 2,
                "ttlSecondsAfterFinished": 120,
                "template": {
                    "spec": {
                        "restartPolicy": "Never",
                        "containers": [{
                            "name": "worker",
                            "image": image,
                            "imagePullPolicy": "IfNotPresent",
                            "command": ["python", "/app/experiments/mlflow/03_optuna_worker.py"],
                            "args": ["--model", name],
                            "env": [
                                {"name": "BROADWAY_MLFLOW_CONFIG",
                                 "value": "/app/configs/experiments/mlflow.yaml"},
                            ],
                            "volumeMounts": [
                                {"name": "config", "mountPath": "/etc/broadway/config.yaml",
                                 "subPath": "config.yaml", "readOnly": True},
                                {"name": "optuna-db", "mountPath": "/etc/broadway/secret",
                                 "readOnly": True},
                            ],
                        }],
                        "volumes": [
                            {"name": "config", "configMap": {"name": "optuna-config"}},
                            {"name": "optuna-db", "secret": {"secretName": "optuna-db"}},
                        ],
                    },
                },
            },
        })
    return jobs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", default="broadway-optuna-worker:latest")
    parser.add_argument(
        "--hpo-config",
        default=str(ROOT / "configs" / "experiments" / "mlflow.yaml"),
    )
    args = parser.parse_args()
    hpo = HPOConfig(**yaml.safe_load(Path(args.hpo_config).read_text())["hpo"])
    for job in render_jobs(hpo, args.image):
        yaml.safe_dump(job, sys.stdout, sort_keys=False)
        print("---")


if __name__ == "__main__":
    main()
