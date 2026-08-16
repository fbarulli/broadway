"""verify_experiments.py — fast entry point to verify the experiments work.

Runs ONLY lightweight checks — no model training, no figures, no cluster, no
full pytest: syntax-compiles every experiment script, validates the YAML
configs against the platform's require_keys, imports the experiment shared
modules, and spot-checks cheap structural invariants against the module's own
constants (single source of truth — no hardcoded numbers here). Exit code 0 =
verified; non-zero = something is broken (fail loud).
"""

import ast
import importlib.util
import sys
from pathlib import Path

import yaml

from broadway.evaluate.metrics import binary_metrics, compute_metrics
from broadway.stats.robust import winsorize
from broadway.training.optuna_worker import compose_db_url
from broadway.utils import require_keys

ROOT = Path(__file__).resolve().parent
EXPERIMENTS = ROOT / "experiments"
UNIVARIATE = EXPERIMENTS / "univariate" / "fare_amount_trip_distance"
MULTIVARIATE = EXPERIMENTS / "multivariate"
MLFLOW = EXPERIMENTS / "mlflow"

K8S_CONFIG_KEYS = ["dataset", "databases", "optuna", "mlflow"]


def load_module(name: str, path: Path):
    """Load a module from a file under a unique name (no shadowing)."""
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def compile_experiments() -> list[str]:
    """Syntax-compile every experiment script; return problem files."""
    bad = []
    for py in sorted(EXPERIMENTS.rglob("*.py")):
        try:
            ast.parse(py.read_text(), filename=str(py))
        except SyntaxError:
            bad.append(str(py))
    return bad


def validate_configs() -> list[str]:
    """Validate the YAML configs against require_keys; return problems."""
    problems = []
    for cfg_path in (MULTIVARIATE / "config.yaml",):
        cfg = yaml.safe_load(cfg_path.read_text())
        try:
            require_keys(cfg, ["target", "categorical", "sample", "baseline"],
                         f"{cfg_path.name}")
        except ValueError as exc:
            problems.append(str(exc))
    k8s_cfg = yaml.safe_load((ROOT / "k8s" / "optuna" / "configmap.yaml")
                             .read_text())["data"]["config.yaml"]
    try:
        require_keys(yaml.safe_load(k8s_cfg), K8S_CONFIG_KEYS, "k8s config.yaml")
    except ValueError as exc:
        problems.append(str(exc))
    return problems


def check_univariate() -> list[str]:
    """Load the univariate loader and check structural invariants."""
    problems = []
    common = load_module("_uni_verify_common", UNIVARIATE / "_common.py")
    sample = common.load_metered()
    if sample.empty:
        problems.append("univariate: load_metered returned no rows")
    if not (sample["fare_amount"] > common.MIN_FARE).all():
        problems.append("univariate: min-fare filter not honored")
    if not (sample["duration_minutes"] < common.MAX_DURATION_MINUTES).all():
        problems.append("univariate: max-duration filter not honored")
    if "trip_distance" not in sample.columns:
        problems.append("univariate: missing trip_distance column")
    return problems


def check_multivariate() -> list[str]:
    """Load the multivariate setup + config; check the sample + dummies."""
    problems = []
    setup = load_module("_mv_verify_setup", MULTIVARIATE / "_setup.py")
    cfg = setup.load_config()
    sample = setup.load_manhattan_sample(cfg)
    pickup_col = cfg["borough"]["pickup"]["column"]
    keep = cfg["sample"]["pickup_borough"]
    if sample.empty:
        problems.append("multivariate: manhattan sample empty")
    if not (sample[pickup_col] == keep).all():
        problems.append("multivariate: pickup filter not honored")
    dummies = setup.build_borough_dummies(sample.head(50), cfg)
    if dummies.shape[0] != 50:
        problems.append("multivariate: borough dummies row mismatch")
    return problems


def check_mlflow() -> list[str]:
    """Import the mlflow experiment module; spot-check metrics + url helper."""
    problems = []
    load_module("_mlflow_verify_common", MLFLOW / "_common.py")
    metrics = compute_metrics([1.0, 2.0, 3.0], [1.0, 2.0, 3.0])
    if set(metrics) != {"mae", "rmse", "r2", "mape", "max_error",
                        "median_ae", "explained_var"}:
        problems.append("mlflow: compute_metrics key set mismatch")
    binm = binary_metrics([1.0, 2.0, 3.0], [1.0, 1.0, 2.0], threshold=1.5)
    if set(binm) != {"roc_auc", "pr_auc"}:
        problems.append("mlflow: binary_metrics key set mismatch")
    url = compose_db_url("postgresql+psycopg2", "u", "p", "h", "5432", "d")
    if url != "postgresql+psycopg2://u:p@h:5432/d":
        problems.append("mlflow: compose_db_url mismatch")
    return problems


def check_robust() -> list[str]:
    """Spot-check the promoted robust helper on the real sample."""
    problems = []
    common = load_module("_uni_verify_robust", UNIVARIATE / "_common.py")
    sample = common.load_metered()
    clipped = winsorize(sample, ["fare_amount", "trip_distance"], 0.995)
    for col in ("fare_amount", "trip_distance"):
        cap = sample[col].quantile(0.995)
        if (clipped[col] > cap).any() or clipped[col].isna().any():
            problems.append(f"robust: winsorize invariant broken for {col}")
    return problems


def main() -> int:
    checks = {
        "compile experiments": compile_experiments,
        "validate configs": validate_configs,
        "univariate loader": check_univariate,
        "multivariate setup": check_multivariate,
        "mlflow + metrics": check_mlflow,
        "robust helpers": check_robust,
    }
    failed = False
    for label, fn in checks.items():
        problems = fn()
        if problems:
            failed = True
            print(f"FAIL  {label}:")
            for p in problems:
                print(f"      - {p}")
        else:
            print(f"PASS  {label}")
    if failed:
        print("\nverification FAILED")
        return 1
    print("\nverification OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
