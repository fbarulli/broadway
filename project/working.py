from pathlib import Path
import yaml

_CFG = None
def _load_cfg():
    global _CFG
    if _CFG is None:
        with open("configs/experiments/working.yaml") as f:
            _CFG = yaml.safe_load(f)
    return _CFG

def require_keys(d, keys):
    missing = [k for k in keys if k not in d]
    if missing:
        raise KeyError(f"config missing required keys: {missing}")

def load_working():
    cfg = _load_cfg()
    require_keys(cfg, ["parquet", "columns", "min_target_value", "max_duration_minutes", "time_buckets", "time_bucket_default"])
    require_keys(cfg["columns"], ["target", "pickup_datetime", "dropoff_datetime"])
    df = __import__("pandas").read_parquet(cfg["parquet"])
    TARGET_COL = cfg["columns"]["target"]
    PICKUP_DATETIME_COL = cfg["columns"]["pickup_datetime"]
    DROPOFF_DATETIME_COL = cfg["columns"]["dropoff_datetime"]
    df = df.rename(columns={
        PICKUP_DATETIME_COL: "pickup_datetime",
        DROPOFF_DATETIME_COL: "dropoff_datetime",
        TARGET_COL: "target",
    })
    MIN_TARGET_VALUE = float(cfg["min_target_value"])
    return df[df["target"] > MIN_TARGET_VALUE]

def load_metered():
    cfg = _load_cfg()
    require_keys(cfg, ["parquet", "columns", "min_target_value", "max_duration_minutes", "time_buckets", "time_bucket_default"])
    require_keys(cfg["columns"], ["target", "pickup_datetime", "dropoff_datetime"])
    df = __import__("pandas").read_parquet(cfg["parquet"])
    TARGET_COL = cfg["columns"]["target"]
    PICKUP_DATETIME_COL = cfg["columns"]["pickup_datetime"]
    DROPOFF_DATETIME_COL = cfg["columns"]["dropoff_datetime"]
    df = df.rename(columns={
        PICKUP_DATETIME_COL: "pickup_datetime",
        DROPOFF_DATETIME_COL: "dropoff_datetime",
        TARGET_COL: "target",
    })
    return df

def time_buckets():
    cfg = _load_cfg()
    return cfg["time_buckets"], cfg["time_bucket_default"]
