# gpu_training — model memory probe + HPO fan-out

Self-contained lane for measuring a model's training footprint and launching
parallel HPO workers with the measured micro-batch. No vendor model code lives
here.

Our models:
- **DeepAR** (GluonTS torch)
- **GluonTS WaveNet** (MQCNN is absent from GluonTS ≥ 0.16)
- **TimesFM 2.5** zero-shot
- **Prophet** — CPU-only; never in the VRAM probe
- The tabular linear/LightGBM + split-conformal calibration lane is separate.

## Components

- `memory_probe.py` — sweeps micro-batches through a model adapter, records
  peak VRAM (CUDA) or RSS (CPU), and writes a baseline.
- `run_hpo.py` — reads the baseline and the study config, preflights the real
  dataset, then launches parallel workers: exact trial split, one worker per
  GPU/MPS slot, storage URL redacted from printed commands.
- `hpo_kernel.py` — ONE Optuna worker over the canonical Broadway HPO
  (`broadway.training.hpo.run_model_study`): per-worker sampler seed, storage
  fallback, run record with the dataset identity.
- `hpo_study.py` — the study definition loader shared by the launcher and the
  kernel (HPO spec + project dataset/experiment + storage policy).
- `lane_contract.py` — the `module:function` adapter resolver both entries use.
- `config/` — probe config, study config (`hpo_kernel.yaml`), fan-out profiles
  (`hpo_run.yaml`, `hpo_run_kaggle.yaml`). SSOT; no values in code.
- `adapters/` — the adapter contract; `cpu_smoke.py` is the probe's reference
  implementation and `taxi.py` is the real HPO objective for this project's
  taxi dataset. Add one adapter per model (`gluonts`, `timesfm`, `prophet`).


## The memory question

Two axes:

1. **Backbone size** — the adapter returns the parameter count; run the probe
   once per candidate backbone.
2. **Micro-batch / step footprint** — sweep `micro_batch`, run one real
   forward+backward+optimizer step, record peak allocated/reserved VRAM.

The baseline stores the largest batch under `headroom`, the per-trial
footprint, the per-GPU concurrency
(`floor(free * headroom / footprint)`), and the **total** concurrency
(`per_gpu × device_count`). On a 15.6 GB T4 a 10.7 GB/trial model fits once, so
a 2×T4 host runs 2 parallel trials; the launcher must list both GPUs
(`gpus: [0, 1], mps_slots_per_gpu: 1`). (10.7 GB/trial is the 2026-11-25 T4
hardware measurement — not a taxi-model figure.) When no batch fits,
`max_batch` is `null` and the launcher refuses to start.

`peak_rss_mb` on CPU is process-cumulative (`ru_maxrss` high-water mark), not a
per-batch footprint.

## Step adapter contract

`--step module:function` resolves to:

```python
def step(batch_size: int, seq_len: int, device: str, dtype: str) -> int | None:
    """Run ONE full training step and return the model parameter count."""
```

It must raise `torch.cuda.OutOfMemoryError` (or a `RuntimeError` containing
"out of memory") on OOM; the probe records it and stops the sweep.
`adapters/cpu_smoke.py` is the reference implementation.

## Run the probe

CPU smoke (requires torch; use any torch-enabled interpreter):

```bash
PYTHONPATH=gpu_training "${TORCH_PYTHON:-python}" gpu_training/memory_probe.py \
  --config gpu_training/config/memory_probe.yaml --device cpu --batches 1,2,4 \
  --output /tmp/opencode/memory_baseline.json
```

On the GPU host, point it at the model adapter:

```bash
PYTHONPATH=gpu_training python gpu_training/memory_probe.py \
  --config gpu_training/config/memory_probe.yaml \
  --step adapters.gluonts_step:step
```

Output (`baseline/memory_baseline.json` by default):

```json
{
  "device": "cuda",
  "device_name": "Tesla T4",
  "device_count": 2,
  "total_mb": 15636,
  "free_mb": 15526,
  "params": 321908998,
  "max_batch": 32,
  "per_trial_mb": 10699,
  "concurrent_trials_per_gpu": 1,
  "concurrent_trials": 2,
  "batches": [{"batch_size": 16, "ok": true, "peak_reserved_mb": 8198}]
}
```

## The HPO study (single source of truth)

`config/hpo_kernel.yaml` names the study and nothing else:

```yaml
hpo_spec: project/config/experiments/mlflow.yaml   # -> hpo: search spaces, direction, metric, budget
model: lgbm                                        # the model's entry in that spec
dataset: taxi                                      # project/config/dataset/taxi.yaml
experiment: taxi                                   # project/config/experiment/taxi.yaml
micro_batch_param: null                            # tabular model: no batch dial
storage_env: OPTUNA_STORAGE_URL
local_storage_url: sqlite:///artifacts/gpu_training/hpo_study.db
```

The lane declares no search space, budget, metric or column: the HPO spec owns
the space/budget/metric and the dataset contract owns the path, target,
datetime and feature columns. The fan-out profiles add hardware only —
`hpo_run_kaggle.yaml` extends `hpo_run.yaml` with `gpus: [0, 1]`,
`mps_slots_per_gpu: 1` and `storage_required: false`.

## Worker kernel contract

`--objective module:function` resolves to a factory:

```python
def build(study: Mapping[str, Any]) -> Objective:
    """study = the validated study config (project config names, no secrets)."""
# Objective: objective(params: dict[str, float | int], trial) -> float
```

`adapters/taxi.py` is the real one: it loads the project dataset/experiment,
splits through the canonical splitter and returns the canonical
`broadway.training.hpo.make_objective` — the kernel never fits a model itself.

Each worker seeds the canonical TPE sampler `seed + worker`. One inherited seed
makes every worker draw the same cold-start sequence (`constant_liar` only
guards observed points), which halves a parallel budget on duplicates — a real
2xT4 run did exactly that. Storage resolves `--storage` → the study config's
env var → the local SQLite study, so a host without a shared URL still runs one
study. The URL is never printed and a traceback is scrubbed; every error exits
non-zero with a full traceback. The worker prints one JSON run record (study,
model, worker, sampler seed, trial count, dataset path/rows/schema contract,
best value/params, elapsed seconds).

## Launch the HPO fan-out

```bash
python gpu_training/run_hpo.py \
  --baseline baseline/memory_baseline.json \
  --config gpu_training/config/hpo_run.yaml \
  --dry-run          # preflight + redacted worker commands
```

The dry-run is the preflight: it parses the study config, then proves the
declared parquet exists with its declared columns and prints its identity
(path, rows, schema contract). A missing dataset fails loud — the lane never
runs on absent data. `--execute` starts all workers at once, splits the budget
exactly, and injects `micro_batch` from the baseline. The storage URL is never
printed.

## Kaggle 2xT4 profile

`config/hpo_run_kaggle.yaml` is the 2xT4 hardware profile. The concurrency
constant comes from the **2026-11-25 Kaggle 2xT4 hardware measurement**, not
from the taxi model: one patched trial peaked at 10 699 MiB on a 15 360 MiB
Tesla T4, so two co-tenant trials on one card OOM and the profile runs exactly
one trial per GPU. The taxi HPO itself (LightGBM, CPU) has **no GPU footprint
measurement** yet; `gpus: [0, 1]` there means two workers, not two CUDA models.

```bash
python gpu_training/run_hpo.py \
  --baseline baseline/memory_baseline.json \
  --config gpu_training/config/hpo_run_kaggle.yaml --dry-run
# -> 2 workers (GPU 0/1), budget split from the HPO spec, storage <redacted>
```

Transport for a live Kaggle run (the lane ships no Kaggle client on purpose):

1. Upload the parquet as a private dataset once —
   `kaggle datasets create -p <dir-with-training_data.parquet+dataset-metadata.json>`
   (or `kaggle datasets version -p <dir> -m "..."`). Credentials come from
   `/home/opc/ONE/.env` (`KAGGLE_USERNAME`, `KAGGLE_API_KEY`) via the documented
   `set -a; . /home/opc/ONE/.env; set +a`; never print or commit them.
2. In the GPU kernel, attach that dataset and put the file where the dataset
   config declares it (`data/processed/training_data.parquet`), or point the
   config overlay (`BROADWAY_CONFIG_OVERLAY_DIR=<dir>` with `dataset/taxi.yaml`
   + `experiment/taxi.yaml` whose `path` is the `/kaggle/input/...` mount).
3. Run the lane in the kernel (repo root, project installed):
   `python gpu_training/run_hpo.py --baseline <baseline>.json --config gpu_training/config/hpo_run_kaggle.yaml --execute`
4. Fetch `artifacts/gpu_training/hpo_study.db` (the study) and the worker logs.

## Transport

The probe and launcher are transport-agnostic. On a remote host, push this
directory with the installed `kaggle` or `colab` CLI, run the probe/HPO there,
and fetch the artifacts. No vendor-specific model or launcher code is vendored
in this repo.
