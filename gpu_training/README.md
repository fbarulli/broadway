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
- `run_hpo.py` — reads the baseline and launches parallel workers: exact trial
  split, one worker per GPU/MPS slot, storage URL redacted from printed commands.
- `config/` — probe and fan-out config (SSOT; no values in code).
- `adapters/` — the adapter contract; `cpu_smoke.py` is a reference
  implementation. Add one adapter per model (`gluonts`, `timesfm`, `prophet`).

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
(`gpus: [0, 1], mps_slots_per_gpu: 1`). When no batch fits, `max_batch` is
`null` and the launcher refuses to start.

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

## Launch several HPO runs

```bash
python gpu_training/run_hpo.py \
  --baseline baseline/memory_baseline.json \
  --config gpu_training/config/hpo_run.yaml \
  --dry-run          # prints redacted worker commands
```

`--execute` starts all workers, splits the trial budget exactly, and injects
`micro_batch` from the baseline. The command template, study name, budget and
storage env-var name are config; the storage URL is never printed.

## Transport

The probe and launcher are transport-agnostic. On a remote host, push this
directory with the installed `kaggle` or `colab` CLI, run the probe/HPO there,
and fetch the artifacts. No vendor-specific model or launcher code is vendored
in this repo.
