# gpu_training — GPU/Colab/Kaggle training lane (ER reference) + memory baseline

## What this is

This directory has two parts:

1. **`er_reference/`** — a faithful copy of ER's Kaggle/Colab training lane
   (`/home/opc/ONE/ER` at commit **`d5eab456`**, 66 files:
   `src/cli/{kaggle,colab,laya_hpo,laya_kernel_text,laya_*}.py`,
   the `scripts/` launchers, `config/laya_hpo_space.yaml`, `config/training.yaml`).
   These files are a **reference**, not a runnable package: they import ER's
   `core`, `model_tracks`, and `laya` modules, which were not copied. Keep them
   as the source of truth for how the lane is wired. The copy is excluded from
   the ruff gate (`ruff check … --exclude gpu_training/er_reference`): a verbatim
   copy must stay verbatim, so its diagnostics (147 today) are not this repo's
   to fix.
2. **The runnable part here** — a memory probe and an HPO fan-out that do not
   depend on ER:
   - `memory_probe.py` — measures the real training step's footprint per
     micro-batch and writes a baseline.
   - `run_hpo.py` — reads the baseline and launches several HPO workers with
     the measured `micro_batch` and concurrency.
   - `config/` — probe and run config (no literals in code).
   - `adapters/` — the one function the probe calls (a smoke adapter ships; a
     `laya_step` adapter is what you point at ER's trainer on the GPU host).

## ER's HPO baseline (from the copied config)

`er_reference/config/laya_hpo_space.yaml`:

- model: **421M parameters** (`model_key: laya`);
- target GPU: **T4, 15 GB**;
- `micro_batch` **pinned at 16** with `grad_accum: 4` (effective batch 64) —
  the comment states 16 "keeps the 421M model + AdamW + best-state copy inside
  15GB";
- HPO: Optuna/TPE, 10 trials, shared-study budget, `n_jobs: 2` per T4 session;
- objective: maximize `dev_accuracy`, never test.

The probe replaces the hand-pinned number with a measurement on the actual host.

## The memory question

"Which model size fits memory" has two axes; the probe measures both:

1. **Backbone size** — the adapter returns the parameter count; run the probe
   once per candidate backbone to compare params vs peak VRAM.
2. **Micro-batch / step footprint** — sweep `micro_batch`, run one full
   forward+backward+optimizer step, and record peak allocated/reserved VRAM.

It reports the largest batch that fits under `headroom` of total, the per-trial
footprint, and an upper bound on concurrent trials
(`floor(free * headroom / footprint)`). That is the baseline the HPO run needs.
When no successful batch fits, `max_batch` is `null` — the launcher then refuses
to start rather than running a batch the probe already rejected.

On CPU, `peak_rss_mb` is **process-cumulative** (`ru_maxrss` high-water mark for
the whole probe process), not the footprint of that one batch. It still answers
"does the machine survive this batch", and it is monotonic, so the largest
observed value is the honest one.

## Step adapter contract

`--step module:function` resolves to:

```python
def step(batch_size: int, seq_len: int, device: str, dtype: str) -> int | None:
    """Run ONE full training step and return the model parameter count."""
```

The function must raise `torch.cuda.OutOfMemoryError` (or a `RuntimeError`
containing "out of memory") on OOM; the probe records it and stops the sweep.
`adapters/cpu_smoke.py` is a tiny reference implementation.

## Run the probe

CPU smoke (works without a GPU; uses ER's torch venv):

```bash
PYTHONPATH=gpu_training /home/opc/ONE/ER/.venv/bin/python gpu_training/memory_probe.py \
  --config gpu_training/config/memory_probe.yaml --device cpu --batches 1,2,4 \
  --output /tmp/opencode/memory_baseline.json
```

On the GPU host, point it at the real trainer adapter:

```bash
PYTHONPATH=gpu_training python gpu_training/memory_probe.py \
  --config gpu_training/config/memory_probe.yaml \
  --step adapters.laya_step:step
```

Output (`baseline/memory_baseline.json` by default):

```json
{
  "device": "cuda",
  "device_name": "Tesla T4",
  "total_mb": 15360,
  "free_mb": 15000,
  "params": 421000000,
  "max_batch": 16,
  "per_trial_mb": 12800,
  "concurrent_trials": 1,
  "batches": [{"batch_size": 16, "ok": true, "peak_reserved_mb": 12800}]
}
```

## Launch several HPO runs

```bash
python gpu_training/run_hpo.py \
  --baseline baseline/memory_baseline.json \
  --config gpu_training/config/hpo_run.yaml \
  --dry-run          # prints the resolved worker commands (storage URL redacted)
```

`--execute` starts the workers (one per assigned GPU/MPS slot), splits the
trial budget, and injects `micro_batch` from the baseline. The command template
is config; no study name, storage URL, or budget is code. The shared Optuna URL
comes from the env var named in `config/hpo_run.yaml` (`OPTUNA_STORAGE_URL`),
never a file or a literal.

Budget split: the trial count is divided exactly — floor per worker, remainder
to the first workers — so 5 trials over 2 workers is 3 + 2, never 3 + 3. Workers
never outnumber trials.

Fan-out is concurrent: `--execute` starts every worker with `Popen` and only
then waits on each exit code, so two workers cost the slower one, not the sum.
Any non-zero code fails the launch (exit 1) with the failing codes.

Secret handling: the storage URL is redacted in **everything printed** — the
dry-run plan and the `--execute` banner show `--storage <redacted>`; only the
child argv carries the real value. Residual exposure: the value is still in the
child's `/proc/<pid>/cmdline` while a worker runs, so anything able to read
process arguments on the host can see it. A template that passes the env-var
name (e.g. `--storage-env {storage_env}`) instead of the value would remove
that too; that is a change on the ER HPO kernel side.

## What stays in ER

The model, trainer, and remote launcher (`laya_train_patch_*`,
`laya_kernel_text_*`, `colab.py`, `kaggle_*`) still execute in ER; this lane
provides the measured memory baseline and the fan-out. Credentials remain in
`/home/opc/ONE/.env` and are resolved by ER's credential owner, never copied
here.
