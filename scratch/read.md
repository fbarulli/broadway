

---

### 1. The MLE Bundle (Serving & Infrastructure)
The MLE doesn't care about your AUC or feature importance charts. They need a deterministic, containerizable package that won't crash the API.

| Artifact | Format | Purpose |
|---|---|---|
| **Inference Pipeline** | `.pkl` (or MLflow PyFunc) | The **entire** bundle: `Preprocessor` + `Feature Engine` + `Model`. Never hand the MLE a raw model that expects raw data. |
| **Input/Output Signature** | `openapi.json` or MLflow Signature | Exact JSON schema of what the API expects (e.g., `{"fare": float}`) and returns (e.g., `{"prob": float}`). |
| **Environment Lock** | `conda.yaml` or `requirements.txt` | Exact dependency tree (with hashes) to build the Docker image. |
| **Performance Profile** | `latency_report.json` | P50, P95, P99 inference latency (in ms) and memory footprint (in MB). The MLE needs this to configure auto-scaling and Kubernetes limits. |
| **Fallback / Default Rules** | `fallback.json` | What the API should return if the model times out or throws an error (e.g., `{"default_prediction": 0.5}`). |

**Where it lives:** `artifacts/serving/`

---

### 2. The Data Tracking Bundle (Lineage & Drift Baselines)
You cannot monitor production drift if you don't save the "truth" of what the model was trained on. You need a snapshot of the training data's exact state.

| Artifact | Format | Purpose |
|---|---|---|
| **Data Manifest** | `data_manifest.json` | Cryptographic hash of the training parquet, exact row count, min/max timestamps, and schema version. |
| **Drift Baselines** | `baseline_stats.json` | The exact mean, std, and distribution (PSI/CSI bins) of every feature at training time. **This is what your production monitoring compares against.** |
| **Lineage Graph** | `graph.json` + `graph.md` | Mermaid diagram and JSON mapping of: Raw Data → Cleaned Data → Features → Model. |
| **Data Card / Fact Sheet** | `data_card.md` | High-level summary: Date range, known biases, exclusions, and missingness rates. |

**Where it lives:** `artifacts/tracking/`

---

### 3. The DS Bundle (The Story & Logic)
This is what you already have, but it needs to be formalized so the business can audit it.

| Artifact | Format | Purpose |
|---|---|---|
| **Executive Summary** | `reports/index.md` | The "so what?" Business metrics, expected profit, and go/no-go recommendation. |
| **Walkthrough Timeline** | `reports/timeline.md` | Step-by-step hypothesis testing (the gates/decisions from your CLI). |
| **Feature Importance** | `figures/shap_summary.png` | Global explanations. |
| **Error Analysis** | `figures/residuals.png` | Where the model fails (e.g., "fails on rides > $100"). |
| **Counterfactuals** | `reports/recourse.md` | Examples of DiCE outputs (what a user must change to flip the prediction). |

**Where it lives:** `reports/` and `artifacts/evaluation/`

---
