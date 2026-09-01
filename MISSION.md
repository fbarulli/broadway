# MISSION.md — euromonitor entity resolution

Mission: build a defensible entity-resolution (record-linkage) pipeline for the
euromonitor SKU dataset, mapping every SKU to a stable ITEM_ID. Ground truth is
partial: GTINs are known for some SKUs only. Every phase below maps to an
evaluation criterion.

Data source: https://ds-item-task-782206759520.europe-west1.run.app/
(Shiny app; credentials via env vars, never committed) → `project/data/euromonitor/`.

## 1. EDA (keep tight, ~30% of notebook time)

- **GTIN coverage**: % of SKUs with a GTIN, distribution of SKUs-per-GTIN, how
  many GTINs appear across multiple retailers/countries → how much "free"
  labelled data exists.
- **Field completeness**: missing rates per column (brand, category, attribute,
  price).
- **Noise check**: sample sku_name_eng/description_short_eng across retailers
  for the same GTIN to see how much text varies (translation noise, ordering,
  units).
- **Attribute JSON**: parse and flatten it; check key consistency across
  retailers (same attribute may have different key names).
- **Duplicates within retailer**: check whether a single retailer ever lists
  the same GTIN twice (variant listings).

## 2. Framing the problem

This is **entity resolution / record linkage**, not classification — there are
no exhaustive labels, only partial GTIN ground truth. Two-stage design:

- **Stage A — GTIN-known SKUs**: trivially group by GTIN. Use these as the
  **labelled training/validation set** for stage B.
- **Stage B — GTIN-missing SKUs**: predict which existing item (or new item)
  they belong to using similarity learning.

## 3. Feature engineering

- Normalize text: lowercase, strip units/packaging noise, extract numeric
  quantities (volume, pack count) via regex.
- Structured features from attributes JSON: flavor, volume, pack type,
  caffeine, brand — align key names across retailers first.
- Candidate blocking keys to reduce pair explosion (70K SKUs → no all-pairs):
  brand + category + rough price band, or normalized quantity, as blocking
  keys.
- Text embeddings (TF-IDF or a lightweight sentence embedding) on
  sku_name_eng + description as fallback signal when structured attributes are
  missing.

## 4. Matching approach

- **Blocking** → generate only plausible candidate pairs (same
  brand/category cluster).
- **Pairwise similarity model**: combine cosine similarity on text embeddings
  + exact/fuzzy match on structured attributes (volume, flavor, pack type) +
  brand match into a feature vector.
- Train a simple classifier (logistic regression / gradient boosting) on
  GTIN-labelled pairs (same GTIN = positive pair, different GTIN within same
  block = negative) to predict "same item" probability.
- **Clustering**: turn pairwise scores into a graph (edge if score >
  threshold), find connected components / use agglomerative clustering —
  assign ITEM_ID per component. Handles transitivity better than pure
  pairwise thresholding.
- Merge with GTIN-based clusters from Stage A (GTIN clusters are ground truth;
  override predictions where GTIN exists).

## 5. Validation

- Hold out a slice of GTIN-labelled data, hide the GTIN, run the pipeline,
  compare predicted clusters vs true GTIN clusters using **Rand Index**
  (matches the evaluation metric) — internal proxy score before submission.
- Error analysis: false merges (different GTIN merged) vs false splits.

## 6. Deliverables

- EDA section with 3–4 key plots/tables.
- Clear write-up of reasoning (why blocking, why this similarity combo, why
  clustering over pure classification).
- Modular code (functions, not stream-of-consciousness cells), docstrings, a
  config section for thresholds.
- Final output table: SKU_ID, ITEM_ID.
- Short "limitations & next steps" section (e.g., cross-lingual noise, no deep
  learning needed given the labelled subset is small, could scale with a
  Siamese network if more GTIN data existed).

## Deferred decisions

- **Taxi information disposition** — tracked in STATE-20260901-008 (open).
  Full inventory of taxi experiment code/configs/loader/tests recorded there;
  deletion is recoverable from git history. Revisit when taxi surfaces block
  the euromonitor pipeline or the owner calls it.













**Short answer: No — don’t start with a classic ML model (Random Forest / XGBoost / logistic regression on engineered features) as the primary approach.**

Here’s why, ranked by importance:

### 1. The strongest signal is already deterministic
- Same GTIN → same item.  
  This is a hard rule, not something a classifier needs to learn.  
  Starting with classic ML would force you to re-learn something that is already near-perfect and would introduce unnecessary risk of splitting true GTIN groups.

### 2. Classic supervised ML needs good labels
- You only have partial GTIN coverage.  
- The GTIN groups themselves contain some conflicts (the volume inconsistencies you already found).  
- You do **not** have a large, clean set of positive/negative pairs labeled by humans.  
  Training a pairwise classifier or a multi-class item classifier on noisy/partial labels usually underperforms a well-designed embedding + blocking approach on this kind of data.

### 3. Embeddings are the modern “classic” for this exact problem
- Product matching / entity resolution at this scale is almost always solved today with:
  1. Blocking
  2. Dense embeddings (sentence-transformers)
  3. Lightweight scoring + connected components

  This is simpler, more robust to noisy text, and handles multilingual retailer names better than hand-crafted features + XGBoost.

### 4. When classic ML *does* make sense (later, not first)
You can still use it usefully in a second stage:
- Train a small pairwise classifier **on top of** the embedding cosine + rule features (brand match, volume match, name Jaccard, etc.) to re-rank candidate pairs.
- Or use it only for conflict resolution inside GTIN groups.

But this should come **after** you already have a strong GTIN + embedding baseline.

### Recommended order of attack

1. **GTIN-first grouping** + conflict detection (do this today)
2. **Strong volume standardization** (you already have the regex foundation)
3. **Embedding-based residual matching** (`multilingual-e5-small` or MiniLM + blocking + connected components)
4. Only then, if the residual error is still high, try a classic ML pairwise re-ranker or a lightweight fine-tuned cross-encoder.

---

**Bottom line:**  
Starting with classic ML is the slower and riskier path here.  
Start with the hybrid deterministic + embedding approach. It will get you a strong, explainable baseline much faster and is what the evaluation criteria (reasoning + feature engineering + results) actually reward.

Want me to sketch the exact first notebook cells for the GTIN + embedding baseline so you can run it immediately?