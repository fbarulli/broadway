"""_matching.py: TF-IDF cosine title matching for euromonitor resolution.

Single source for the title-similarity machinery (corpus vectorizer + cosine
scoring). Used by 04_tfidf_matching.py for the ground-truth validation and by
the step-03 pairwise classifier later. Regex/text parsing stays in _text.py;
this module is the sklearn side (no regexes here).
"""

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

DEFAULT_MAX_FEATURES = 2000


def build_vectorizer(max_features: int = DEFAULT_MAX_FEATURES) -> TfidfVectorizer:
    """Corpus vectorizer: (1,2) word ngrams, sublinear TF, English stopwords.

    (1,2) ngrams let "coconut water" and "coconut" overlap even when word
    order/extra words differ; sublinear_tf dampens repeated tokens.
    """
    return TfidfVectorizer(
        stop_words="english", lowercase=True, ngram_range=(1, 2),
        max_features=max_features, sublinear_tf=True,
    )


def score_pairs(X, idx_a, idx_b) -> "list[float]":
    """Cosine similarity for index-aligned pair arrays: X[idx_a[k]] vs
    X[idx_b[k]] for each k. Computed per-pair (L2-normalize then row-wise dot
    product) to avoid materializing the full n x n pairwise matrix."""
    Xa = normalize(X[idx_a], norm="l2")
    Xb = normalize(X[idx_b], norm="l2")
    return np.asarray(Xa.multiply(Xb).sum(axis=1)).ravel().tolist()
