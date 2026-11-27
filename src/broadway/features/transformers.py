"""The single preprocessing owner: fitted transformers + typed recipe steps.

Two layers live here, both fit on the train split only:

* **Estimators** (sklearn ``fit``/``transform``): the frequency encoder,
  missing-value imputation (median/mean/mode/KNN), column scaling
  (standard/minmax/robust, delegating the arithmetic to sklearn's canonical
  scalers), winsorization, and the target transforms (order-``k``
  differencing and BoxCox) that carry an exact inverse.
* **The step vocabulary** (``STEP_PARAMS``) + ``PreprocessingFactory``: the one
  place a ``PreprocessingStepConfig`` is validated and turned into an
  estimator. ``broadway.features.recipe`` composes pipelines through it;
  ``broadway.features.engineering`` delegates its tabular adapters to these
  estimators instead of keeping a parallel registry.

Transform always returns a copy — inputs are never mutated.
"""

from __future__ import annotations

from typing import Any, ClassVar, Protocol, Self

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype
from pydantic import BaseModel, ConfigDict, field_validator
from scipy.special import inv_boxcox
from scipy.stats import boxcox, boxcox_normmax
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, RobustScaler, StandardScaler

from broadway.config.schema import (
    ImputationStrategy,
    PreprocessingStepConfig,
    ScalingStrategy,
    StepType,
)


def _key_series(frame: pd.DataFrame, columns: list[str]) -> pd.Series:
    """Per-row category key: the single column value, or a tuple across columns."""
    if len(columns) == 1:
        return frame[columns[0]]
    return pd.Series(list(zip(*[frame[col] for col in columns])), index=frame.index)


class FrequencyEncoding(BaseEstimator, TransformerMixin):
    """Relative-frequency encoding keyed by one categorical column or a composite.

    Fit computes ``value_counts`` per key — normalized by default (identical to
    the legacy ``fit_frequency_encoding``); pass ``normalize=False`` for raw
    counts (a project binding's frequency-encoded route). Unseen/NaN keys at
    transform map to the ``fill`` value (default 0, matching the legacy
    signature). The output column is named ``feature_name`` when provided
    (verbatim), else ``<joined_cols>_freq_enc``.
    """

    def __init__(
        self,
        columns: list[str],
        feature_name: str | None = None,
        normalize: bool = True,
    ) -> None:
        self.columns = list(columns)
        self.feature_name = feature_name
        self.normalize = normalize

    def __sklearn_clone__(self) -> FrequencyEncoding:
        return type(self)(**self.get_params(deep=False))

    def fit(self, X: pd.DataFrame, y=None) -> FrequencyEncoding:
        key = self.columns if len(self.columns) > 1 else self.columns[0]
        self.mapping_ = X[key].value_counts(normalize=self.normalize).to_dict()
        self.column_name_ = (
            self.feature_name
            if self.feature_name is not None
            else f"{'_'.join(self.columns)}_freq_enc"
        )
        return self

    def transform(self, X: pd.DataFrame, fill: float = 0) -> pd.DataFrame:
        values = _key_series(X, self.columns).map(self.mapping_).fillna(fill)
        result = X.copy()
        result[self.column_name_] = values
        return result


# --------------------------------------------------------------------------- #
# Imputation — fit on train, apply to any frame
# --------------------------------------------------------------------------- #
_SIMPLE_STRATEGIES: dict[ImputationStrategy, str] = {
    ImputationStrategy.MEDIAN: "median",
    ImputationStrategy.MEAN: "mean",
    ImputationStrategy.MODE: "most_frequent",
}


class FeatureImputer(BaseEstimator, TransformerMixin):
    """Fill missing values with TRAIN statistics (median/mean/mode/KNN).

    ``median``/``mean``/``mode`` are fitted per column, so a mixed numeric +
    categorical block keeps each column's own dtype and statistic. ``knn`` is
    the one multivariate strategy (donors are found across all listed columns)
    and is therefore numeric-only.

    Unseen-category semantics (``mode`` only): a *non-numeric* value the fit
    frame never observed resolves to that column's fitted mode — and unseen
    numeric values are left untouched (a new number is data, not an unknown
    category). A column whose train split observed no value fails loud: an
    all-NaN column must not silently keep its NaNs.
    """

    def __init__(
        self,
        columns: list[str],
        strategy: ImputationStrategy,
        knn_neighbors: int = 5,
    ) -> None:
        self.columns = columns
        self.strategy = strategy
        self.knn_neighbors = knn_neighbors

    def fit(self, X: pd.DataFrame, y=None) -> FeatureImputer:
        self._require_columns(X)
        strategy = ImputationStrategy(self.strategy)
        unobserved = [column for column in self.columns if not X[column].notna().any()]
        if unobserved:
            raise ValueError(
                f"column(s) {unobserved} have no observed values in the fit frame — "
                f"imputation strategy '{strategy.value}' cannot be fitted"
            )
        self.imputers_: dict[str, SimpleImputer] = {}
        self.knn_: KNNImputer | None = None
        self.statistics_: dict[str, Any] = {}
        self.categories_: dict[str, frozenset[Any]] = {}
        if strategy is ImputationStrategy.KNN:
            non_numeric = [c for c in self.columns if not is_numeric_dtype(X[c])]
            if non_numeric:
                raise ValueError(f"knn imputation requires numeric columns; got {non_numeric}")
            self.knn_ = KNNImputer(n_neighbors=self.knn_neighbors).fit(X[self.columns])
        else:
            for column in self.columns:
                imputer = SimpleImputer(strategy=_SIMPLE_STRATEGIES[strategy]).fit(X[[column]])
                self.imputers_[column] = imputer
                self.statistics_[column] = imputer.statistics_[0]
                if strategy is ImputationStrategy.MODE and not is_numeric_dtype(X[column]):
                    self.categories_[column] = frozenset(X[column].dropna().unique())
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not hasattr(self, "imputers_"):
            raise ValueError("FeatureImputer.transform called before fit")
        self._require_columns(X)
        result = X.copy()
        if self.knn_ is not None:
            result[self.columns] = self.knn_.transform(result[self.columns])
            return result
        for column in self.columns:
            result[column] = self.imputers_[column].transform(result[[column]])[:, 0]
            if column in self.categories_:
                result[column] = result[column].where(
                    result[column].isin(self.categories_[column]), self.statistics_[column]
                )
        return result

    def _require_columns(self, X: pd.DataFrame) -> None:
        if not self.columns:
            raise ValueError("FeatureImputer requires at least one column")
        missing = [column for column in self.columns if column not in X.columns]
        if missing:
            raise ValueError(f"FeatureImputer: missing columns {missing}")


# --------------------------------------------------------------------------- #
# Scaling — fit on train, invertible
# --------------------------------------------------------------------------- #
# The one strategy vocabulary -> sklearn's canonical scaler (also read by the
# darts engine adapter, which needs the class rather than the wrapper).
SCALERS: dict[ScalingStrategy, type] = {
    ScalingStrategy.STANDARD: StandardScaler,
    ScalingStrategy.MINMAX: MinMaxScaler,
    ScalingStrategy.ROBUST: RobustScaler,
}
SCALER_BY_CLASS_NAME: dict[str, type] = {cls.__name__: cls for cls in SCALERS.values()}


class FeatureScaler(BaseEstimator, TransformerMixin):
    """Column-subset scaling over sklearn's Standard/MinMax/Robust scalers.

    The wrapper contributes the declared column list, the copy transform, and
    the single strategy vocabulary; the statistics and arithmetic stay with
    sklearn (fit on the train split only). ``inverse_transform`` restores the
    original units, which is what a scaled target needs.
    """

    def __init__(self, columns: list[str], strategy: ScalingStrategy) -> None:
        self.columns = columns
        self.strategy = strategy

    def fit(self, X: pd.DataFrame, y=None) -> FeatureScaler:
        self._require_columns(X)
        self.scaler_ = SCALERS[ScalingStrategy(self.strategy)]().fit(X[self.columns])
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        self._require_fitted()
        result = X.copy()
        result[self.columns] = self.scaler_.transform(result[self.columns])
        return result

    def inverse_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        self._require_fitted()
        result = X.copy()
        result[self.columns] = self.scaler_.inverse_transform(result[self.columns])
        return result

    def _require_fitted(self) -> None:
        if not hasattr(self, "scaler_"):
            raise ValueError("FeatureScaler called before fit")

    def _require_columns(self, X: pd.DataFrame) -> None:
        if not self.columns:
            raise ValueError("FeatureScaler requires at least one column")
        missing = [column for column in self.columns if column not in X.columns]
        if missing:
            raise ValueError(f"FeatureScaler: missing columns {missing}")


# --------------------------------------------------------------------------- #
# Winsorizing — one semantics: explicit train-split quantiles
# --------------------------------------------------------------------------- #
class Winsorizer(BaseEstimator, TransformerMixin):
    """Clip each column to fitted quantile bounds (one tail or both).

    Bounds come from the fit split only. Clipping destroys the original values,
    so there is no inverse — that is the caller's accepted trade.
    """

    def __init__(
        self,
        columns: list[str],
        lower_quantile: float | None = None,
        upper_quantile: float | None = None,
    ) -> None:
        self.columns = columns
        self.lower_quantile = lower_quantile
        self.upper_quantile = upper_quantile

    def fit(self, X: pd.DataFrame, y=None) -> Winsorizer:
        missing = [column for column in self.columns if column not in X.columns]
        if missing:
            raise ValueError(f"Winsorizer: missing columns {missing}")
        if self.lower_quantile is None and self.upper_quantile is None:
            raise ValueError("Winsorizer requires at least one quantile bound")
        for name, quantile in (
            ("lower_quantile", self.lower_quantile),
            ("upper_quantile", self.upper_quantile),
        ):
            if quantile is not None and not 0.0 < quantile < 1.0:
                raise ValueError(f"Winsorizer {name} must be in (0, 1), got {quantile}")
        self.bounds_: dict[str, tuple[float | None, float | None]] = {
            column: (
                self._bound(X[column], self.lower_quantile),
                self._bound(X[column], self.upper_quantile),
            )
            for column in self.columns
        }
        return self

    @staticmethod
    def _bound(series: pd.Series, quantile: float | None) -> float | None:
        return None if quantile is None else float(series.quantile(quantile))

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not hasattr(self, "bounds_"):
            raise ValueError("Winsorizer.transform called before fit")
        result = X.copy()
        for column, (lower, upper) in self.bounds_.items():
            result[column] = result[column].clip(lower=lower, upper=upper)
        return result


# --------------------------------------------------------------------------- #
# Target transforms — invertible, so a prediction can return to target space
# --------------------------------------------------------------------------- #
class TargetTransform(Protocol):
    """A fitted, invertible target transform."""

    @property
    def n_dropped(self) -> int:
        """Rows the transform consumes at the head of the series."""
        ...

    def fit(self, y: pd.Series) -> Self: ...

    def transform(self, y: pd.Series) -> pd.Series: ...

    def inverse_transform(self, values: pd.Series) -> pd.Series: ...


class TargetDifference:
    """Order-``k`` differencing with the exact inverse from ``stats.time_series``.

    Boundaries come from the series being differenced (its own observed past),
    so ``transform`` seeds itself per split: the train split's seeds invert the
    train target, the validation split's seeds invert the validation target.
    ``transform`` returns a shorter series indexed by the surviving tail rows;
    the caller aligns the feature frame with ``y.index[order:]``.
    """

    def __init__(self, order: int) -> None:
        if order < 1:
            raise ValueError(f"difference order must be >= 1, got {order}")
        self.order = order

    @property
    def n_dropped(self) -> int:
        return self.order

    def fit(self, y: pd.Series) -> TargetDifference:
        if len(y) <= self.order:
            raise ValueError(
                f"target has {len(y)} observations; difference order {self.order} "
                "leaves nothing to fit"
            )
        return self

    def transform(self, y: pd.Series) -> pd.Series:
        from broadway.stats.time_series import difference, first_values

        values = y.to_numpy(dtype=float)
        self.firsts_ = first_values(values, self.order)
        self.original_index_ = y.index
        return pd.Series(
            difference(values, self.order), index=y.index[self.order :], name=y.name
        )

    def inverse_transform(self, values: pd.Series) -> pd.Series:
        from broadway.stats.time_series import inverse_difference

        if not hasattr(self, "firsts_"):
            raise ValueError("TargetDifference.inverse_transform called before transform")
        restored = inverse_difference(np.asarray(values, dtype=float), self.firsts_, self.order)
        return pd.Series(restored, index=self.original_index_)


class TargetBoxCox:
    """BoxCox power transform, invertible through ``scipy.special.inv_boxcox``.

    ``lmbda`` is taken from config when set, otherwise estimated by maximum
    likelihood on the fit split only. The transform requires strictly positive
    observations (BoxCox's domain) and fails loud otherwise.
    """

    def __init__(self, lmbda: float | None = None) -> None:
        self.lmbda = lmbda

    @property
    def n_dropped(self) -> int:
        return 0

    def fit(self, y: pd.Series) -> TargetBoxCox:
        values = self._positive(y)
        self.lmbda_ = (
            float(self.lmbda)
            if self.lmbda is not None
            else float(boxcox_normmax(values, method="mle"))
        )
        return self

    def transform(self, y: pd.Series) -> pd.Series:
        if not hasattr(self, "lmbda_"):
            raise ValueError("TargetBoxCox.transform called before fit")
        self.original_index_ = y.index
        return pd.Series(boxcox(self._positive(y), lmbda=self.lmbda_), index=y.index, name=y.name)

    def inverse_transform(self, values: pd.Series) -> pd.Series:
        if not hasattr(self, "lmbda_"):
            raise ValueError("TargetBoxCox.inverse_transform called before fit")
        if not hasattr(self, "original_index_"):
            raise ValueError("TargetBoxCox.inverse_transform called before transform")
        restored = inv_boxcox(np.asarray(values, dtype=float), self.lmbda_)
        return pd.Series(restored, index=self.original_index_)

    @staticmethod
    def _positive(y: pd.Series) -> np.ndarray:
        values = y.to_numpy(dtype=float)
        if not np.all(np.isfinite(values)):
            raise ValueError("TargetBoxCox requires finite observations")
        if not np.all(values > 0.0):
            raise ValueError("TargetBoxCox requires strictly positive observations")
        return values


class TargetTransformChain:
    """Apply the configured target transforms in order, and invert them.

    ``fit`` fits each transform on its intermediate series (BoxCox's lambda is
    therefore estimated on what it actually sees); ``transform`` reuses those
    fitted parameters but lets ``TargetDifference`` re-seed from the incoming
    series. ``inverse_transform`` walks the chain backwards into target units.
    """

    def __init__(self, transforms: list[TargetTransform]) -> None:
        if not transforms:
            raise ValueError("TargetTransformChain requires at least one transform")
        self.transforms = transforms

    @property
    def n_dropped(self) -> int:
        return sum(transform.n_dropped for transform in self.transforms)

    def fit(self, y: pd.Series) -> TargetTransformChain:
        self._apply(y, fitting=True)
        return self

    def transform(self, y: pd.Series) -> pd.Series:
        return self._apply(y, fitting=False)

    def inverse_transform(self, values: pd.Series) -> pd.Series:
        restored = values
        for transform in reversed(self.transforms):
            restored = transform.inverse_transform(restored)
        return restored

    def _apply(self, y: pd.Series, *, fitting: bool) -> pd.Series:
        values = y
        for transform in self.transforms:
            if fitting:
                transform.fit(values)
            values = transform.transform(values)
        return values


# --------------------------------------------------------------------------- #
# Typed step params + the one factory
# --------------------------------------------------------------------------- #
class StepParams(BaseModel):
    """Typed params of one recipe step — and the one place it is constructed.

    Each model declares its step type once (``step_type``), so the vocabulary,
    the param validity, and the estimator construction live in a single
    declaration rather than in parallel registries and dispatch branches.
    """

    model_config = ConfigDict(extra="forbid")

    step_type: ClassVar[StepType]

    def build_feature_step(self, columns: list[str]) -> object:
        """Build an X-transformer step; target-only steps refuse loudly."""
        raise ValueError(
            f"preprocessing step '{self.step_type.value}' changes the target — "
            "declare it under target_transforms"
        )

    def build_target_transform(self) -> TargetTransform:
        """Build a target transform; feature-frame steps refuse loudly."""
        raise ValueError(
            f"preprocessing step '{self.step_type.value}' transforms the feature frame — "
            "declare it under preprocessing"
        )


class FrequencyEncodingParams(StepParams):
    step_type: ClassVar[StepType] = StepType.FREQUENCY_ENCODING
    normalize: bool = True

    @field_validator("normalize", mode="before")
    @classmethod
    def _coerce_bool(cls, value: object) -> bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, str) and value.lower() in {"true", "false"}:
            return value.lower() == "true"
        raise ValueError(f"normalize must be a bool or 'true'/'false', got {value!r}")

    def build_feature_step(self, columns: list[str]) -> object:
        return FrequencyEncoding(columns=columns, normalize=self.normalize)


class PassthroughParams(StepParams):
    """No params: the identity step."""

    step_type: ClassVar[StepType] = StepType.PASSTHROUGH

    def build_feature_step(self, columns: list[str]) -> object:
        return "passthrough"


class OneHotParams(StepParams):
    """No params: ``handle_unknown='ignore'`` is fixed by the recipe policy."""

    step_type: ClassVar[StepType] = StepType.ONE_HOT

    def build_feature_step(self, columns: list[str]) -> object:
        # upgrade path: set_output(transform="pandas") lifts the sole-step
        # restriction enforced in broadway.features.recipe.
        return ColumnTransformer(
            [("one_hot", OneHotEncoder(handle_unknown="ignore"), columns)],
            remainder="passthrough",
        )


class ImputationParams(StepParams):
    step_type: ClassVar[StepType] = StepType.IMPUTATION
    strategy: ImputationStrategy
    knn_neighbors: int = 5

    def build_feature_step(self, columns: list[str]) -> object:
        return FeatureImputer(
            columns=columns, strategy=self.strategy, knn_neighbors=self.knn_neighbors
        )


class ScalingParams(StepParams):
    step_type: ClassVar[StepType] = StepType.SCALING
    strategy: ScalingStrategy

    def build_feature_step(self, columns: list[str]) -> object:
        return FeatureScaler(columns=columns, strategy=self.strategy)


class DifferenceParams(StepParams):
    step_type: ClassVar[StepType] = StepType.DIFFERENCE
    order: int

    def build_target_transform(self) -> TargetTransform:
        return TargetDifference(order=self.order)


class BoxCoxParams(StepParams):
    step_type: ClassVar[StepType] = StepType.BOXCOX
    lmbda: float | None = None

    def build_target_transform(self) -> TargetTransform:
        return TargetBoxCox(lmbda=self.lmbda)


# The single preprocessing-step registry, derived from the declarations above:
# type -> typed params model. The recipe builds through this; no caller keeps
# its own allowlist.
STEP_PARAM_MODELS: tuple[type[StepParams], ...] = (
    FrequencyEncodingParams,
    OneHotParams,
    PassthroughParams,
    ImputationParams,
    ScalingParams,
    DifferenceParams,
    BoxCoxParams,
)
STEP_PARAMS: dict[StepType, type[StepParams]] = {
    model.step_type: model for model in STEP_PARAM_MODELS
}


class PreprocessingFactory:
    """Validate one recipe step's params and build its estimator."""

    @classmethod
    def parse(cls, step: PreprocessingStepConfig) -> StepParams:
        """Validate the step's params against its typed model, then parse."""
        step_type = StepType(step.type)
        spec = STEP_PARAMS[step_type]
        allowed = set(spec.model_fields)
        invalid = sorted(set(step.params) - allowed)
        if invalid:
            raise ValueError(
                f"invalid params for preprocessing step '{step_type.value}': {invalid}. "
                f"valid params: {sorted(allowed)}"
            )
        required = {name for name, field in spec.model_fields.items() if field.is_required()}
        missing = sorted(required - set(step.params))
        if missing:
            raise ValueError(
                f"preprocessing step '{step_type.value}' requires param(s) {missing}"
            )
        return spec.model_validate(step.params)

    @classmethod
    def build(cls, step: PreprocessingStepConfig) -> object:
        """Build an X-transformer step for the sklearn recipe."""
        return cls.parse(step).build_feature_step(step.columns)

    @classmethod
    def build_target_transform(cls, step: PreprocessingStepConfig) -> TargetTransform:
        """Build a target transform step for the recipe's target chain."""
        return cls.parse(step).build_target_transform()
