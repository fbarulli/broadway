from __future__ import annotations

from enum import Enum, StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    PrivateAttr,
    field_validator,
    model_validator,
)

from broadway.analysis.contracts import AnalysisContract, AnalysisMode
from broadway.calibration.contracts import CalibrationConfig
from broadway.evaluate.contracts import DecisionConfig, ErrorAnalysisConfig
from broadway.training.models.registry import allowed_params

if TYPE_CHECKING:
    from broadway.features.plan import ColumnPlan


class TaskType(str, Enum):
    REGRESSION = "regression"
    CLASSIFICATION = "classification"


class ColumnRole(str, Enum):
    FEATURE = "feature"
    TARGET = "target"
    DATETIME = "datetime"
    IGNORE = "ignore"


def normalize_dtype(dtype: str) -> str:
    return "datetime64" if dtype.startswith("datetime64") else dtype


class ColumnSchema(BaseModel):
    dtype: str
    null_count: int
    role: ColumnRole

    @field_validator("dtype", mode="before")
    @classmethod
    def _normalize_dtype(cls, v: object) -> object:
        return normalize_dtype(v) if isinstance(v, str) else v


class LookupValuePolicy(BaseModel):
    model_config = ConfigDict(extra="forbid")
    sentinel_values: list[str] = []


class LookupSpec(BaseModel):
    path: str
    key: str
    value_policies: dict[str, LookupValuePolicy] = {}
    na_values: list[str] = []


class DatasetContract(BaseModel):
    name: str
    path: str
    target: str
    task: TaskType
    datetime_column: str | None
    columns: dict[str, ColumnSchema]
    lookup_tables: dict[str, LookupSpec]
    exclude_from_profiling: list[str] = []


class EnvironmentConfig(BaseModel):
    log_level: str
    log_dir: str | None = None
    log_file: str | None = None
    data_dir: str
    raw_subdir: str
    processed_subdir: str
    mlflow_tracking_uri: str
    database_user: str
    database_password: str
    database_name: str
    database_host: str
    database_port: int
    sample_size_ci: int
    sample_size_stats: int
    api_replicas_min: int
    api_replicas_max: int
    api_hpa_cpu_threshold: int


class DerivedFeature(BaseModel):
    name: str
    func: str
    source: str


class ModelConfig(BaseModel):
    type: str
    params: dict[str, float | int | str]


class SplitConfig(BaseModel):
    type: Literal["time", "random", "stratified"]
    validation_size: float = Field(ge=0.0, le=1.0)


class ModelHPOSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str
    search_space: dict[str, list[float | int]] = {}


class HPOConfig(BaseModel):
    engine: str
    direction: str = "minimize"
    total_trials: int
    initial_trials_per_model: int
    top_k: int
    target_metric: str
    models: list[ModelHPOSpec]
    storage_url: str | None = None


class DataSourceRef(BaseModel):
    """The experiment's declared data source — a required, typed config field.

    ``loader`` names the loader that resolves the experiment's data
    (``canonical`` parquet, ``joined`` cache, ``named_sample`` @version,
    ``pinned`` artifact); ``version`` is the immutable sample version and is
    required exactly when ``loader == "named_sample"``; ``schema_contract``
    names the schema module the source is bound to.
    """

    model_config = ConfigDict(extra="forbid")
    loader: Literal["canonical", "joined", "named_sample", "pinned"]
    version: str | None = None
    schema_contract: str

    @model_validator(mode="after")
    def _validate_version_rule(self) -> DataSourceRef:
        if self.loader == "named_sample" and self.version is None:
            raise ValueError(
                "data_source.loader='named_sample' requires data_source.version "
                "(the immutable sample version, e.g. 'v3')"
            )
        if self.loader != "named_sample" and self.version is not None:
            raise ValueError(
                f"data_source.version is only valid for loader='named_sample'; "
                f"got version={self.version!r} for loader={self.loader!r}"
            )
        return self


class StepType(StrEnum):
    """Preprocessing step vocabulary — the single source every layer reads.

    ``preprocessing`` carries the X-transformer steps; ``target_transforms``
    carries the target steps (``difference``/``boxcox``) whose inverse is owned
    by ``broadway.features.transformers``.
    """

    TARGET_ENCODING = "target_encoding"
    FREQUENCY_ENCODING = "frequency_encoding"
    ONE_HOT = "one_hot"
    PASSTHROUGH = "passthrough"
    IMPUTATION = "imputation"
    SCALING = "scaling"
    DIFFERENCE = "difference"
    BOXCOX = "boxcox"


class ImputationStrategy(StrEnum):
    """Fitted imputation strategies (the owner lives in features.transformers)."""

    MEDIAN = "median"
    MEAN = "mean"
    MODE = "mode"
    KNN = "knn"


class ScalingStrategy(StrEnum):
    """Column-wise scaling strategies (the owner lives in features.transformers)."""

    STANDARD = "standard"
    MINMAX = "minmax"
    ROBUST = "robust"


# Step placement: differencing/BoxCox change the target, everything else
# transforms the feature frame. Declaring one in the wrong list fails at parse.
TARGET_TRANSFORM_STEP_TYPES: frozenset[StepType] = frozenset(
    {StepType.DIFFERENCE, StepType.BOXCOX}
)


class PreprocessingStepConfig(BaseModel):
    """One ordered preprocessing step of an experiment's pipeline recipe.

    ``type`` names a step kind from the ``StepType`` vocabulary; ``columns`` is
    the name-driven column list enforced against the bound schema contract;
    ``params`` carries the step's typed tuning values (e.g. ``smoothing``),
    validated by the owner in ``broadway.features.transformers``.
    """

    model_config = ConfigDict(extra="forbid")
    type: StepType
    columns: list[str]
    params: dict[str, float | int | str | bool] = {}


# Prefix marking HPO search-space params that tune preprocessing (pre__<step>__<param>)
# rather than the registry-validated model params. Single source: schema.py owns the
# constant, trainer.py imports it — never reversed.
PRE_PARAM_PREFIX = "pre__"


class ExperimentConfig(BaseModel):
    data_source: DataSourceRef
    feature_plan: str
    model: ModelConfig
    split: SplitConfig
    random_state: int
    target_metric: str
    hpo: HPOConfig | None = None
    preprocessing: list[PreprocessingStepConfig] = []
    target_transforms: list[PreprocessingStepConfig] = []
    calibration: CalibrationConfig | None = None

    # Resolved by the loader from ``feature_plan`` + the dataset contract.
    _plan: ColumnPlan | None = PrivateAttr(default=None)

    @field_validator("feature_plan")
    @classmethod
    def _validate_plan_reference(cls, value: str) -> str:
        module, separator, attribute = value.partition(":")
        if not separator or not module or not attribute:
            raise ValueError(
                "feature_plan must be a 'module:callable' reference "
                f"(e.g. 'broadway.features.plans:baseline'), got {value!r}"
            )
        return value

    @property
    def plan(self) -> ColumnPlan:
        """The resolved column plan; unbound plans fail loud at the read boundary."""
        if self._plan is None:
            raise ValueError(
                f"experiment feature_plan '{self.feature_plan}' is not resolved — "
                "load the config through broadway.config.loader (or bind the dataset plan)"
            )
        return self._plan

    def bind_plan(self, plan: ColumnPlan) -> ExperimentConfig:
        """Attach the plan resolved from this experiment's code reference."""
        self._plan = plan
        return self

    @model_validator(mode="after")
    def _validate_step_placement(self) -> ExperimentConfig:
        """Refuse target steps in the X recipe and vice versa (loud, at parse)."""
        misplaced = [
            step.type.value
            for step in self.preprocessing
            if step.type in TARGET_TRANSFORM_STEP_TYPES
        ]
        if misplaced:
            raise ValueError(
                f"target transform step(s) {misplaced} belong in 'target_transforms', "
                "not 'preprocessing': they change the target, not the feature frame"
            )
        misplaced = [
            step.type.value
            for step in self.target_transforms
            if step.type not in TARGET_TRANSFORM_STEP_TYPES
        ]
        if misplaced:
            raise ValueError(
                f"feature step(s) {misplaced} belong in 'preprocessing', not "
                "'target_transforms': only difference/boxcox change the target"
            )
        return self

    @model_validator(mode="after")
    def _validate_hpo_search_space(self) -> ExperimentConfig:
        if self.hpo is None:
            return self
        for spec in self.hpo.models:
            valid_params = allowed_params(spec.name)
            model_params = {
                key: value
                for key, value in spec.search_space.items()
                if not key.startswith(PRE_PARAM_PREFIX)
            }
            invalid_params = set(model_params) - valid_params
            if invalid_params:
                raise ValueError(
                    f"invalid HPO search-space params for model '{spec.name}': "
                    f"{sorted(invalid_params)}. valid params: {sorted(valid_params)} "
                    f"({PRE_PARAM_PREFIX} params tune preprocessing and are not registry-validated)"
                )
        return self


class DiscoverStep(BaseModel):
    csv_path: str
    target_column: str
    task: TaskType
    datetime_column: str | None
    ignore_columns: list[str]


class EtlStep(BaseModel):
    ci_sample_size: int = Field(ge=0)
    random_state: int
    train_file: str
    val_file: str
    training_data_file: str
    train_features_file: str
    val_features_file: str
    max_drop_fraction: float = Field(ge=0.0, le=1.0)
    missing_encodings: list[str]


class ContractsStep(BaseModel):
    null_threshold: float


class EdaStep(BaseModel):
    """Exploratory-data-analysis knobs; every threshold lives in the config.

    ``sample_size`` of 0 means "all rows"; ``plot_format`` selects the image
    encoding embedded in the self-contained HTML report (plotly is not a
    dependency — matplotlib renders the images).
    """

    model_config = ConfigDict(extra="forbid")

    sample_size: int = Field(ge=0)
    plot_format: Literal["png", "svg"]
    output_dir: str
    output_file: str
    max_categories: int = Field(ge=1)
    max_plots: int = Field(ge=1)
    iqr_multiplier: float = Field(gt=0.0)
    z_threshold: float = Field(gt=0.0)
    imbalance_min_share: float = Field(gt=0.0, le=1.0)

    @property
    def output_path(self) -> Path:
        """Report path (``output_dir/output_file``)."""
        return Path(self.output_dir) / self.output_file


class MissingInjectionConfig(BaseModel):
    """Train-only MCAR missingness injection for the imputation impact study.

    A churn rate the study applies to the named columns of the TRAIN split; the
    validation split stays clean so every strategy is scored on the same rows.
    """

    model_config = ConfigDict(extra="forbid")

    rate: float = Field(gt=0.0, lt=1.0)
    columns: list[str] = Field(min_length=1)


class ImputationStudyConfig(BaseModel):
    """Measured imputation/scaling impact study — reporting, never a gate.

    ``strategies`` are imputed per fit on the train split; ``complete_case`` is
    always the per-scaling baseline (rows with any missing feature are dropped).
    ``sample_size`` of 0 means every row; a bounded sample keeps the KNN
    strategy's pairwise-distance cost finite.
    """

    model_config = ConfigDict(extra="forbid")

    strategies: list[ImputationStrategy] = Field(min_length=1)
    scalings: list[ScalingStrategy | Literal["none"]] = Field(min_length=1)
    metric_direction: Literal["minimize", "maximize"]
    sample_size: int = Field(ge=0)
    knn_neighbors: int = Field(ge=1)
    missing_injection: MissingInjectionConfig
    output_dir: str
    output_file: str

    @field_validator("strategies")
    @classmethod
    def _unique_strategies(
        cls, strategies: list[ImputationStrategy]
    ) -> list[ImputationStrategy]:
        if len(set(strategies)) != len(strategies):
            raise ValueError(f"imputation strategy list has duplicates: {strategies}")
        return strategies

    @property
    def output_path(self) -> Path:
        """Delta-table path (``output_dir/output_file``)."""
        return Path(self.output_dir) / self.output_file


class FeaturesStep(BaseModel):
    """Features-step knobs; the optional impact study writes its own table."""

    model_config = ConfigDict(extra="forbid")

    encoding_smoothing: int
    frequency_fill: float
    pipeline_file: str
    max_drop_fraction: float = Field(ge=0.0, le=1.0)
    imputation_study: ImputationStudyConfig | None = None


class StatsStep(BaseModel):
    """Descriptive/time-series knobs; every numeric is validated at the boundary.

    ``decompose_model`` mirrors ``broadway.stats.decompose.DecomposeModel`` as a
    Literal on purpose: importing the stats module here would pull statsmodels
    onto the CLI's config-load path. The canonical cast happens at the caller
    (``DecomposeModel(config.decompose_model)``), so a drift fails loudly.
    """

    model_config = ConfigDict(extra="forbid")

    sample_fraction: float
    output_dir: str
    output_file: str
    min_rows_for_sampling: int
    per_group_sample_fraction: float
    time_slice_start: str
    time_slice_end: str
    time_split_cutoff: str
    acf_lags: int = Field(ge=0)
    decompose_period: int = Field(ge=2)
    decompose_model: Literal["additive", "multiplicative"]
    decompose_lowess_frac: float = Field(gt=0.0, le=1.0)
    changepoint_threshold: float = Field(gt=0.0)
    changepoint_min_size: int = Field(ge=1)
    sample_size_dev: int
    sample_size_live: int
    time_slice_start_dev: str
    time_slice_end_dev: str
    time_slice_start_live: str
    time_slice_end_live: str


class CausalStep(BaseModel):
    treatment_column: str
    outcome_column: str
    power: float
    alpha: float
    effect_size: float = Field(gt=0.0)
    output_dir: str
    output_file: str


class LearningCurveConfig(BaseModel):
    """Training-size curve knobs: metric vs fraction of the training frame.

    ``fractions`` must include 1.0 (the full-data reference). The tolerance
    band is ``tolerance * |reference|`` with ``absolute_tolerance`` as a floor,
    so a zero reference does not collapse it to exact equality. ``direction``
    says which way the metric improves; the minimum-data answer is the
    smallest fraction inside the band. ``metric_threshold`` is the binary
    target split used by classification metrics (``roc_auc``/``pr_auc``) and
    is required for a classification task.
    """

    model_config = ConfigDict(extra="forbid")

    fractions: list[float]
    repeats: int = Field(ge=1)
    direction: Literal["minimize", "maximize"]
    tolerance: float = Field(ge=0.0)
    absolute_tolerance: float = Field(ge=0.0)
    metric_threshold: float | None = None
    output_dir: str
    output_file: str

    @field_validator("fractions")
    @classmethod
    def _validate_fractions(cls, values: list[float]) -> list[float]:
        if not values:
            raise ValueError("learning_curve.fractions must not be empty")
        if len(set(values)) != len(values):
            raise ValueError(f"learning_curve.fractions has duplicates: {values}")
        invalid = [value for value in values if not 0.0 < value <= 1.0]
        if invalid:
            raise ValueError(
                f"learning_curve.fractions must be in (0, 1]; got {invalid}"
            )
        if 1.0 not in values:
            raise ValueError(
                "learning_curve.fractions must include 1.0 (the full-data reference)"
            )
        return values

    @property
    def output_path(self) -> Path:
        """JSON result path (``output_dir/output_file``)."""
        return Path(self.output_dir) / self.output_file

    @property
    def csv_path(self) -> Path:
        """CSV table path: same stem as the JSON result."""
        return self.output_path.with_suffix(".csv")

    @property
    def png_path(self) -> Path:
        """PNG plot path: same stem as the JSON result."""
        return self.output_path.with_suffix(".png")


class TrackingBaselineConfig(BaseModel):
    """Train-time baseline writer: output file plus distribution knobs."""

    model_config = ConfigDict(extra="forbid")

    output_file: str
    n_bins: int = Field(ge=1)
    top_k: int = Field(ge=1)
    smoothing: float = Field(ge=0.0)


class TrackingDataCardConfig(BaseModel):
    """Data-card renderer: output file plus presentation knobs."""

    model_config = ConfigDict(extra="forbid")

    output_file: str
    missing_flag_rate: float = Field(ge=0.0, le=1.0)
    biases: list[str] = Field(default_factory=list)
    generated_at: str | None = None


class TrackingDriftConfig(BaseModel):
    """Train-vs-serve drift input.

    No score threshold lives here: thresholds are caller parameters and the
    train hook passes none, so drift is a report input, never a gate.
    """

    model_config = ConfigDict(extra="forbid")

    bins: int = Field(ge=1)
    epsilon: float = Field(gt=0.0)
    serve_frame: str | None = None


class TrackingConfig(BaseModel):
    """The data-tracking bundle: ignored machine truth + tracked report surface."""

    model_config = ConfigDict(extra="forbid")

    machine_dir: str
    reports_dir: str
    profile_file: str
    baseline: TrackingBaselineConfig
    data_card: TrackingDataCardConfig
    drift: TrackingDriftConfig


class TrainStep(BaseModel):
    random_state: int
    n_jobs: int
    cv_folds: int
    cv_kind: Literal["kfold", "time_series_split"]
    model_file: str
    n_estimators: int
    learning_rate: float
    num_leaves: int
    subsample: float
    colsample_bytree: float
    quantile_tail: float
    output_dir: str
    output_file: str
    learning_curve: LearningCurveConfig | None = None
    tracking: TrackingConfig | None = None


class EvaluateStep(BaseModel):
    target_metric: str
    promotion_threshold: float
    output_dir: str
    output_file: str
    coverage_tolerance: float | None = Field(default=None, ge=0.0, le=1.0)
    min_period_rows: int | None = Field(default=None, ge=1)
    error_analysis: ErrorAnalysisConfig = Field(default_factory=ErrorAnalysisConfig)
    decision: DecisionConfig | None = None


class BaselineStep(BaseModel):
    output_dir: str
    output_file: str


class FullStep(BaseModel):
    flows: dict[str, str]

    @model_validator(mode="after")
    def _validate_flow_modes(self) -> FullStep:
        valid = {mode.value for mode in AnalysisMode}
        invalid = sorted(set(self.flows) - valid)
        if invalid:
            raise ValueError(
                f"invalid flow mode(s) {invalid}. valid modes: {sorted(valid)}"
            )
        return self


class FlowConfig(BaseModel):
    steps: list[str]


class PipelineConfig(BaseModel):
    analysis: AnalysisContract | None = None
    dataset: DatasetContract | None = None
    baseline: BaselineStep | None = None
    environment: EnvironmentConfig
    experiment: ExperimentConfig | None = None
    discover: DiscoverStep | None = None
    etl: EtlStep | None = None
    contracts: ContractsStep | None = None
    eda: EdaStep | None = None
    features: FeaturesStep | None = None
    stats: StatsStep | None = None
    causal: CausalStep | None = None
    train: TrainStep | None = None
    evaluate: EvaluateStep | None = None
    full: FullStep | None = None
