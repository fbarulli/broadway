from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from broadway.analysis.contracts import AnalysisContract, AnalysisMode
from broadway.training.models.registry import allowed_params


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


class EncodingConfig(BaseModel):
    type: str
    columns: list[str]
    smoothing: int | None


class BuilderParams(BaseModel):
    """Declared inputs for multi-input builders (e.g. ``same_group``).
    Absent block -> builders keep their generic-column defaults."""
    group_col: str
    lookup_col: str


class FeatureConfig(BaseModel):
    include: list[str]
    exclude: list[str]
    derived: list[DerivedFeature]
    encodings: list[EncodingConfig]
    builder_module: str | None = None
    builder_params: BuilderParams | None = None


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


class PreprocessingStepConfig(BaseModel):
    """One ordered preprocessing step of an experiment's pipeline recipe.

    ``type`` names the builder-registered step kind (``target_encoding``,
    ``frequency_encoding``, ``one_hot``, ``passthrough``); ``columns`` is the
    name-driven column list enforced against the bound schema contract;
    ``params`` carries step-specific tuning values (e.g. ``smoothing``).
    """

    model_config = ConfigDict(extra="forbid")
    type: Literal["target_encoding", "frequency_encoding", "one_hot", "passthrough"]
    columns: list[str]
    params: dict[str, float | int | str | bool] = {}


# Prefix marking HPO search-space params that tune preprocessing (pre__<step>__<param>)
# rather than the registry-validated model params. Single source: schema.py owns the
# constant, trainer.py imports it — never reversed.
PRE_PARAM_PREFIX = "pre__"


class ExperimentConfig(BaseModel):
    data_source: DataSourceRef
    features: FeatureConfig
    model: ModelConfig
    split: SplitConfig
    random_state: int
    target_metric: str
    hpo: HPOConfig | None = None
    preprocessing: list[PreprocessingStepConfig] = []

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


class FeaturesStep(BaseModel):
    encoding_smoothing: int
    frequency_fill: float
    pipeline_file: str
    max_drop_fraction: float = Field(ge=0.0, le=1.0)


class StatsStep(BaseModel):
    sample_fraction: float
    output_dir: str
    output_file: str
    min_rows_for_sampling: int
    per_group_sample_fraction: float
    time_slice_start: str
    time_slice_end: str
    time_split_cutoff: str
    acf_lags: int
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


class EvaluateStep(BaseModel):
    target_metric: str
    promotion_threshold: float
    output_dir: str
    output_file: str


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
    features: FeaturesStep | None = None
    stats: StatsStep | None = None
    causal: CausalStep | None = None
    train: TrainStep | None = None
    evaluate: EvaluateStep | None = None
    full: FullStep | None = None
