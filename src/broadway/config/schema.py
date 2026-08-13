from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from broadway.analysis.contracts import AnalysisContract, AnalysisMode


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


class DatasetContract(BaseModel):
    name: str
    path: str
    target: str
    task: TaskType
    datetime_column: str | None
    columns: dict[str, ColumnSchema]
    lookup_tables: dict[str, LookupSpec]


class EnvironmentConfig(BaseModel):
    log_level: str
    data_dir: str
    raw_subdir: str
    processed_subdir: str
    download_chunk_size: int
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
    monitoring_schedule: str


class DerivedFeature(BaseModel):
    name: str
    func: str
    source: str


class EncodingConfig(BaseModel):
    type: str
    columns: list[str]
    smoothing: int | None


class FeatureConfig(BaseModel):
    include: list[str]
    exclude: list[str]
    derived: list[DerivedFeature]
    encodings: list[EncodingConfig]
    builder_module: str | None = None


class ModelConfig(BaseModel):
    type: str
    params: dict[str, float | int | str]


class SplitConfig(BaseModel):
    type: Literal["time", "random", "stratified"]
    validation_size: float = Field(ge=0.0, le=1.0)


class HPOConfig(BaseModel):
    engine: str
    trials: int
    search_space: dict[str, list[float | int]]


VALID_MODEL_PARAMS: dict[str, frozenset[str]] = {
    "linear": frozenset({"fit_intercept", "positive", "copy_X", "n_jobs"}),
    "lgbm": frozenset({"n_estimators", "learning_rate", "num_leaves", "max_depth", "subsample", "colsample_bytree", "random_state", "n_jobs"}),
    "xgb": frozenset({"n_estimators", "learning_rate", "max_depth", "subsample", "colsample_bytree", "random_state", "n_jobs", "tree_method"}),
    "rf": frozenset({"n_estimators", "max_depth", "max_samples", "random_state", "n_jobs"}),
}


class ExperimentConfig(BaseModel):
    features: FeatureConfig
    model: ModelConfig
    split: SplitConfig
    random_state: int
    target_metric: str
    hpo: HPOConfig | None = None

    @model_validator(mode="after")
    def _validate_hpo_search_space(self) -> ExperimentConfig:
        if self.hpo is None:
            return self
        valid_params = VALID_MODEL_PARAMS.get(self.model.type, frozenset())
        invalid_params = set(self.hpo.search_space) - valid_params
        if invalid_params:
            raise ValueError(
                f"invalid HPO search-space params for model type '{self.model.type}': "
                f"{sorted(invalid_params)}. valid params: {sorted(valid_params)}"
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


class ProjectConfig(BaseModel):
    raw_dir: str
    processed_dir: str
    processed_file: str
    min_trip_distance: float
    max_trip_distance: float
    min_trip_duration_minutes: float
    max_trip_duration_minutes: float
    min_pickup_datetime: str
    min_passenger_count: int
    max_passenger_count: int
    rename_map: dict[str, str]
    borough_column: str
    borough_lookup_column: str
    lookup_path: str
    rush_hour_morning_start: int
    rush_hour_morning_end: int
    rush_hour_evening_start: int
    rush_hour_evening_end: int
    night_start: int
    night_end: int


class ContractsStep(BaseModel):
    null_threshold: float


class EdaStep(BaseModel):
    output_dir: str
    max_columns: int
    output_file: str
    mcar_alpha: float
    outlier_iqr_multiplier: float


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
    def _validate_flow_modes(self) -> "FullStep":
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
