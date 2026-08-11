from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import BaseModel


class TaskType(str, Enum):
    REGRESSION = "regression"
    CLASSIFICATION = "classification"


class ColumnRole(str, Enum):
    FEATURE = "feature"
    TARGET = "target"
    DATETIME = "datetime"
    IGNORE = "ignore"


class ColumnSchema(BaseModel):
    dtype: str
    null_count: int
    role: ColumnRole


class DatasetContract(BaseModel):
    name: str
    path: str
    target: str
    task: TaskType
    datetime_column: str | None
    columns: dict[str, ColumnSchema]
    lookup_tables: dict[str, str]
    row_count: int


class EnvironmentConfig(BaseModel):
    log_level: str
    data_dir: str
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
    on: list[str]
    smoothing: int | None


class FeatureConfig(BaseModel):
    include: list[str]
    exclude: list[str]
    derived: list[DerivedFeature]
    encodings: list[EncodingConfig]


class ModelConfig(BaseModel):
    type: str
    params: dict[str, float | int | str]


class SplitConfig(BaseModel):
    type: Literal["time", "random", "stratified"]
    cutoff: str | None
    validation_size: float


class HPOConfig(BaseModel):
    engine: str
    trials: int
    search_space: dict[str, list[float | int]]


class ExperimentConfig(BaseModel):
    features: FeatureConfig
    model: ModelConfig
    split: SplitConfig
    random_state: int
    target_metric: str
    hpo: HPOConfig | None


class DiscoverStep(BaseModel):
    csv_path: str
    target_column: str
    task: TaskType
    datetime_column: str | None
    ignore_columns: list[str]


class EtlStep(BaseModel):
    ci_sample_size: int


class ContractsStep(BaseModel):
    null_threshold: float
    range_tolerance: float


class EdaStep(BaseModel):
    sample_size: int
    plot_format: str
    output_dir: str


class FeaturesStep(BaseModel):
    encoding_smoothing: int
    frequency_fill: int


class StatsStep(BaseModel):
    group_column: str
    group_values: list[str]
    sample_fraction: float


class CausalStep(BaseModel):
    treatment_column: str
    outcome_column: str
    power: float
    alpha: float


class TrainStep(BaseModel):
    random_state: int
    n_jobs: int
    cv_folds: int


class EvaluateStep(BaseModel):
    target_metric: str
    promotion_threshold: float


class FullStep(BaseModel):
    steps: list[str]


class PipelineConfig(BaseModel):
    dataset: DatasetContract | None
    environment: EnvironmentConfig
    experiment: ExperimentConfig | None
    discover: DiscoverStep | None
    etl: EtlStep | None
    contracts: ContractsStep | None
    eda: EdaStep | None
    features: FeaturesStep | None
    stats: StatsStep | None
    causal: CausalStep | None
    train: TrainStep | None
    evaluate: EvaluateStep | None
    full: FullStep | None
