from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from broadway.config.loader import load_config
from broadway.config.schema import (
    BuilderParams,
    DataSourceRef,
    ExperimentConfig,
    FeatureConfig,
    HPOConfig,
    LookupSpec,
    ModelConfig,
    ModelHPOSpec,
    SplitConfig,
    TrainStep,
)


@pytest.fixture
def train_cfg():
    cfg = load_config("train", dataset="test", experiment="baseline")
    assert cfg.train is not None
    return cfg.train


@pytest.fixture
def evaluate_cfg():
    cfg = load_config("evaluate", dataset="test", experiment="baseline")
    assert cfg.evaluate is not None
    return cfg.evaluate


@pytest.fixture
def etl_cfg():
    cfg = load_config("etl", dataset="test", experiment="baseline")
    assert cfg.etl is not None
    return cfg.etl


@pytest.fixture
def stats_cfg():
    cfg = load_config("stats", dataset="test", experiment="baseline")
    assert cfg.stats is not None
    return cfg.stats


def test_load_train_config(train_cfg) -> None:
    assert train_cfg.model_file
    assert train_cfg.random_state


def _train_step_kwargs(**overrides: object) -> dict[str, object]:
    kwargs: dict[str, object] = {
        "random_state": 42,
        "n_jobs": -1,
        "cv_folds": 5,
        "cv_kind": "kfold",
        "model_file": "model.pkl",
        "n_estimators": 500,
        "learning_rate": 0.05,
        "num_leaves": 63,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "quantile_tail": 0.9,
        "output_dir": "artifacts/training",
        "output_file": "training_result.json",
    }
    kwargs.update(overrides)
    return kwargs


def test_train_step_cv_kind_required() -> None:
    kwargs = _train_step_kwargs()
    del kwargs["cv_kind"]
    with pytest.raises(ValidationError):
        TrainStep(**kwargs)


def test_train_step_cv_kind_invalid_raises() -> None:
    with pytest.raises(ValidationError):
        TrainStep(**_train_step_kwargs(cv_kind="bogus"))


def test_train_step_cv_kind_time_series_split_parses() -> None:
    step = TrainStep(**_train_step_kwargs(cv_kind="time_series_split"))
    assert step.cv_kind == "time_series_split"


def test_train_step_cv_kind_kfold_parses() -> None:
    step = TrainStep(**_train_step_kwargs(cv_kind="kfold"))
    assert step.cv_kind == "kfold"


def test_load_evaluate_config(evaluate_cfg) -> None:
    assert evaluate_cfg.target_metric
    assert evaluate_cfg.promotion_threshold


def test_load_etl_with_pipeline_fields(etl_cfg) -> None:
    assert etl_cfg.train_file
    assert etl_cfg.max_drop_fraction >= 0.0


def test_missing_dataset_raises() -> None:
    with pytest.raises(FileNotFoundError):
        load_config("train", dataset="nonexistent", experiment="baseline")


def test_missing_experiment_raises() -> None:
    with pytest.raises(FileNotFoundError):
        load_config("full", dataset="test", experiment="nonexistent")


def test_invalid_step_raises() -> None:
    with pytest.raises(ValueError, match="unknown step"):
        load_config("bogus")


def test_stats_config_fields(stats_cfg) -> None:
    assert stats_cfg.min_rows_for_sampling
    assert stats_cfg.per_group_sample_fraction


def _experiment_config(model_type: str, search_space: dict[str, list[float | int]]) -> ExperimentConfig:
    return ExperimentConfig(
        data_source=DataSourceRef(loader="canonical", schema_contract="raw"),
        features=FeatureConfig(include=["rooms"], exclude=[], derived=[], encodings=[]),
        model=ModelConfig(type=model_type, params={}),
        split=SplitConfig(type="random", validation_size=0.2),
        random_state=42,
        target_metric="rmse",
        hpo=HPOConfig(
            engine="optuna",
            total_trials=10,
            initial_trials_per_model=5,
            top_k=1,
            target_metric="rmse",
            models=[ModelHPOSpec(name=model_type, search_space=search_space)],
        ),
    )


def test_hpo_search_space_valid_for_lgbm() -> None:
    config = _experiment_config(
        "lgbm",
        {"max_depth": [3, 10], "learning_rate": [0.01, 0.3]},
    )
    assert config.hpo is not None
    assert set(config.hpo.models[0].search_space) == {"max_depth", "learning_rate"}


def test_hpo_search_space_invalid_for_linear_raises() -> None:
    with pytest.raises(ValidationError):
        _experiment_config("linear", {"max_depth": [3, 10]})


def test_hpo_search_space_validated_per_model() -> None:
    with pytest.raises(ValidationError):
        ExperimentConfig(
            data_source=DataSourceRef(loader="canonical", schema_contract="raw"),
            features=FeatureConfig(include=["rooms"], exclude=[], derived=[], encodings=[]),
            model=ModelConfig(type="lgbm", params={}),
            split=SplitConfig(type="random", validation_size=0.2),
            random_state=42,
            target_metric="rmse",
            hpo=HPOConfig(
                engine="optuna",
                total_trials=10,
                initial_trials_per_model=5,
                top_k=1,
                target_metric="rmse",
                models=[
                    ModelHPOSpec(name="lgbm", search_space={"learning_rate": [0.01, 0.3]}),
                    ModelHPOSpec(name="xgb", search_space={"num_leaves": [10, 50]}),
                ],
            ),
        )


def test_lookup_spec_value_policies_parse() -> None:
    spec = LookupSpec(**yaml.safe_load(
        """
        path: lookup.csv
        key: LocationID
        value_policies:
          district:
            sentinel_values:
              - Unknown
        """
    ))
    assert spec.value_policies["district"].sentinel_values == ["Unknown"]


def test_lookup_spec_value_policies_default_empty() -> None:
    spec = LookupSpec(path="lookup.csv", key="LocationID")
    assert spec.value_policies == {}


def test_lookup_spec_na_values_parse() -> None:
    spec = LookupSpec(**yaml.safe_load(
        """
        path: lookup.csv
        key: LocationID
        na_values:
          - ""
          - N/A
        """
    ))
    assert spec.na_values == ["", "N/A"]


def test_lookup_spec_na_values_default_empty() -> None:
    spec = LookupSpec(path="lookup.csv", key="LocationID")
    assert spec.na_values == []


def test_hpo_configs_parse() -> None:
    """Both HPO configs must parse as the unified HPOConfig schema."""
    import yaml

    from broadway.config.schema import HPOConfig

    for path in (
        Path(__file__).resolve().parents[1] / "configs" / "experiment" / "hyperopt.yaml",
        Path(__file__).resolve().parents[1] / "configs" / "experiments" / "mlflow.yaml",
    ):
        raw = yaml.safe_load(path.read_text())
        assert "hpo" in raw, f"{path.name} missing hpo block"
        cfg = HPOConfig(**raw["hpo"])
        assert cfg.models, f"{path.name}: no models in hpo spec"


def _experiment_kwargs() -> dict[str, object]:
    return {
        "features": FeatureConfig(include=["rooms"], exclude=[], derived=[], encodings=[]),
        "model": ModelConfig(type="linear", params={}),
        "split": SplitConfig(type="random", validation_size=0.2),
        "random_state": 42,
        "target_metric": "rmse",
    }


def test_experiment_config_requires_data_source() -> None:
    with pytest.raises(ValidationError):
        ExperimentConfig(**_experiment_kwargs())


@pytest.mark.parametrize("loader", ["bogus", "canonical_", "Joined"])
def test_data_source_rejects_invalid_loader(loader: str) -> None:
    with pytest.raises(ValidationError):
        DataSourceRef(loader=loader, schema_contract="raw")


def test_named_sample_requires_version() -> None:
    with pytest.raises(ValidationError, match="requires data_source.version"):
        DataSourceRef(loader="named_sample", schema_contract="raw")


@pytest.mark.parametrize("loader", ["canonical", "joined", "pinned"])
def test_version_rejected_for_non_named_sample_loader(loader: str) -> None:
    with pytest.raises(ValidationError, match="only valid for loader='named_sample'"):
        DataSourceRef(loader=loader, version="v3", schema_contract="raw")


@pytest.mark.parametrize(
    ("loader", "version"),
    [
        ("canonical", None),
        ("joined", None),
        ("named_sample", "v3"),
        ("pinned", None),
    ],
)
def test_data_source_valid_cases_parse(loader: str, version: str | None) -> None:
    ref = DataSourceRef(loader=loader, version=version, schema_contract="raw")
    assert ref.loader == loader
    assert ref.version == version
    assert ref.schema_contract == "raw"


def test_experiment_config_parses_with_data_source() -> None:
    config = ExperimentConfig(
        data_source=DataSourceRef(loader="canonical", schema_contract="raw"),
        **_experiment_kwargs(),
    )
    assert config.data_source.loader == "canonical"
    assert config.data_source.schema_contract == "raw"


def test_experiment_preprocessing_defaults_to_empty_list() -> None:
    config = ExperimentConfig(
        data_source=DataSourceRef(loader="canonical", schema_contract="raw"),
        **_experiment_kwargs(),
    )
    assert config.preprocessing == []


def test_experiment_preprocessing_steps_parse() -> None:
    config = ExperimentConfig(
        data_source=DataSourceRef(loader="canonical", schema_contract="raw"),
        preprocessing=[
            {
                "type": "target_encoding",
                "columns": ["rooms"],
                "params": {"smoothing": 20},
            },
            {"type": "passthrough", "columns": ["area"]},
        ],
        **_experiment_kwargs(),
    )
    assert [step.type for step in config.preprocessing] == [
        "target_encoding",
        "passthrough",
    ]
    assert config.preprocessing[0].params == {"smoothing": 20}


def test_preprocessing_step_unknown_type_rejected() -> None:
    with pytest.raises(ValidationError):
        ExperimentConfig(
            data_source=DataSourceRef(loader="canonical", schema_contract="raw"),
            preprocessing=[{"type": "bogus", "columns": ["rooms"]}],
            **_experiment_kwargs(),
        )


def test_preprocessing_step_extra_fields_forbidden() -> None:
    with pytest.raises(ValidationError):
        ExperimentConfig(
            data_source=DataSourceRef(loader="canonical", schema_contract="raw"),
            preprocessing=[{"type": "one_hot", "columns": ["rooms"], "extra": 1}],
            **_experiment_kwargs(),
        )


def test_experiment_builder_params_round_trip_from_yaml() -> None:
    raw = yaml.safe_load(
        """
        data_source:
          loader: canonical
          schema_contract: raw
        features:
          include: [rooms]
          exclude: []
          derived: []
          encodings: []
          builder_params:
            group_col: Borough
            lookup_col: Borough_lookup
        model:
          type: linear
          params: {}
        split:
          type: random
          validation_size: 0.2
        random_state: 42
        target_metric: rmse
        """
    )
    config = ExperimentConfig(**raw)
    assert config.features.builder_params == BuilderParams(
        group_col="Borough", lookup_col="Borough_lookup"
    )


def test_feature_config_builder_params_absent_is_none() -> None:
    config = ExperimentConfig(
        data_source=DataSourceRef(loader="canonical", schema_contract="raw"),
        **_experiment_kwargs(),
    )
    assert config.features.builder_params is None
