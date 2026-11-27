"""Project wiring for the 08_timeseries step config."""

from datetime import date


def test_step_config_declares_the_datetime_source() -> None:
    from project import activate_config_overlay

    activate_config_overlay()
    from broadway.config.loader import load_config

    cfg = load_config("08_timeseries", dataset="olist")
    assert cfg.timeseries is not None
    assert cfg.timeseries.datetime_column == "order_purchase_timestamp"
    assert cfg.timeseries.start == date(2017, 1, 1)
    assert cfg.timeseries.end == date(2018, 8, 31)
    assert cfg.timeseries.primary_series == "orders"
    assert set(cfg.timeseries.series) == {"orders", "revenue"}
    assert cfg.timeseries.rule.metric == "relative_mae_improvement"
