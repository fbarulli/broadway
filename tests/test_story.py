"""Public behavior of the project story page.

A synthetic checkout with two valid pipeline records, their EDA sidecar and one
malformed record: the story must lead with the question, narrate each stage
conclusions-first, reference generated figures (and every referenced URL must
serve a PNG), hyperlink the decision records, say "not run yet" for a stage with
no record, escape record text, and never raise for the malformed record (one
logged traceback instead). A second fixture pins the evidence-scoping rule: a
stale foreign CSV must never be read, and an `act` record without a sidecar must
not be reported as "found nothing". The acceptance ruling is pinned explicitly:
every section that makes a data claim carries a figure, while "What we learned /
what is next" carries numbers and record links and deliberately no chart.
"""

from __future__ import annotations

import json
import logging
import re
import shutil
from pathlib import Path

from fastapi.testclient import TestClient

from broadway.reports import experiments_dashboard as ui
from broadway.reports import story as story_module
from broadway.reports import story_page
from broadway.reports.decision import Decision
from broadway.reports.story_models import StoryStepMap


def _decision(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "question": "Does the metric clear its threshold?",
        "metric": "association_<x>",
        "value": 0.94,
        "interval_low": 0.93,
        "interval_high": 0.95,
        "reference": 0.0,
        "delta": 0.94,
        "threshold": 0.05,
        "null_threshold": 0.0089,
        "direction": "max",
        "verdict": "act",
        "action": "keep the feature",
        "owner": "synthetic",
        "sample_size": 50000,
        "unit": None,
        "min_sample_size": 100,
        "evidence": {
            "baseline": "no-effect shuffle of the target at this n",
            "delta": None,
            "segments": [{"segment": "weekday", "value": 0.94, "sample_size": 30000}],
            "sensitivity": {"half_sample": 0.93},
            "falsifier": "the interval falls at or below the threshold",
            "provenance": {"dataset": "synthetic", "seed": "42"},
        },
    }
    payload.update(overrides)
    return payload


def _write(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _steps() -> StoryStepMap:
    return StoryStepMap(
        configured=(
            "etl",
            "eda",
            "features",
            "baseline",
            "train",
            "selection",
            "hpo",
            "evaluate",
        ),
        planned=(
            "01_clean",
            "02_missing",
            "03_outliers",
            "04_frame",
            "06_correlation",
            "07_clustering",
            "08_timeseries",
        ),
    )


def _config(dataset: str) -> story_module.StoryConfig:
    return story_module.StoryConfig(
        dataset=dataset,
        steps=_steps(),
        question="Which <b>orders</b> end badly?",
        why="Bad reviews are the cost this project predicts, at purchase time.",
        target_label="bad review",
        target_rule="score ≤ 2",
        decision_moment="purchase time",
        source_name="Synthetic",
        source_tables=9,
        post_purchase_note=(
            "Delivery timestamps are post-purchase context, never model features."
        ),
        excluded_note="Review score and delivery timestamps are excluded as leakage.",
        caveats=("imbalanced", "purchase-time only", "prediction not causation"),
    )


def _configure(monkeypatch, root: Path, dataset: str) -> None:
    monkeypatch.setattr(story_page, "_CONFIG", _config(dataset))
    monkeypatch.setattr(story_page, "_ROOT_PROVIDER", lambda: root)
    monkeypatch.setattr(ui, "EXPERIMENTS_ROOT", root)


def test_story_leads_with_conclusions_and_skips_bad_records(
    monkeypatch, tmp_path: Path, caplog
) -> None:
    root = tmp_path / "experiments"
    etl = root / "results" / "pipeline" / "etl"
    _write(
        etl / "etl_decision.json",
        Decision.model_validate(
            _decision(
                question="Did <b>ingest</b> lose rows the contract explains?",
                metric="rows_dropped_unexplained_share",
                value=0.0,
                interval_low=0.0,
                interval_high=0.0,
                unit="share",
                verdict="no_action",
                sample_size=50,
                evidence={
                    "baseline": "every dropped row carries a reason",
                    "delta": None,
                    "segments": [{"segment": "unexplained", "value": 0.0, "sample_size": 50}],
                    "sensitivity": {"rows_out": 50.0},
                    "falsifier": "the interval clears the drop ceiling",
                    "provenance": {
                        "dataset": "synthetic",
                        "rows_in": "60",
                        "rows_out": "50",
                        "reasons": "null target",
                    },
                },
            )
        ).model_dump(mode="json"),
    )

    eda = root / "results" / "pipeline" / "eda"
    _write(
        eda / "eda_decision.json",
        Decision.model_validate(
            _decision(
                metric="association_rating_accuracy_score",
                value=0.25,
                interval_low=0.18,
                interval_high=0.31,
                unit="epsilon_squared",
                sample_size=50,
                evidence={
                    "baseline": "no-effect shuffle of the target at this n",
                    "delta": None,
                    "segments": [{"segment": "weekday", "value": 0.94, "sample_size": 30000}],
                    "sensitivity": {"half_sample": 0.93},
                    "falsifier": "the interval falls at or below the threshold",
                    "provenance": {
                        "dataset": "synthetic",
                        "seed": "42",
                        "family_size": "21",
                        "acted": "2",
                        "investigated": "1",
                    },
                },
            )
        ).model_dump(mode="json"),
    )
    _write(
        eda / "eda_feature_decisions.json",
        {
            "dataset": "synthetic",
            "features": [
                _decision(
                    metric="association_rating_accuracy_score",
                    value=0.25,
                    interval_low=0.18,
                    interval_high=0.31,
                    sample_size=50,
                ),
                _decision(
                    metric="association_personalization_score",
                    value=0.19,
                    interval_low=0.14,
                    interval_high=0.24,
                    sample_size=50,
                ),
                _decision(
                    metric="association_browsing_frequency",
                    value=0.01,
                    interval_low=0.0,
                    interval_high=0.02,
                    verdict="no_action",
                    sample_size=50,
                ),
            ],
            "outliers": [
                _decision(
                    metric="outlier_share_age",
                    value=0.03,
                    interval_low=0.02,
                    interval_high=0.04,
                    sample_size=50,
                ),
            ],
            "comparison": [
                _decision(
                    metric="standardized_shift_estimated_days",
                    value=4.02e-16,
                    interval_low=4.02e-16,
                    interval_high=0.0113,
                    sample_size=50,
                    verdict="no_action",
                    null_threshold=0.009,
                )
            ],
        },
    )

    # Malformed record: the story must skip it and log one traceback.
    hpo = root / "results" / "pipeline" / "hpo"
    hpo.mkdir(parents=True)
    (hpo / "hpo_decision.json").write_text("[]", encoding="utf-8")

    _configure(monkeypatch, root, "synthetic")

    with caplog.at_level(logging.ERROR):
        client = TestClient(ui.app)
        response = client.get("/story")
        missing_figure = client.get("/story/figure/model")

    assert response.status_code == 200
    page = response.text
    assert "The question" in page
    assert "Bad reviews are the cost this project predicts" in page
    assert "What the data says" in page
    assert "rating accuracy score" in page
    # The correction family is the record's family_size, not the sidecar row count.
    assert "21 tested, top 3 by effect decided (2 act, 1 no action)" in page
    assert "The BH correction family is 21 features; the top 3 by effect are ranked." in page
    # A tiny no_action shift is no shift, not "one-sided".
    assert "consistent with no material change" in page
    assert "one-sided" not in page
    # Conclusions first, graph, then numbers and the record link.
    assert "Verdict: no_action" in page
    assert "verdict: act" in page
    assert '<a href="/experiments/etl?focus=pipeline">ETL record</a>' in page
    # A stage without a record says so.
    assert "not run yet — the features decision record has not been published." in page
    # The project's configured steps and planned lanes are both explicit.
    assert "The step map" in page
    assert "2 of 8 configured steps done; 7 planned lanes not configured." in page
    assert "etl — done" in page
    assert "eda — done" in page
    assert "baseline — not run yet" in page
    assert "01_clean — planned lane — not configured yet" in page
    step_map = page.split("<h2>The step map</h2>", 1)[1].split("</section>", 1)[0]
    assert "<figure>" not in step_map
    assert "<img" not in step_map
    # Dynamic record text is escaped.
    assert "&lt;b&gt;ingest&lt;/b&gt;" in page
    assert "<b>ingest</b>" not in page
    # Every img src the page emits must serve a PNG (URL and route agree).
    images = re.findall(r'<img src="([^"]+)"', page)
    assert "/story/figure/question" in images
    assert "/story/figure/data" in images
    assert "/story/figure/eda" in images
    assert "/story/figure/change" in images
    for url in images:
        figure = client.get(url)
        assert figure.status_code == 200, url
        assert figure.headers["content-type"] == "image/png", url
        assert figure.content.startswith(b"\x89PNG"), url
    # The model has not run in this fixture.
    assert missing_figure.status_code == 404
    # The root dashboard keeps the story as the first surface a human sees.
    dashboard = client.get("/?focus=pipeline")
    assert dashboard.status_code == 200
    assert 'class="story-banner"' in dashboard.text
    assert '<a href="/story">Read the project story' in dashboard.text
    assert caplog.text.count("unreadable decision record") == 1


def _write_olist_eda(eda: Path) -> None:
    _write(
        eda / "eda_decision.json",
        Decision.model_validate(
            _decision(
                metric="association_delivery_days",
                value=0.99,
                interval_low=0.90,
                interval_high=0.99,
                threshold=0.05,
                unit="rank_biserial_r",
                verdict="act",
                owner="olist",
                sample_size=100,
                evidence={
                    "baseline": "no-effect shuffle of the target at this n",
                    "delta": None,
                    "segments": [{"segment": "all", "value": 0.99, "sample_size": 100}],
                    "sensitivity": {"half_sample": 0.98},
                    "falsifier": "the interval falls at or below the threshold",
                    "provenance": {"dataset": "olist", "seed": "42"},
                },
            )
        ).model_dump(mode="json"),
    )


def test_story_ignores_stale_foreign_evidence(monkeypatch, tmp_path: Path) -> None:
    root = tmp_path / "experiments"
    eda = root / "results" / "pipeline" / "eda"
    _write_olist_eda(eda)
    # A stale table from another dataset sits in the same results dir.
    (eda / "eda_effects.csv").write_text(
        "feature,effect,verdict\nAMAZON_STALE_FEATURE,0.99,act\n", encoding="utf-8"
    )
    # And a stale decision record from another dataset must be filtered out even
    # when the env override is empty (empty means "use the configured dataset").
    _write(
        root / "results" / "pipeline" / "etl" / "etl_decision.json",
        Decision.model_validate(
            _decision(
                question="AMAZON STALE RECORD",
                metric="rows_dropped_unexplained_share",
                value=0.0,
                interval_low=0.0,
                interval_high=0.0,
                owner="amazon",
                sample_size=10,
                evidence={
                    "baseline": "stale",
                    "delta": None,
                    "segments": [{"segment": "all", "value": 0.0, "sample_size": 10}],
                    "sensitivity": {"rows_out": 10.0},
                    "falsifier": "stale",
                    "provenance": {"dataset": "amazon"},
                },
            )
        ).model_dump(mode="json"),
    )

    _configure(monkeypatch, root, "olist")
    monkeypatch.setenv("BROADWAY_STORY_DATASET", "")
    page = TestClient(ui.app).get("/story").text

    assert "AMAZON_STALE_FEATURE" not in page
    assert "AMAZON STALE RECORD" not in page
    assert "not run yet — the etl decision record has not been published." in page
    assert "association_delivery_days = 0.99" in page


def test_story_features_figure_and_learned_numbers(monkeypatch, tmp_path: Path) -> None:
    """Data-claim sections carry a figure; the learned section carries numbers/links only."""
    root = tmp_path / "experiments"
    features = root / "results" / "pipeline" / "features"
    _write(
        features / "features_decision.json",
        Decision.model_validate(
            _decision(
                question="Did the feature step keep every row?",
                metric="row_loss_share",
                value=0.0,
                interval_low=0.0,
                interval_high=0.0,
                unit="share",
                verdict="no_action",
                sample_size=50,
                evidence={
                    "baseline": "row-preserving transform",
                    "delta": None,
                    "segments": [{"segment": "all", "value": 0.0, "sample_size": 50}],
                    "sensitivity": {
                        "features_added_n50": 3.0,
                        "features_removed_n50": 1.0,
                        "plan_columns_n50": 4.0,
                    },
                    "falsifier": "the row-loss interval clears the drop ceiling",
                    "provenance": {
                        "dataset": "synthetic",
                        "columns_added": "first_enc, second_enc, third_enc",
                    },
                },
            )
        ).model_dump(mode="json"),
    )
    evaluate = root / "results" / "pipeline" / "evaluate"
    _write(
        evaluate / "champion_promotion_decision.json",
        Decision.model_validate(
            _decision(
                question="Does the candidate beat the champion on the holdout?",
                metric="improvement_rmse",
                value=0.20,
                interval_low=0.10,
                interval_high=0.30,
                threshold=0.05,
                unit="relative_improvement",
                verdict="act",
                sample_size=50,
            )
        ).model_dump(mode="json"),
    )
    _configure(monkeypatch, root, "synthetic")
    client = TestClient(ui.app)
    page = client.get("/story").text

    # A features data claim carries its feature-surface figure.
    assert "Feature surface: added 3, removed 1, plan columns 4." in page
    assert "/story/figure/features" in page
    assert client.get("/story/figure/features").status_code == 200
    # The learned section is next-steps: numbers and a record link, never a chart.
    learned = page.split("<h2>What we learned / what is next</h2>", 1)[1].split(
        "</section>", 1
    )[0]
    assert "The result is bounded by" in learned
    assert "holdout ·" in learned
    assert '<a href="/experiments/evaluate?focus=pipeline">holdout record</a>' in learned
    assert "<figure>" not in learned
    assert "<img" not in learned


def test_story_eda_count_uses_family_size(monkeypatch, tmp_path: Path) -> None:
    """21 tested / 12 decided must come from the record, not the 12 decision rows."""
    root = tmp_path / "experiments"
    eda = root / "results" / "pipeline" / "eda"
    _write(
        eda / "eda_decision.json",
        Decision.model_validate(
            _decision(
                metric="association_delivery_days",
                value=0.41,
                interval_low=0.36,
                interval_high=0.51,
                unit="rank_biserial_r",
                sample_size=100,
                evidence={
                    "baseline": "no-effect shuffle of the target at this n",
                    "delta": None,
                    "segments": [{"segment": "all", "value": 0.41, "sample_size": 100}],
                    "sensitivity": {"half_sample": 0.39},
                    "falsifier": "the interval falls at or below the threshold",
                    "provenance": {
                        "dataset": "synthetic",
                        "family_size": "21",
                        "acted": "11",
                        "investigated": "1",
                    },
                },
            )
        ).model_dump(mode="json"),
    )
    features = [
        _decision(
            metric=f"association_feature_{index}",
            value=0.20 - index * 0.01,
            interval_low=0.10,
            interval_high=0.30,
            sample_size=50,
        )
        for index in range(11)
    ]
    features.append(
        _decision(
            metric="association_feature_11",
            value=0.01,
            interval_low=0.0,
            interval_high=0.02,
            sample_size=50,
            verdict="investigate",
        )
    )
    _write(
        eda / "eda_feature_decisions.json",
        {"dataset": "synthetic", "features": features},
    )
    _configure(monkeypatch, root, "synthetic")
    page = TestClient(ui.app).get("/story").text

    assert "21 tested, top 12 by effect decided (11 act, 1 investigate)" in page
    assert (
        "The BH correction family is 21 features; the top 12 by effect are ranked."
        in page
    )


def test_story_reads_provenance_named_class_balance(
    monkeypatch, tmp_path: Path
) -> None:
    root = tmp_path / "experiments"
    etl = root / "results" / "pipeline" / "etl"
    _write(
        etl / "etl_decision.json",
        Decision.model_validate(
            _decision(
                metric="rows_dropped_unexplained_share",
                value=0.0,
                interval_low=0.0,
                interval_high=0.0,
                unit="share",
                verdict="no_action",
                sample_size=99441,
                evidence={
                    "baseline": "every dropped row carries a reason",
                    "delta": None,
                    "segments": [{"segment": "all", "value": 0.0, "sample_size": 99441}],
                    "sensitivity": {"rows_out": 98673.0},
                    "falsifier": "the interval clears the drop ceiling",
                    "provenance": {
                        "dataset": "synthetic",
                        "rows_in": "99441",
                        "rows_out": "98673",
                        "reasons": "null target: -768 rows",
                        "class_balance": "class_balance.csv",
                    },
                },
            )
        ).model_dump(mode="json"),
    )
    (etl / "class_balance.csv").write_text(
        "class,count,share_of_total,share_of_kept\n"
        "bad_review,14495,0.1457,0.1469\n"
        "good_review,84178,0.8466,0.8531\n"
        "missing_target,768,0.0077,0.0\n",
        encoding="utf-8",
    )
    _configure(monkeypatch, root, "synthetic")
    page = TestClient(ui.app).get("/story").text

    assert (
        "bad review 14,495 / good review 84,178 (n=98,673); "
        "768 unlabeled rows were dropped"
    ) in page
    assert "missing target 768" not in page


def test_story_does_not_invent_a_conclusion_when_evidence_is_missing(
    monkeypatch, tmp_path: Path
) -> None:
    root = tmp_path / "experiments"
    _write_olist_eda(root / "results" / "pipeline" / "eda")
    _configure(monkeypatch, root, "olist")
    client = TestClient(ui.app)
    page = client.get("/story").text

    assert "The EDA record reports act on association_delivery_days" in page
    assert "association_delivery_days = 0.99" in page
    assert "verdict: act" in page
    assert "evidence table has not been published" in page
    assert "found nothing" not in page
    assert "/story/figure/eda" not in page
    assert client.get("/story/figure/eda").status_code == 404


def test_story_eda_figure_survives_the_real_biased_rows(
    monkeypatch, tmp_path: Path
) -> None:
    """The committed zip/city interval is biased; the EDA figure must still be 200."""
    root = tmp_path / "experiments"
    eda = root / "results" / "pipeline" / "eda"
    source = Path("project/experiments/results/pipeline/eda")
    for name in ("eda_decision.json", "eda_feature_decisions.json", "eda_effects.csv"):
        eda.mkdir(parents=True, exist_ok=True)
        shutil.copy(source / name, eda / name)

    _configure(monkeypatch, root, "olist")
    client = TestClient(ui.app)
    figure = client.get("/story/figure/eda")
    page = client.get("/story").text

    assert figure.status_code == 200
    assert figure.headers["content-type"] == "image/png"
    assert "customer zip code prefix, customer city: bootstrap-biased" in page
    assert "interval not centred on the point" in page
    assert "In the purchase-time feature ranking" in page
    assert "are the strongest context associations" in page
    assert "12 of 12 numeric feature columns" in page


def test_story_selection_survives_malformed_candidates(
    monkeypatch, tmp_path: Path, caplog
) -> None:
    """A bad candidate row degrades the section; it never 500s the story page."""
    root = tmp_path / "experiments"
    selection = root / "results" / "pipeline" / "selection"
    _write(
        selection / "selection_decision.json",
        Decision.model_validate(
            _decision(
                question="Does the selected champion beat the naive reference?",
                metric="improvement_over_naive_rmse",
                value=0.18,
                interval_low=0.13,
                interval_high=0.23,
                threshold=0.02,
                unit="relative_improvement",
                verdict="act",
                sample_size=50,
            )
        ).model_dump(mode="json"),
    )
    # The champion record exists but cv_mean is non-numeric; the CSV is the fallback.
    _write(selection / "selection_champion.json", {"champion": "rf", "cv_mean": "n/a"})
    csv_path = selection / "selection_candidates.csv"
    csv_path.write_text(
        "candidate,rmse,rmse_low,rmse_high,role\n"
        "oops,oops,oops,oops,candidate\n"
        "rf,0.80,0.78,0.82,candidate\n",
        encoding="utf-8",
    )
    _configure(monkeypatch, root, "synthetic")
    client = TestClient(ui.app)

    with caplog.at_level(logging.ERROR):
        response = client.get("/story")
        figure = client.get("/story/figure/model")
        # Now only the malformed row survives: the section must degrade, not 500.
        csv_path.write_text(
            "candidate,rmse,rmse_low,rmse_high,role\n"
            "oops,oops,oops,oops,candidate\n",
            encoding="utf-8",
        )
        degraded = client.get("/story")
        degraded_figure = client.get("/story/figure/model")

    assert response.status_code == 200
    assert "rf won the repeated-CV leaderboard at 0.800 RMSE [0.780, 0.820]" in response.text
    assert figure.status_code == 200
    assert degraded.status_code == 200
    assert "candidate evidence table has not been published" in degraded.text
    assert degraded_figure.status_code == 404
    assert caplog.text.count("malformed candidate row") == 1


def test_story_model_uses_the_record_metric(monkeypatch, tmp_path: Path) -> None:
    """A binary target's pr_auc leaderboard is narrated, not silenced for lacking rmse."""
    root = tmp_path / "experiments"
    selection = root / "results" / "pipeline" / "selection"
    _write(
        selection / "selection_decision.json",
        Decision.model_validate(
            _decision(
                question="Does the selected champion beat the naive reference?",
                metric="improvement_over_naive_pr_auc",
                value=0.18,
                interval_low=0.13,
                interval_high=0.23,
                threshold=0.02,
                unit="relative_improvement",
                verdict="act",
                sample_size=50,
            )
        ).model_dump(mode="json"),
    )
    _write(selection / "selection_champion.json", {"champion": "rf"})
    (selection / "selection_candidates.csv").write_text(
        "candidate,pr_auc,pr_auc_low,pr_auc_high,role\n"
        "rf,0.82,0.79,0.85,candidate\n",
        encoding="utf-8",
    )
    _configure(monkeypatch, root, "synthetic")
    client = TestClient(ui.app)
    page = client.get("/story").text

    assert "rf won the repeated-CV leaderboard at 0.820 PR AUC [0.790, 0.850]" in page
    assert client.get("/story/figure/model").status_code == 200
