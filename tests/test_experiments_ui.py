"""UI surface tests — synthetic only (no project-layer coupling, no server).

Covers the failure classes that have bitten the experiment UI:
raw-JS leaking out of its ``<script>`` tags, evidence-table rendering,
and HTML escaping of user/file-derived content.
"""

from __future__ import annotations

import asyncio

import pandas as pd
from starlette.requests import Request

from broadway.reports import experiments_dashboard as ui


def _seed_dashboard(monkeypatch, tmp_path):
    root = tmp_path / "experiments"
    series = root / "alpha"
    series.mkdir(parents=True)
    (series / "01_ingest.py").write_text(
        '\"\"\"01: Load the source data.\"\"\"\nFULL_PARQUET = "full.parquet"\ndf.to_parquet(FULL_PARQUET)\n',
        encoding="utf-8",
    )
    (series / "02_clean.py").write_text(
        '\"\"\"02: Clean the source data.\"\"\"\nCLEAN_PARQUET = FULL_PARQUET\nload_working()\n',
        encoding="utf-8",
    )
    results = root / "results" / "alpha"
    results.mkdir(parents=True)
    (results / "01_ingest.csv").write_text("stat,value\ncount,1\n", encoding="utf-8")
    (results / "01_ingest.png").write_bytes(b"png")
    observations = tmp_path / "observations"
    monkeypatch.setattr(ui, "EXPERIMENTS_ROOT", root)
    monkeypatch.setattr(ui, "OBSERVATIONS_DIR", observations)
    return root, series, observations


def _request(body: bytes) -> Request:
    async def receive() -> dict[str, bytes | bool | str]:
        return {"type": "http.request", "body": body, "more_body": False}

    return Request({"type": "http", "method": "POST", "headers": []}, receive)


def test_series_page_script_is_wrapped() -> None:
    """The drag-and-drop JS must appear ONLY inside ``<script>`` tags."""
    html = ui._render_series_page(
        from_stem="01_a",
        to_stem="03_c",
        stems=["01_a", "02_b", "03_c"],
        profiles={
            "01_a": ui.ScriptProfile(produced=set(), consumed=set(), external=True),
            "02_b": ui.ScriptProfile(produced=set(), consumed=set(), external=True),
            "03_c": ui.ScriptProfile(produced=set(), consumed=set(), external=True),
        },
        focus="test",
        series=["test"],
    )
    assert html.count("<script>") == 1
    assert html.count("</script>") == 1
    inside = html.split("<script>")[1].split("</script>")[0]
    assert "() =>" in inside  # the JS lives inside the tags
    outside = html.split("<script>")[0] + html.split("</script>")[1]
    assert "() =>" not in outside  # and nowhere else (the raw-leak bug)


def test_evidence_table_escapes_and_formats(monkeypatch, tmp_path) -> None:
    """Evidence CSVs render as escaped tables; count rows show integers."""
    csv_path = tmp_path / "05_step_describe.csv"
    pd.DataFrame(
        {
            "feature": ["price", "area"],
            "mean": [10.5, 3.25],
            "count": [988077.0, 988077.0],
        },
        index=["mean", "count"],
    ).to_csv(csv_path)
    monkeypatch.setattr(ui, "_results_dir", lambda focus: tmp_path)
    html = ui._render_evidence("05_step", "test")
    assert '<table class="evidence">' in html
    assert "988,077" in html  # whole floats render as integers with commas
    assert "10.50" in html


def test_evidence_table_escapes_html_in_cells(monkeypatch, tmp_path) -> None:
    """User/file-derived cell content must be HTML-escaped."""
    csv_path = tmp_path / "05_step_describe.csv"
    pd.DataFrame({"feature": ["<script>alert(1)</script>"], "mean": [1.5]},
                 index=["weird"]).to_csv(csv_path)
    monkeypatch.setattr(ui, "_results_dir", lambda focus: tmp_path)
    html = ui._render_evidence("05_step", "test")
    assert "<script>alert(1)</script>" not in html
    assert "&lt;script&gt;" in html


def test_default_focus_uses_first_discovered_series(monkeypatch, tmp_path) -> None:
    for series in ("beta", "alpha"):
        directory = tmp_path / series
        directory.mkdir()
        (directory / "01_step.py").write_text('"""step"""', encoding="utf-8")
    monkeypatch.setattr(ui, "EXPERIMENTS_ROOT", tmp_path)

    assert ui._resolve_focus("") == "alpha"
    assert ui._resolve_focus("beta") == "beta"


def test_dashboard_helpers_cover_discovery_rendering_and_renumbering(monkeypatch, tmp_path) -> None:
    _, series, observations = _seed_dashboard(monkeypatch, tmp_path)

    assert ui.list_series() == ["alpha"]
    assert [script.stem for script in ui.series_scripts("alpha")] == ["01_ingest", "02_clean"]
    assert ui._h("/series?from=01_ingest", "a/b") == "/series?from=01_ingest&focus=a%2Fb"
    assert ui._question_from_docstring(series / "01_ingest.py") == "Load the source data."
    malformed = series / "bad.py"
    malformed.write_text("def", encoding="utf-8")
    assert ui._question_from_docstring(malformed) == ""

    profiles = ui.series_profiles("alpha")
    assert profiles["01_ingest"].produced == {"FULL_PARQUET"}
    assert profiles["02_clean"].consumed == {"FULL_PARQUET", "CLEAN_PARQUET"}
    assert profiles["02_clean"].external
    assert ui.census_rows("alpha")[0]["status"] == "ran"
    assert ui._box_class("01_ingest", "alpha") == "box green"
    assert ui._box_class("02_clean", "alpha") == "box gray"
    assert ui._producers_of("FULL_PARQUET", profiles) == ["01_ingest"]
    assert ui._consumers_of("FULL_PARQUET", "01_ingest", profiles) == ["02_clean"]
    assert "upstream (project data / working dataset)" in ui._graph_html("02_clean", profiles, "alpha")
    assert ui._node_href("01_ingest", "alpha") == "/experiments/01_ingest?focus=alpha"

    assert "01_ingest" in ui._render_table(ui.census_rows("alpha"), "alpha")
    assert "01_ingest.png" in ui._render_artifacts("01_ingest", "alpha")
    assert ui._render_artifacts("02_clean", "alpha") == "<p>No artifacts yet.</p>"
    assert "supported" in ui._verdict_options("supported")
    assert "saved" in ui._render_observations_form(
        "01_ingest",
        {"verdict": "supported", "observations": "looks good", "updated_at": "now"},
        "alpha",
        "/series",
    )
    assert ui._prev_next_hrefs(["01_ingest", "02_clean"], "01_ingest", "/experiments/") == (
        "",
        "/experiments/02_clean",
    )
    assert "current" in ui._series_strip(
        ["01_ingest", "02_clean"], "01_ingest", lambda stem: f"/{stem}", "", "/02_clean"
    )
    assert ui._next_prefix("01_ingest", ["01_ingest", "02_clean"]) == "01a"
    assert ui._next_prefix("02_clean", ["01_ingest", "02_clean"]) == "03"
    assert "01: Question" in ui._scaffold_text("01", "Question")
    assert ui._stem_key("01a_ingest") == (1, "a", "ingest")
    assert ui._stem_key("invalid") is None

    (observations / "01_ingest.json").parent.mkdir(parents=True)
    (observations / "01_ingest.json").write_text("not json", encoding="utf-8")
    assert ui.load_observations("01_ingest") == {}
    (observations / "01_ingest.json").write_text('{"verdict": "supported"}', encoding="utf-8")
    assert ui.load_observations("01_ingest") == {"verdict": "supported"}

    reordered = ui._renumber_stems(["02_clean", "01_ingest"], "alpha")
    assert reordered == ["01_clean", "02_ingest"]
    assert (series / "01_clean.py").is_file()
    assert (observations / "02_ingest.json").is_file()


def test_dashboard_routes_validate_and_persist_real_files(monkeypatch, tmp_path) -> None:
    _, series, observations = _seed_dashboard(monkeypatch, tmp_path)
    assert ui.index("alpha").status_code == 200
    assert ui.index("missing").status_code == 404
    assert ui.experiment_page("01_ingest", "alpha").status_code == 200
    assert ui.experiment_page("missing", "alpha").status_code == 404
    assert ui.series_page(from_="", to="", focus="alpha").status_code == 200
    assert ui.series_page("02_clean", "01_ingest", "alpha").status_code == 200
    assert ui.series_page("missing", focus="alpha").status_code == 404
    assert ui.new_step_page("01_ingest", "alpha").status_code == 200
    assert ui.new_step_page("missing", "alpha").status_code == 404

    invalid_create = asyncio.run(
        ui.create_step("01_ingest", _request(b"name=bad-name&question="), "alpha")
    )
    assert invalid_create.status_code == 422
    created = asyncio.run(
        ui.create_step("01_ingest", _request(b"name=audit&question=Audit+the+result"), "alpha")
    )
    assert created.status_code == 303
    assert (series / "01a_audit.py").is_file()

    invalid_observation = asyncio.run(
        ui.save_observations("01_ingest", _request(b"verdict=unknown"), "alpha")
    )
    assert invalid_observation.status_code == 422
    saved = asyncio.run(
        ui.save_observations(
            "01_ingest",
            _request(b"verdict=supported&observations=works&next=%2F%2Funsafe"),
            "alpha",
        )
    )
    assert saved.status_code == 303
    assert saved.headers["location"] == "/experiments/01_ingest?focus=alpha"
    assert ui.load_observations("01_ingest")["observations"] == "works"

    invalid_reorder = asyncio.run(
        ui.reorder_series(_request(b'{"order": ["missing"]}'), "alpha")
    )
    assert invalid_reorder.status_code == 422
    reordered = asyncio.run(
        ui.reorder_series(_request(b'{"order": ["02_clean", "01_ingest"]}'), "alpha")
    )
    assert reordered == {"reordered": ["01_clean", "02_ingest"]}
    assert (series / "01_clean.py").is_file()
    assert (observations / "02_ingest.json").is_file()
