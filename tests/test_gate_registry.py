"""Verification gate for the machine-readable gate registry SSOT.

Subject: agents/ledger/gates.yaml (sole SSOT; source fragments
agents/ledger/GATES.md + agents/ledger/gates/*.md retired @ HEAD 5016e93,
verbatim content preserved under agents/ledger/arbitration/2026-08-24/),
rendered views agents/ledger/DIGEST.md + agents/ledger/index/by_*.yaml via
agents/tools/render_gates.py.

Checks:
  a. meta.counts match the actual registry body.
  b. id hygiene: unique, GATE-<PREFIX>-NN shape, prefix↔band-range consistency,
     dense ascending order integers 1..N.
  c. every owner path exists in the repo AND a named-symbol candidate appears
     in that file. Exact line numbers are deliberately NOT pinned (they drift;
     they are advisory), and owner descriptors are prose-y, so symbol matching
     accepts any of a deterministic candidate cascade extracted from the owner
     string itself (leading call/name, words, embedded paths) or a structural
     fallback (named sibling modules of the owned package exist).
  d. every validated_by node id exists under tests/ or project/tests/
     (existence via literal/split matching — this is NOT pytest collection).
  e. reference closure: every id-like token referenced in any inputs[] /
     outputs[] / if_changed[] resolves to a declared id — a registry gate id,
     an externally-declared entry of the top-level ``vocabulary:``, or an
     id-like token used outside the referencing slot itself.
  f. renderer round-trip: agents/tools/render_gates.py main() runs and
     rewrites DIGEST.md and the index sidecars byte-identically on a second
     call, and ``--help`` exits 0.
"""
from __future__ import annotations

import importlib.util
import re
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[1]
GATES_YAML = REPO / "agents" / "ledger" / "gates.yaml"
DIGEST = REPO / "agents" / "ledger" / "DIGEST.md"
INDEX_DIR = REPO / "agents" / "ledger" / "index"
INDEX_FILES = [
    INDEX_DIR / "by_owner_file.yaml",
    INDEX_DIR / "by_artifact.yaml",
    INDEX_DIR / "by_library.yaml",
]

# Band ranges: bands 01–09 + 80–89 are FULL to their dense ceilings (slate §0);
# appended ids 100…154 continue the GLOBAL dense order prefixed by the owning
# band, so several prefixes carry a second (appended) range. Burned ids
# (121,130,132,134,135,136,140,153 + not-landed 106/109/110/115) simply hold
# no rows; SURF appended range extended to 104…139 for the euromonitor
# experiment surfaces (surfaces-fix + _link/08_pipeline); INFRA appended range
# extended to 121…149 for the graphify surface-reconciliation gate
# (GATE-INFRA-149); future mints append ≥155 (HPO) / ≥140 (SURF) / ≥150 (INFRA).
BAND_RANGES = {
    "INGEST": [(1, 9)],
    "ETL": [(10, 19), (115, 117)],
    "FEAT": [(20, 29)],
    "TRAIN": [(30, 39), (118, 120)],
    "STATS": [(40, 49), (113, 113)],
    "TLINE": [(50, 59), (114, 114)],
    "SURF": [(60, 69), (100, 103), (104, 139)],
    "CFG": [(70, 79), (103, 112)],
    "HPO": [(80, 89), (154, 154)],
    "INFRA": [(90, 99), (121, 149)],
    "CUST": [(147, 153)],
}
ID_RX = re.compile(r"^GATE-[A-Z]+-\d{2,3}$")
NON_GATE_TOKEN_RES = [
    re.compile(r"(?<![A-Z-])(ARTIFACT-[A-Z0-9]+(?:-[A-Z0-9]+)*)"),
    re.compile(r"(?<![A-Z-])(CFG-[A-Z0-9]+(?:-[A-Z0-9]+)*)"),
    re.compile(r"(?<![A-Z-])(VOCAB-[A-Z0-9]+(?:-[A-Z0-9]+)*)"),
]
GATE_TOKEN_RX = re.compile(
    r"\bGATE-[A-Z]+-\d{2,3}(?:/-?\d{2,3})*(?:\.\.\d{2,3})?(?!\d)")


def _load_registry() -> dict:
    with GATES_YAML.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


REG = _load_registry()
GATES = REG["gates"]
VOCABULARY = set(REG.get("vocabulary") or [])
GATE_IDS = {g["id"] for g in GATES}


def _expand_gate_token(token: str) -> set[str]:
    head = re.match(r"(GATE-[A-Z]+-)(\d{2,3})", token)
    out = {head.group(0)}
    for m in re.finditer(r"/-?(\d{2,3})", token):
        out.add(head.group(1) + m.group(1))
    rng = re.search(r"\.\.(\d{2,3})", token)
    if rng:
        start, end = int(head.group(2)), int(rng.group(1))
        width = max(2, len(str(end)))
        out.update(f"{head.group(1)}{k:0{width}d}" for k in range(start + 1, end + 1))
    return out


def _id_tokens(text: str) -> set[str]:
    out: set[str] = set()
    for rx in NON_GATE_TOKEN_RES:
        out.update(m.group(1) for m in rx.finditer(text))
    for m in GATE_TOKEN_RX.finditer(text):
        out |= _expand_gate_token(m.group(0))
    return out


def _band_of(gate_id: str) -> str:
    return {"INGEST": "01-ingest", "ETL": "02-etl-lookup", "FEAT": "03-features",
            "TRAIN": "04-training", "STATS": "05-stats", "TLINE": "06-timeline",
            "SURF": "07-surfaces", "CFG": "08-config", "HPO": "80-hpo-optuna",
            "INFRA": "09-infra", "CUST": "81-object-custody"}[
                gate_id.split("-")[1]]


def _renderer():
    spec = importlib.util.spec_from_file_location(
        "render_gates_under_test", REPO / "agents" / "tools" / "render_gates.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------- (a) counts


def test_meta_counts_match_body():
    counts = REG["meta"]["counts"]
    assert counts["total"] == len(GATES)
    assert counts["finding_strings"] == sum(len(g["findings"]) for g in GATES)
    assert counts["gates_with_findings"] == sum(1 for g in GATES if g["findings"])
    actual_bands = {}
    for gate in GATES:
        slot = actual_bands.setdefault(_band_of(gate["id"]),
                                       {"gates": 0, "flagged": 0, "finding_strings": 0})
        slot["gates"] += 1
        slot["flagged"] += 1 if gate["findings"] else 0
        slot["finding_strings"] += len(gate["findings"])
    for stem, expected in counts["by_band"].items():
        assert actual_bands[stem] == expected, f"counts.by_band[{stem}] stale"


def test_source_fragments_retired():
    """Post-retirement provenance guard: every path listed in
    meta.source_fragments must stay retired; verbatim content is preserved
    under agents/ledger/arbitration/2026-08-24/ (preservation file)."""
    fragments = REG["meta"]["source_fragments"]
    assert len(fragments) == 10
    for rel in fragments:
        assert not (REPO / rel).exists(), f"retired fragment resurrected: {rel}"
    assert not (REPO / "agents" / "ledger" / "GATES.md").exists(), \
        "retired GATES.md resurrected"


# --------------------------------------------------------------- (b) id shape


def test_ids_unique_and_wellformed():
    ids = [g["id"] for g in GATES]
    assert len(ids) == len(set(ids)), "duplicate gate ids in registry"
    bad = [gid for gid in ids if not ID_RX.match(gid)]
    assert bad == [], f"ids violating GATE-<PREFIX>-NN shape: {bad}"
    wrong_band = [g["id"] for g in GATES
                  if not any(lo <= int(g["id"].rsplit("-", 1)[1]) <= hi
                             for lo, hi in BAND_RANGES[g["id"].split("-")[1]])]
    assert wrong_band == [], f"id number outside its band range: {wrong_band}"


def test_order_dense_ascending():
    orders = [g["order"] for g in sorted(GATES, key=lambda g: g["order"])]
    assert orders == list(range(1, len(GATES) + 1)), "order must be dense 1..N"
    bands = [_band_of(g["id"]) for g in sorted(GATES, key=lambda g: g["order"])]
    canon = ["01-ingest", "02-etl-lookup", "03-features", "04-training", "05-stats",
             "06-timeline", "07-surfaces", "08-config", "80-hpo-optuna", "09-infra",
             "81-object-custody"]
    seen = [b for i, b in enumerate(bands) if i == 0 or bands[i - 1] != b]
    assert seen == canon, "bands must appear in canonical order (81+ ordinal, decoupled from id decades)"


def test_phase_uniform_per_band():
    phases = {}
    for gate in GATES:
        phases.setdefault(_band_of(gate["id"]), set()).add(gate["phase"])
    mixed = {b: p for b, p in phases.items() if len(p) != 1}
    assert mixed == {}, f"bands with inconsistent phase fields: {mixed}"


# ------------------------------------------------------- (c) owner anchoring


_FILE_CACHE: dict[str, str] = {}


def _repo_text(rel: Path) -> str | None:
    key = str(rel)
    if key not in _FILE_CACHE:
        try:
            _FILE_CACHE[key] = rel.read_text(encoding="utf-8", errors="replace")
        except OSError:
            _FILE_CACHE[key] = ""
    return _FILE_CACHE[key]


def _owner_segments(owner: str) -> list[str]:
    segments = []
    for part in re.split(r"\s+->\s+", owner):
        # split on ' + ' only where a further repo path follows
        segments.extend(re.split(r"\s+\+\s+(?=[\w.]+/)", part))
    return segments


def _owner_paths(owner: str) -> list[str]:
    paths = []
    for chunk in _owner_segments(owner):
        head = re.match(r"[^\s:]+", chunk)
        if not head:
            continue
        token = head.group(0)
        if "/" in token or re.search(r"\.[A-Za-z0-9]+$", token):
            paths.append(token)
    return paths


def _symbol_candidates(owner: str) -> tuple[list[str], list[str]]:
    primary: list[str] = []
    extra: list[str] = []
    siblings: list[str] = []
    for chunk in _owner_segments(owner):
        head = re.match(r"[^\s:]+", chunk)
        rest = chunk[head.end():].strip() if head else chunk
        rest = re.sub(r"^:\d+(?:-\d+)?\s*", "", rest)
        call = re.match(r"([A-Za-z_]\w*)\s*\(", rest)
        if call:
            primary.append(call.group(1))
        word = re.match(r"[A-Za-z_]\w{2,}", rest)
        if word:
            primary.append(word.group(0))
        extra.extend(w for w in re.findall(r"[A-Za-z_]\w{4,}", rest)[:6])
        extra.extend(t for t in re.findall(r"[\w][\w./\\-]{5,}", rest) if "/" in t)
        siblings.extend(re.findall(r"\b([A-Za-z_]\w*\.(?:py|sh|ya?ml|toml|cfg))\b", rest))

    def _dedupe(items: list[str]) -> list[str]:
        return list(dict.fromkeys(items))

    return _dedupe(primary) + _dedupe(extra), _dedupe(siblings)


@pytest.mark.parametrize("gate", GATES, ids=lambda g: g["id"])
def test_owner_path_exists_and_symbol_matches(gate):
    owner = gate["owner"]
    relpaths = [REPO / p for p in _owner_paths(owner)]
    assert relpaths, f"{gate['id']}: no repo path parsed from owner: {owner!r}"
    missing = [str(p) for p in relpaths if not p.exists()]
    assert missing == [], f"{gate['id']}: owner path(s) missing from repo: {missing}"
    primary = relpaths[0]
    names, siblings = _symbol_candidates(owner)
    contents = [_repo_text(p) for p in relpaths]
    hit = next((n for n in names if n in contents[0]), None) or next(
        (n for n in names for c in contents if n in c), None)
    if hit is None:
        hit = next((s for s in siblings
                    if (primary.parent / s).exists()), None)
    assert hit is not None, (
        f"{gate['id']}: none of the symbol candidates {names[:8]} appear in "
        f"{primary} (siblings tried: {siblings[:6]}) — owner prose drifted?"
    )


# --------------------------------............................ (d) validations


@pytest.fixture(scope="module")
def test_corpus() -> dict[str, str]:
    corpus: dict[str, str] = {}
    for base in ("tests", "project/tests"):
        root = REPO / base
        if not root.is_dir():
            continue
        for path in sorted(root.rglob("*.py")):
            corpus[str(path.relative_to(REPO))] = path.read_text(
                encoding="utf-8", errors="replace")
    return corpus


def _node_id_exists(node_id: str, corpus: dict[str, str]) -> bool:
    if any(node_id in text for text in corpus.values()):
        return True
    if "::" not in node_id:
        return False
    file_part, _, name_part = node_id.partition("::")
    text = None
    for rel, content in corpus.items():
        if rel == file_part or rel.endswith("/" + file_part):
            text = content
            break
    if text is None:
        return False
    bare = re.sub(r"\[[^\]]*\]", "", name_part).strip()
    if bare and bare in text:
        return True
    tokens = [t for t in re.findall(r"[A-Za-z_]\w{3,}", bare)]
    return any(t in text for t in tokens)


def test_validated_by_node_ids_exist(test_corpus):
    missing = []
    for gate in GATES:
        for node_id in gate["validated_by"]:
            if not _node_id_exists(node_id, test_corpus):
                missing.append(f"{gate['id']}: {node_id}")
    assert missing == [], (
        f"{len(missing)} validated_by node ids not found under tests/ or "
        f"project/tests/ (existence probe, not collection): {missing[:8]} ..."
    )


# --------------------------------------------------------- (e) closure check


def test_reference_closure_inputs_outputs_if_changed():
    non_io_union: set[str] = set()
    io_tokens: dict[str, set] = {}
    for gate in GATES:
        for field in ("owner", "transforms", "touched_by", "validated_by", "findings"):
            non_io_union |= _id_tokens(str(gate[field]))
    for gate in GATES:
        acc: set[str] = set()
        for field in ("inputs", "outputs", "if_changed"):
            acc |= _id_tokens(str(gate[field]))
        io_tokens[gate["id"]] = acc
    dangling = []
    for gid, tokens in io_tokens.items():
        others: set[str] = set()
        for other_gid, other_tokens in io_tokens.items():
            if other_gid != gid:
                others |= other_tokens
        for token in sorted(tokens):
            declared = (token in GATE_IDS or token in VOCABULARY
                        or token in non_io_union or token in others)
            if not declared:
                dangling.append((gid, token))
    assert dangling == [], (
        "id-like references in inputs/outputs/if_changed that resolve nowhere "
        f"(neither gate ids, vocabulary, nor any other registry occurrence): {dangling}"
    )


def test_vocabulary_entries_are_id_shaped_and_external():
    rx = re.compile(r"^(?:ARTIFACT|CFG|VOCAB)-[A-Z0-9]+(?:-[A-Z0-9]+)*$")
    bad = sorted(v for v in VOCABULARY if not rx.match(v))
    assert bad == [], f"vocabulary holds non-id-shaped entries: {bad}"
    overlap = sorted(VOCABULARY & GATE_IDS)
    assert overlap == [], f"vocabulary must not shadow gate ids: {overlap}"


# ------------------------------------------------------------ (f) round-trip


def test_renderer_roundtrip_byte_identical_and_help(tmp_path):
    renderer = _renderer()
    with pytest.raises(SystemExit) as excinfo:
        renderer.main(["--help"])
    assert excinfo.value.code == 0

    targets = [DIGEST] + INDEX_FILES
    assert renderer.main([]) == 0
    first = {p: p.read_bytes() for p in targets}
    assert renderer.main([]) == 0
    second = {p: p.read_bytes() for p in targets}
    drift = [str(p) for p in targets if first[p] != second[p]]
    assert drift == [], f"renderer output not idempotent for: {drift}"

    digest_text = DIGEST.read_text(encoding="utf-8")
    assert digest_text.startswith("# DIGEST.md — rendered from agents/ledger/gates.yaml")
    assert f"| **total** | | **{len(GATES)}** |" in digest_text
    assert digest_text.count("\n### ") == len({_band_of(g["id"]) for g in GATES}), \
        "digest must carry one section per band"

    scratch_digest = tmp_path / "digest_alt.md"
    scratch_index = tmp_path / "idx"
    assert renderer.main(["--digest", str(scratch_digest),
                          "--index-dir", str(scratch_index)]) == 0
    assert scratch_digest.read_bytes() == first[DIGEST]
    for rel in INDEX_FILES:
        assert (scratch_index / rel.name).read_bytes() == first[rel]


# --------------------------------------------------- (g) REG-QL query layer
# Appended by the QUERY-LAYER contract (REG-QL): read-only query surface on
# agents/tools/render_gates.py. Query modes print human-readable text to
# stdout, exit 0 on >=1 hit / 3 on zero hits, never mutate gates.yaml nor any
# rendered view, and leave the legacy default render path byte-identical.

import csv  # appended with the REG-QL section; stdlib only

QUERY_MODE_FLAGS = ["--gate", "--file", "--artifact", "--library", "--test",
                    "--findings", "--blast-radius", "--csv"]


def _run_renderer(capsys, argv: list[str]):
    code = _renderer().main(argv)
    captured = capsys.readouterr()
    return code, captured.out, captured.err


def test_help_documents_every_query_mode(capsys):
    renderer = _renderer()
    with pytest.raises(SystemExit) as excinfo:
        renderer.main(["--help"])
    assert excinfo.value.code == 0
    help_text = capsys.readouterr().out
    for flag in QUERY_MODE_FLAGS:
        assert flag in help_text, f"--help must document {flag}"
    assert "exit 3" in help_text, "zero-hit exit convention must be documented"


def test_query_gate_prints_full_yaml_entry(capsys):
    expected = next(g for g in GATES if g["id"] == "GATE-ETL-16")
    code, out, err = _run_renderer(capsys, ["--gate", "GATE-ETL-16"])
    assert code == 0 and err == ""
    loaded = yaml.safe_load(out)
    assert loaded == expected
    assert set(loaded) == {"id", "phase", "order", "owner", "inputs", "outputs",
                           "transforms", "touched_by", "validated_by",
                           "if_changed", "findings"}
    assert loaded["owner"].startswith("src/broadway/data/loader.py:112")


def test_query_gate_unknown_exit_3(capsys):
    code, out, err = _run_renderer(capsys, ["--gate", "GATE-NOPE-00"])
    assert code == 3 and out == "" and "GATE-NOPE-00" in err


def test_query_file_lists_owners_and_referencers(capsys):
    code, out, err = _run_renderer(capsys, ["--file", "project/etl/process.py"])
    assert code == 0 and err == ""
    ids = [line.split(" · ")[0] for line in out.splitlines()
           if line.startswith("GATE-")]
    # every owner gate under project/etl/process.py (by_owner_file.yaml keys)
    assert {"GATE-INGEST-04", "GATE-ETL-17"} <= set(ids)
    # GATE-INGEST-02 does not own the path but its if_changed range
    # GATE-INGEST-03..09 covers owner GATE-INGEST-04 -> must be listed too
    assert "GATE-INGEST-02" in ids
    owner_line = next(l for l in out.splitlines()
                      if l.startswith("GATE-INGEST-04 "))
    assert "· owns" in owner_line
    ref_line = next(l for l in out.splitlines()
                    if l.startswith("GATE-INGEST-02 "))
    assert "refs" in ref_line and "GATE-INGEST-04" in ref_line
    assert out.rstrip().endswith("=="), "sheet closes with a count footer"


def test_query_file_suffix_match_equals_full_path(capsys):
    _, full_out, _ = _run_renderer(capsys, ["--file", "project/etl/process.py"])
    code, suffix_out, _ = _run_renderer(capsys, ["--file", "etl/process.py"])
    assert code == 0
    gate_lines = lambda text: [l for l in text.splitlines()
                               if l.startswith("GATE-")]
    assert gate_lines(suffix_out) == gate_lines(full_out), \
        "suffix match must hit the same owner set"
    assert "== 3 gate(s) touch" in suffix_out


@pytest.mark.parametrize("argv", [
    ["--file", "no/such/path.py"],
    ["--artifact", "ARTIFACT-NOPE"],
    ["--library", "no-such-lib"],
    ["--test", "tests/no_such_test.py::nope"],
])
def test_query_zero_hits_exit_3_with_empty_stdout(capsys, argv):
    code, out, err = _run_renderer(capsys, argv)
    assert code == 3 and out == "" and err != ""


def test_query_artifact_marks_producers_and_consumers(capsys):
    code, out, err = _run_renderer(capsys, ["--artifact", "ARTIFACT-RAW-FRAME"])
    assert code == 0 and err == ""
    producer = next(l for l in out.splitlines() if l.startswith("GATE-INGEST-02 "))
    consumer = next(l for l in out.splitlines() if l.startswith("GATE-INGEST-04 "))
    assert producer.endswith("· produces")
    assert consumer.endswith("· consumes")


def test_query_library_touched_by_lookup(capsys):
    code, out, err = _run_renderer(capsys, ["--library", "pandas"])
    assert code == 0 and err == ""
    ids = {line.split(" · ")[0] for line in out.splitlines()
           if line.startswith("GATE-")}
    assert "GATE-INGEST-02" in ids          # bare 'pandas' touched_by entry
    assert "GATE-SURF-60" not in ids        # no pandas entry on that gate
    code2, out2, _ = _run_renderer(capsys, ["--library", "broadway.stats.anova"])
    assert code2 == 0 and "GATE-STATS-44" in out2


def test_query_test_node_substring(capsys):
    needle = "test_parse_numeric_fractional_records_failure_and_stays_float"
    code, out, err = _run_renderer(capsys, ["--test", needle])
    assert code == 0 and err == ""
    ids = {line.split(" · ")[0] for line in out.splitlines()
           if line.startswith("GATE-")}
    assert {"GATE-INGEST-06", "GATE-ETL-15"} <= ids


def test_query_findings_sheet_matches_meta_counts(capsys):
    code, out, err = _run_renderer(capsys, ["--findings"])
    assert code == 0 and err == ""
    body = [l for l in out.splitlines() if l.startswith("GATE-")]
    footer = out.strip().splitlines()[-1]
    assert len(body) == REG["meta"]["counts"]["gates_with_findings"] == 75
    assert footer == (f"== 75 flagged gate(s) · "
                      f"{REG['meta']['counts']['finding_strings']} finding(s) ==")
    parts = body[0].split(" · ")
    assert len(parts) == 4
    assert parts[0].startswith("GATE-") and parts[2].endswith("finding(s)")
    assert 0 < len(parts[3]) <= 140, "first sentence kept, truncated at 140"


def test_query_blast_radius_deterministic_ordered_and_pinned(capsys):
    argv = ["--blast-radius", "project/etl/process.py"]
    code1, out1, _ = _run_renderer(capsys, argv)
    code2, out2, _ = _run_renderer(capsys, argv)
    assert code1 == code2 == 0
    assert out1 == out2, "same input twice must give identical stdout"
    lines = out1.strip().splitlines()
    assert lines[0].startswith("blast-radius for project/etl/process.py:")
    gate_lines = [l for l in lines if l.startswith("d")]
    orders = [int(l.split()[1].rsplit("-", 1)[1]) for l in gate_lines]
    assert orders == sorted(orders), "affected gates ordered by phase/order"
    depths = {l.split()[1]: l.split()[0] for l in gate_lines}
    assert depths["GATE-INGEST-04"] == "d0"   # direct owner
    assert depths["GATE-ETL-17"] == "d0"      # direct owner
    assert depths["GATE-INGEST-02"] == "d0"   # referencer (range covers -04)
    assert lines[-1].startswith("pins (") and "none" not in lines[-1]


def test_query_blast_radius_one_hop_downstream_marked(capsys):
    _, out, _ = _run_renderer(capsys, ["--blast-radius", "project/etl/process.py"])
    depths = {l.split()[1]: l.split()[0]
              for l in out.splitlines() if l.startswith("d")}
    # depth 1 comes from seeds' outputs→inputs/if_changed closure:
    # GATE-INGEST-03..09 arrive via seed GATE-INGEST-02's GATE-INGEST-03..09
    # range, GATE-INGEST-04's GATE-INGEST-05..09 range, and ARTIFACT-RAW-FRAME
    # consumption.
    assert depths["GATE-INGEST-03"] == "d1"
    assert depths["GATE-INGEST-09"] == "d1"
    assert "GATE-ETL-16" not in depths, "unrelated bands stay outside the radius"


def test_query_csv_round_trip_row_count_header_and_joins(tmp_path, capsys):
    out_csv = tmp_path / "gates_flat.csv"
    code, out, err = _run_renderer(capsys, ["--csv", str(out_csv)])
    assert code == 0 and err == ""
    assert out.strip() == f"wrote {len(GATES)} gate row(s) + header to {out_csv}"
    with out_csv.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        assert reader.fieldnames == ["id", "phase", "order", "owner", "inputs",
                                     "outputs", "transforms", "touched_by",
                                     "validated_by", "if_changed", "findings"]
        rows = list(reader)
    assert len(rows) == len(GATES)
    by_id = {r["id"]: r for r in rows}
    assert by_id["GATE-INGEST-03"]["touched_by"] == ""      # empty list -> empty cell
    ingest06 = next(g for g in GATES if g["id"] == "GATE-INGEST-06")
    assert by_id["GATE-INGEST-06"]["findings"] == " ;; ".join(ingest06["findings"])
    ingest02 = next(g for g in GATES if g["id"] == "GATE-INGEST-02")
    assert by_id["GATE-INGEST-02"]["touched_by"] == " | ".join(ingest02["touched_by"])
    assert by_id["GATE-INGEST-02"]["validated_by"] == " | ".join(ingest02["validated_by"])


def test_query_csv_writes_only_the_named_file(tmp_path, capsys):
    target = tmp_path / "flat.csv"
    sentinel = tmp_path / "untouched.txt"
    sentinel.write_text("keep", encoding="utf-8")
    code, _, _ = _run_renderer(capsys, ["--csv", str(target)])
    assert code == 0
    assert sentinel.read_text(encoding="utf-8") == "keep"
    assert sorted(p.name for p in tmp_path.iterdir()) == ["flat.csv", "untouched.txt"]


def test_query_modes_leave_registry_and_rendered_views_byte_identical(capsys):
    watched = [GATES_YAML, DIGEST] + INDEX_FILES
    before = {p: p.read_bytes() for p in watched}
    for argv in (["--gate", "GATE-ETL-16"],
                 ["--file", "project/etl/process.py"],
                 ["--artifact", "ARTIFACT-CANONICAL-PARQUET"],
                 ["--library", "pyyaml"],
                 ["--test", "test_load_with_audit"],
                 ["--findings"],
                 ["--blast-radius", "src/broadway/data/loader.py"]):
        code, _, _ = _run_renderer(capsys, argv)
        assert code == 0, argv
    assert before == {p: p.read_bytes() for p in watched}, \
        "query modes must never mutate gates.yaml or rendered views"


def test_legacy_render_path_still_byte_idempotent(tmp_path):
    renderer = _renderer()
    scratch_digest = tmp_path / "digest.md"
    scratch_index = tmp_path / "idx"
    argv = ["--digest", str(scratch_digest), "--index-dir", str(scratch_index)]
    assert renderer.main(argv) == 0
    first = {p.name: p.read_bytes() for p in
             [scratch_digest, *scratch_index.iterdir()]}
    assert renderer.main(argv) == 0
    second = {p.name: p.read_bytes() for p in
              [scratch_digest, *scratch_index.iterdir()]}
    assert first == second, "default render stays byte-idempotent post-extension"


def test_query_modes_are_mutually_exclusive():
    renderer = _renderer()
    with pytest.raises(SystemExit) as excinfo:
        renderer.main(["--gate", "GATE-ETL-16", "--findings"])
    assert excinfo.value.code == 2, "argparse standard error for flag clashes"


# ------------------------------- (h) REG-TOOLING exit-law extension modes
# Appended by the REG-TOOLING contract: --symbol / --owner-exact / --has /
# --dupe-registry / --dedupe-proposals on agents/tools/render_gates.py.
# Same discipline as (g): read-only, human-readable stdout, exit 0 on >=1
# hit / 3 on zero hits / argparse 2, legacy default render untouched.

REG_TOOLING_FLAGS = ["--symbol", "--owner-exact", "--has",
                     "--dupe-registry", "--dedupe-proposals"]
ARBITRATION_DIR = REPO / "agents" / "ledger" / "arbitration" / "2026-08-24"
PROPOSAL_CLASSES = ("UNIQUE", "DUP-OF-GATE", "OVERLAP-PATH", "INTERNAL-DUP")


def _gate_lines(out: str) -> list[str]:
    return [line for line in out.splitlines() if line.startswith("GATE-")]


def _gate_ids(out: str) -> list[str]:
    return [line.split(" · ")[0] for line in _gate_lines(out)]


def test_reg_tooling_help_documents_new_modes(capsys):
    renderer = _renderer()
    with pytest.raises(SystemExit) as excinfo:
        renderer.main(["--help"])
    assert excinfo.value.code == 0
    help_text = capsys.readouterr().out
    for flag in REG_TOOLING_FLAGS:
        assert flag in help_text, f"--help must document {flag}"
    assert "exit 3" in help_text, "zero-hit exit convention must be documented"


def test_symbol_substring_covers_owner_and_transforms(capsys):
    needle = "canonicalize"
    code, out, err = _run_renderer(capsys, ["--symbol", needle])
    assert code == 0 and err == ""
    expected = {g["id"] for g in GATES
                if needle in str(g["owner"])
                or any(needle in str(t) for t in g["transforms"])}
    assert set(_gate_ids(out)) == expected
    assert "GATE-INGEST-08" in expected          # src/broadway/data/cleaner.py canonicalize()
    footer = out.strip().splitlines()[-1]
    assert footer == (f"== {len(expected)} gate(s) match symbol "
                      f"{needle} ==")


def test_symbol_miss_exits_3_with_stderr_note(capsys):
    code, out, err = _run_renderer(
        capsys, ["--symbol", "build_distance_features"])
    assert code == 3 and out == "" and "build_distance_features" in err


def test_owner_exact_matches_path_plus_symbol_token_prefix(capsys):
    spec = "project/etl/process.py:select_and_clean_columns"
    code, out, err = _run_renderer(capsys, ["--owner-exact", spec])
    assert code == 0 and err == ""
    # the known identical-owner pair claimed by two gates (dupe finding,
    # reported by --dupe-registry, never auto-fixed)
    assert _gate_ids(out) == ["GATE-INGEST-04", "GATE-ETL-17"]
    # Packet F de-anchored these owners ("numeric anchors stripped pending
    # DP-A07/R2") — the pre-repair `process.py:98` line-number probe no longer
    # resolves, so pin the pair through its current-truth symbol token instead.
    code2, out2, _ = _run_renderer(capsys, ["--owner-exact",
                                            "project/etl/process.py:select_and_clean_columns"])
    assert code2 == 0
    assert set(_gate_ids(out2)) == set(_gate_ids(out))


def test_owner_exact_rejects_suffix_paths_and_unknown_symbols(capsys):
    # unlike --file, owner-exact demands the FULL owner path (no suffix logic)
    code, out, _ = _run_renderer(
        capsys, ["--owner-exact", "etl/process.py:select_and_clean_columns"])
    assert code == 3 and out == ""
    code2, out2, _ = _run_renderer(
        capsys, ["--owner-exact", "project/etl/process.py:no_such_symbol_zz"])
    assert code2 == 3 and out2 == ""


def test_has_file_probe_names_tline_59(capsys):
    code, out, err = _run_renderer(capsys, ["--has", "cli.py"])
    assert code == 0 and err == ""
    assert "GATE-TLINE-59" in out                # src/broadway/cli.py main() owner
    assert "matched file 'cli.py'" in out, "winning probe kind must be named"


def test_has_step_modules_stays_a_drift_proof_miss(capsys):
    """STEP_MODULES (src/broadway/config/loader.py dispatch table) was the
    documented UNOWNED surface (arbitration PROPOSE-GH1-01). GATE-CFG-103
    now owns it (packet F #12, same-commit flip), so the ownership pin is
    --symbol (free-text probe); the --has cascade intentionally has no
    symbol probe, so a bare '--has STEP_MODULES' remains a miss by design —
    this dual pin fails if EITHER the ownership regresses OR the cascade
    silently grows a symbol probe."""
    code, out, err = _run_renderer(capsys, ["--has", "STEP_MODULES"])
    assert code == 3 and out == "" and "STEP_MODULES" in err
    code2, out2, _ = _run_renderer(capsys, ["--symbol", "STEP_MODULES"])
    assert code2 == 0 and any(
        l.startswith("GATE-CFG-103 ") for l in out2.splitlines()
    ), "dispatch-map chokepoint must remain owned by GATE-CFG-103"


def test_step_modules_dispatch_owned_by_cfg_103(capsys):
    """validated_by node for GATE-CFG-103: the STEP_MODULES dispatch map
    stays claimed, and its owner string keeps pointing at the dict literal
    in config/loader.py."""
    code, out, _ = _run_renderer(capsys, ["--gate", "GATE-CFG-103"])
    assert code == 0
    assert "loader.py" in out and "STEP_MODULES" in out
    code2, out2, _ = _run_renderer(capsys, ["--owner-exact",
                                            "src/broadway/config/loader.py:STEP_MODULES"])
    assert code2 == 0 and "GATE-CFG-103" in out2


def test_has_probe_cascade_first_hit_wins(capsys):
    code, out, _ = _run_renderer(capsys, ["--has", "GATE-ETL-16"])
    assert code == 0 and "matched gate-id 'GATE-ETL-16'" in out
    code2, out2, _ = _run_renderer(capsys, ["--has", "ARTIFACT-RAW-PARQUET"])
    assert code2 == 0
    assert set(_gate_ids(out2)) == {"GATE-INGEST-02"}
    assert "matched artifact 'ARTIFACT-RAW-PARQUET'" in out2
    node_id = next(g for g in GATES if g["id"] == "GATE-INGEST-06")[
        "validated_by"][0]
    code3, out3, _ = _run_renderer(capsys, ["--has", node_id])
    assert code3 == 0 and f"matched test-pin '{node_id}'" in out3
    code4, out4, _ = _run_renderer(capsys, ["--has", "pandas"])
    assert code4 == 0 and "matched library 'pandas'" in out4


def test_has_total_miss_exits_3(capsys):
    code, out, err = _run_renderer(capsys, ["--has", "no-such-thing-zz"])
    assert code == 3 and out == "" and "no-such-thing-zz" in err


def test_dupe_registry_reports_real_findings_on_live_ssot(capsys):
    """The 99-gate SSOT is NOT clean under the dupe definitions: known
    findings are pinned here verbatim (reported, never auto-fixed)."""
    argv = ["--dupe-registry"]
    code1, out1, err1 = _run_renderer(capsys, argv)
    code2, out2, _ = _run_renderer(capsys, argv)
    assert code1 == code2 == 3 and err1 == ""
    assert out1 == out2, "same scan twice must give identical stdout"
    assert "OWNER-COLLISIONS (" in out1
    assert "ARTIFACT-TWO-WRITER (" in out1
    assert "NEAR-IDENTICAL (" in out1
    assert "verdict: DUPE(S) FOUND" in out1
    assert ("project/etl/process.py :: select_and_clean_columns"
            " -> GATE-INGEST-04 + GATE-ETL-17") in out1
    assert "ARTIFACT-CANONICAL-PARQUET <- GATE-ETL-10 + GATE-ETL-16" in out1
    assert "src/broadway/training/hpo.py :: outs(1)" \
           " -> GATE-HPO-80 + GATE-HPO-82" in out1


def _write_gate(tmp_path, num, owner, outputs, inputs=(), name="gates.yaml"):
    path = tmp_path / name
    is_new = not path.exists()
    body = [
        f"- id: GATE-TEST-{num:02d}", "  phase: t", f"  order: {num}",
        f"  owner: {owner}",
        "  inputs: [" + ", ".join(inputs) + "]",
        "  outputs: [" + ", ".join(outputs) + "]",
        "  transforms: []", "  touched_by: []", "  validated_by: []",
        "  if_changed: []", "  findings: []",
    ]
    with path.open("a", encoding="utf-8") as fh:
        if is_new:
            fh.write("gates:\n")
        fh.write("\n".join(body) + "\n")
    return path


def test_dupe_registry_clean_synthetic_registry_exits_0(tmp_path, capsys):
    path = _write_gate(tmp_path, 1, "src/x/alpha.py:10 thing_one()",
                       ["ARTIFACT-A"])
    _write_gate(tmp_path, 2, "src/x/beta.py:20 thing_two()",
                ["ARTIFACT-B"], inputs=["ARTIFACT-A"])
    code, out, err = _run_renderer(
        capsys, ["--gates", str(path), "--dupe-registry"])
    assert code == 0 and err == ""
    assert "verdict: clean" in out
    assert out.count("\n  none") == 3, "all three classes must report none"


def test_dupe_registry_injected_dupes_flip_exit_to_3(tmp_path, capsys):
    path = _write_gate(tmp_path, 1, "src/x/alpha.py:10 thing_one()",
                       ["ARTIFACT-A"])
    _write_gate(tmp_path, 2, "src/x/beta.py:20 thing_two()",
                ["ARTIFACT-B"], inputs=["ARTIFACT-A"])
    _write_gate(tmp_path, 3, "src/x/alpha.py:10 thing_one()",
                ["ARTIFACT-A"])
    code, out, _ = _run_renderer(
        capsys, ["--gates", str(path), "--dupe-registry"])
    assert code == 3
    assert ("src/x/alpha.py :: thing_one"
            " -> GATE-TEST-01 + GATE-TEST-03") in out
    assert "ARTIFACT-A <- GATE-TEST-01 + GATE-TEST-03" in out
    assert "verdict: DUPE(S) FOUND" in out


def test_dedupe_proposals_smoke_arbitration_dir(capsys):
    code, out, err = _run_renderer(
        capsys, ["--dedupe-proposals", str(ARBITRATION_DIR)])
    assert code == 0 and err == ""
    header = out.splitlines()[0]
    m = re.search(r"(\d+) candidate\(s\) parsed from (\d+) file\(s\)", header)
    assert m, f"header must carry parse counts: {header}"
    assert int(m.group(1)) >= 40, "full arbitration census must parse >=40 candidates"
    rows = [line for line in out.splitlines() if " · PROPOSE-" in line]
    assert len(rows) == int(m.group(1))
    keys = [(line.split(" · ")[0], line.split(" · ")[1]) for line in rows]
    assert keys == sorted(keys), "table must be sorted by file then id"
    counted = dict.fromkeys(PROPOSAL_CLASSES, 0)
    for row in rows:
        family = row.split(" · ")[2].split(" (")[0]
        if family.startswith("DUP-OF-GATE-"):
            family = "DUP-OF-GATE"
        assert family in PROPOSAL_CLASSES, f"unknown classification: {family}"
        counted[family] += 1
    footer = out.strip().splitlines()[-1]
    assert footer.startswith("== class counts: ") and footer.endswith(" ==")
    for name in PROPOSAL_CLASSES:
        assert f"{name}={counted[name]}" in footer, \
            f"footer must carry {name}={counted[name]}"


def test_dedupe_proposals_unreadable_dir_exits_2(capsys):
    code, out, err = _run_renderer(
        capsys, ["--dedupe-proposals", str(REPO / "no_such_dir_zz")])
    assert code == 2 and out == "" and err != ""


def test_reg_tooling_modes_leave_views_byte_identical(capsys):
    watched = [GATES_YAML, DIGEST] + INDEX_FILES
    before = {p: p.read_bytes() for p in watched}
    for argv in (["--symbol", "canonicalize"],
                 ["--owner-exact", "project/etl/process.py:98"],
                 ["--has", "cli.py"],
                 ["--dupe-registry"],
                 ["--dedupe-proposals", str(ARBITRATION_DIR)]):
        code, _, _ = _run_renderer(capsys, argv)
        assert code in (0, 3), argv
    assert before == {p: p.read_bytes() for p in watched}, \
        "REG-TOOLING modes must never mutate gates.yaml or rendered views"


def test_reg_tooling_modes_are_mutually_exclusive():
    renderer = _renderer()
    with pytest.raises(SystemExit) as excinfo:
        renderer.main(["--symbol", "x", "--has", "y"])
    assert excinfo.value.code == 2, "argparse standard error for flag clashes"
