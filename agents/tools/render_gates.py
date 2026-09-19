"""Render views from the gate-registry SSOT (agents/ledger/gates.yaml).

Outputs (all deterministic; re-running produces byte-identical files):
  1. agents/ledger/index/by_owner_file.yaml  — owner repo-path -> gate ids
  2. agents/ledger/index/by_artifact.yaml    — ARTIFACT-*/CFG-*/VOCAB-* token
       (or the raw entry string when an input/output entry carries no id-like
       token) -> gate ids, over the union of every gate's inputs[] + outputs[]
  3. agents/ledger/index/by_library.yaml     — touched_by name before '@'
       -> gate ids
  4. agents/ledger/DIGEST.md                 — compact human digest (band table
       + per-gate bullets with ⚠FINDING flags), regenerated in the historical
       compact format. NEVER hand-edit; regenerate with this tool.

Query modes (REG-QL; read-only, mutually exclusive, human-readable plain text
on stdout, never mutate gates.yaml or any rendered view):
  --gate ID           print the single full YAML registry entry for that gate id
  --file RELPATH      gates whose owner path matches RELPATH (exact or suffix),
                      plus every gate whose inputs/outputs/if_changed reference
                      an id whose owner resolves under that path
                      (by_owner_file.yaml logic)
  --artifact ID       gates producing/consuming this ARTIFACT-/CFG-/VOCAB- id
                      (by_artifact.yaml logic)
  --library NAME      gates whose touched_by names NAME (the part before '@';
                      by_library.yaml logic)
  --test NODEID       gates whose validated_by contains this node-id substring
  --findings          compact risk sheet: one line per flagged gate plus a
                      summary count footer
  --blast-radius R    LOCATE mechanized (MAIN_AGENT_CONTRACT §14 step 1):
                      union of gates owning R and gates referencing ids owned
                      by R, closed one hop over their outputs → inputs/if_changed
                      downstream references; depth-marked, phase/order-ordered,
                      with a final line listing the distinct test node ids that
                      pin the affected set
  --csv OUTFILE       flatten every gate to one CSV row (list fields joined
                      with ' | ', findings with ' ;; '); writes ONLY OUTFILE

REG-TOOLING extensions (same read-only discipline and exit law):
  --symbol TEXT       case-sensitive substring over owner AND transforms
                      strings of all gates; ids+owners one per line
  --owner-exact SPEC  SPEC is 'path:symbol': gates whose owner path equals
                      path exactly and whose owner symbol token starts with
                      symbol (line numbers are advisory, never matched)
  --has THING         unified agent probe; tried in order: gate-id exact →
                      artifact id exact → validated_by node-id exact →
                      touched_by library exact → file path (--file logic) →
                      owner-exact ('path:symbol' form); first hit wins,
                      winning gate ids printed compactly with the probe kind
  --dupe-registry     SSOT hygiene scan: OWNER-COLLISIONS (identical
                      path:symbol claimed by >=2 gates), ARTIFACT-TWO-WRITER
                      (output artifact produced by >=2 gates, by_artifact key
                      logic on outputs[]), NEAR-IDENTICAL (same owner path +
                      same outputs set, different ids). Grouped report;
                      findings are reported, never auto-fixed
  --dedupe-proposals DIR
                      parse candidate blocks from *.md in DIR: blocks start
                      at a line '- id: PROPOSE-' and run until the next
                      '- id:' or blank-line-then-'#'; '### PROPOSE-…' headings
                      and '- **PROPOSE-…' bullets are also parsed best-effort
                      so full arbitration sheets count. Each candidate is
                      classified UNIQUE / DUP-OF-GATE-<id> / OVERLAP-PATH /
                      INTERNAL-DUP vs registry + within-set; report mode

Exit codes: query modes exit 0 on >=1 hit and 3 on zero hits (the miss note
goes to stderr so stdout stays parseable) — --dupe-registry exits 3 when any
dupe class is non-empty and --dedupe-proposals exits 0 unless DIR is
unreadable (2); argparse errors keep the standard exit 2. With no query flag
the legacy render path runs exactly as before.

Only the standard library is required at import time; PyYAML (already a direct
project dependency, see pyproject.toml) is imported lazily so that
`python agents/tools/render_gates.py --help` works anywhere.
"""
from __future__ import annotations

import argparse
import datetime
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_GATES = REPO_ROOT / "agents" / "ledger" / "gates.yaml"
DEFAULT_INDEX_DIR = REPO_ROOT / "agents" / "ledger" / "index"
DEFAULT_DIGEST = REPO_ROOT / "agents" / "ledger" / "DIGEST.md"

# Gate-id prefix -> canonical band directory stem (canonical band order 01→09;
# the appended region 100+ continues the dense order — ids 147+ mint band
# 81-object-custody per arbitration slate §0 / packet H slot arithmetic).
BAND_STEMS = {
    "INGEST": "01-ingest",
    "ETL": "02-etl-lookup",
    "FEAT": "03-features",
    "TRAIN": "04-training",
    "STATS": "05-stats",
    "TLINE": "06-timeline",
    "SURF": "07-surfaces",
    "CFG": "08-config",
    "HPO": "80-hpo-optuna",
    "INFRA": "09-infra",
    "CUST": "81-object-custody",
}

ID_TOKEN_RES = [
    r"(?<![A-Z-])ARTIFACT-[A-Z0-9]+(?:-[A-Z0-9]+)*",
    r"(?<![A-Z-])CFG-[A-Z0-9]+(?:-[A-Z0-9]+)*",
    r"(?<![A-Z-])VOCAB-[A-Z0-9]+(?:-[A-Z0-9]+)*",
]


def load_registry(path: Path | str | None = None) -> dict:
    """Load gates.yaml (PyYAML imported lazily; it is a project dependency)."""
    import yaml  # deferred so --help never requires it

    path = Path(path) if path is not None else DEFAULT_GATES
    with Path(path).open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def dump_yaml(data: object, sort_keys: bool) -> str:
    import yaml

    return yaml.safe_dump(
        data, sort_keys=sort_keys, allow_unicode=True, width=10**6,
        default_flow_style=False,
    )


def band_of(gate_id: str) -> str:
    return BAND_STEMS[gate_id.split("-")[1]]


def extract_non_gate_tokens(text: str) -> set[str]:
    out: set[str] = set()
    for pattern in ID_TOKEN_RES:
        out.update(m.group(0) for m in re.finditer(pattern, text))
    return out


def build_by_owner_file(registry: dict) -> dict[str, list[str]]:
    """owner leading repo-path token -> gate ids in canonical order."""
    idx: dict[str, list[str]] = {}
    for gate in sorted(registry["gates"], key=lambda g: g["order"]):
        head = re.match(r"[^\s:]+", gate["owner"]).group(0)
        idx.setdefault(head, []).append(gate["id"])
    return dict(sorted(idx.items()))


def build_by_artifact(registry: dict) -> dict[str, list[str]]:
    """inputs ∪ outputs id-like tokens (else raw entry) -> gate ids."""
    idx: dict[str, list[str]] = {}
    for gate in sorted(registry["gates"], key=lambda g: g["order"]):
        entries = list(gate["inputs"]) + list(gate["outputs"])
        keys = set()
        for entry in entries:
            keys |= extract_non_gate_tokens(entry) or {entry.strip()}
        for key in sorted(keys):
            idx.setdefault(key, []).append(gate["id"])
    return dict(sorted(idx.items()))


def build_by_library(registry: dict) -> dict[str, list[str]]:
    """touched_by name before '@' -> gate ids."""
    idx: dict[str, list[str]] = {}
    for gate in sorted(registry["gates"], key=lambda g: g["order"]):
        for entry in gate["touched_by"]:
            lib = entry.split("@")[0].strip()
            idx.setdefault(lib, []).append(gate["id"])
    return dict(sorted(idx.items()))


# ------------------------------------------------- REG-QL query layer (read-only)

QUERY_EXIT_OK = 0
QUERY_EXIT_MISS = 3

# GATE ids incl. slash lists (GATE-X-01/-02) and open ranges (GATE-X-03..09).
GATE_REF_RX = re.compile(r"\bGATE-[A-Z]+-\d{2}(?:/-?\d{2})*(?:\.\.\d{2})?")

CSV_COLUMNS = ["id", "phase", "order", "owner", "inputs", "outputs",
               "transforms", "touched_by", "validated_by", "if_changed",
               "findings"]


def normalize_rel(rel: str) -> str:
    rel = rel.strip()
    while rel.startswith("./"):
        rel = rel[2:]
    return rel.rstrip("/")


def owner_head(owner: str) -> str:
    """Leading repo-path token of an owner string (= by_owner_file.yaml key)."""
    return re.match(r"[^\s:]+", str(owner)).group(0)


def path_matches(token: str, rel: str) -> bool:
    """Exact or suffix match of an owner path against a repo-relative path."""
    return token == rel or token.endswith("/" + rel)


def fields_text(gate: dict, fields: tuple[str, ...]) -> str:
    return "\n".join(str(entry) for field in fields for entry in gate[field])


def expand_gate_refs(text: str) -> set[str]:
    """Gate ids referenced in registry text; slash lists and ranges expanded."""
    out: set[str] = set()
    for m in GATE_REF_RX.finditer(text):
        head = re.match(r"(GATE-[A-Z]+-)(\d{2})", m.group(0))
        out.add(head.group(0))
        for extra in re.finditer(r"/-?(\d{2})", m.group(0)):
            out.add(head.group(1) + extra.group(1))
        if ".." in m.group(0):
            start, end = int(head.group(2)), int(m.group(0).split("..")[1][:2])
            out.update(f"{head.group(1)}{k:02d}" for k in range(start + 1, end + 1))
    return out


def entry_keys(entry: object) -> set[str]:
    """by_artifact.yaml key logic: id-like tokens, else the raw entry string."""
    text = str(entry)
    return extract_non_gate_tokens(text) or {text.strip()}


def gates_in_order(registry: dict) -> list[dict]:
    return sorted(registry["gates"], key=lambda g: g["order"])


def gates_owning(gates: list[dict], rel: str) -> list[dict]:
    return [g for g in gates if path_matches(owner_head(g["owner"]), rel)]


def gates_referencing(gates: list[dict], owned_ids: set[str]) -> dict[str, set[str]]:
    """gate id -> owned ids its inputs[]/outputs[]/if_changed[] reference."""
    hits: dict[str, set[str]] = {}
    for g in gates:
        refs = expand_gate_refs(
            fields_text(g, ("inputs", "outputs", "if_changed"))) & owned_ids
        if refs:
            hits[g["id"]] = refs
    return hits


def consumer_index(gates: list[dict]) -> dict[str, set[str]]:
    """artifact/config key -> gate ids declaring it in inputs[] or if_changed[]."""
    idx: dict[str, set[str]] = {}
    for g in gates:
        for entry in list(g["inputs"]) + list(g["if_changed"]):
            for key in entry_keys(entry):
                idx.setdefault(key, set()).add(g["id"])
    return idx


def owner_short(owner: str, limit: int = 64) -> str:
    """Compact owner for one-line sheets: path+symbol, line numbers stripped."""
    text = str(owner).split("  #", 1)[0]
    text = re.sub(r":\d+(?:-\d+)?", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) > limit:
        text = text[: limit - 1].rstrip() + "…"
    return text


def first_sentence(text: str, limit: int = 140) -> str:
    m = re.search(r"[.!?](?:\s+|$)", str(text))
    sentence = (str(text)[: m.end()] if m else str(text)).strip()
    if len(sentence) > limit:
        sentence = sentence[: limit - 1].rstrip() + "…"
    return sentence


def _hit_lines(lines: list[str]) -> int:
    print("\n".join(lines))
    return QUERY_EXIT_OK


def _miss(message: str) -> int:
    print(f"no matches: {message}", file=sys.stderr)
    return QUERY_EXIT_MISS


def query_gate(registry: dict, gate_id: str) -> int:
    for gate in registry["gates"]:
        if gate["id"] == gate_id:
            sys.stdout.write(dump_yaml(gate, sort_keys=False))
            return QUERY_EXIT_OK
    return _miss(f"no gate id {gate_id!r}")


def query_file(registry: dict, relpath: str) -> int:
    rel = normalize_rel(relpath)
    gates = gates_in_order(registry)
    owners = gates_owning(gates, rel)
    owned_ids = {g["id"] for g in owners}
    refs = gates_referencing(gates, owned_ids)
    if not owned_ids and not refs:
        return _miss(f"no gate owns or references {relpath!r}")
    lines: list[str] = []
    for g in gates:
        roles: list[str] = []
        if g["id"] in owned_ids:
            roles.append("owns")
        if g["id"] in refs:
            roles.append("refs " + ",".join(sorted(refs[g["id"]])))
        if roles:
            lines.append(f"{g['id']} · {g['owner']} · {'+'.join(roles)}")
    n = len(lines)
    lines.append(f"== {n} gate(s) touch {rel} ==")
    return _hit_lines(lines)


def query_artifact(registry: dict, artifact: str) -> int:
    lines: list[str] = []
    for g in gates_in_order(registry):
        produces = any(artifact in entry_keys(e) for e in g["outputs"])
        consumes = any(artifact in entry_keys(e) for e in g["inputs"])
        if produces or consumes:
            role = ("produces+consumes" if produces and consumes
                    else "produces" if produces else "consumes")
            lines.append(f"{g['id']} · {g['owner']} · {role}")
    if not lines:
        return _miss(f"artifact {artifact!r} is not produced/consumed by any gate")
    n = len(lines)
    lines.append(f"== {n} gate(s) produce/consume {artifact} ==")
    return _hit_lines(lines)


def query_library(registry: dict, name: str) -> int:
    lines: list[str] = []
    for g in gates_in_order(registry):
        pins = sum(1 for e in g["touched_by"]
                   if str(e).split("@")[0].strip() == name)
        if pins:
            lines.append(f"{g['id']} · {g['owner']} · {pins} touched_by hit(s)")
    if not lines:
        return _miss(f"library/module {name!r} named in no touched_by entry")
    n = len(lines)
    lines.append(f"== {n} gate(s) touched_by {name} ==")
    return _hit_lines(lines)


def query_test(registry: dict, node_substring: str) -> int:
    lines: list[str] = []
    for g in gates_in_order(registry):
        pins = [v for v in g["validated_by"] if node_substring in v]
        if pins:
            lines.append(f"{g['id']} · {g['owner']} · {len(pins)} pin(s)")
    if not lines:
        return _miss(f"node-id substring {node_substring!r} in no validated_by pin")
    n = len(lines)
    lines.append(f"== {n} gate(s) pinned by {node_substring} ==")
    return _hit_lines(lines)


def query_findings(registry: dict) -> int:
    lines: list[str] = []
    total = 0
    for g in gates_in_order(registry):
        if not g["findings"]:
            continue
        total += len(g["findings"])
        lines.append(f"{g['id']} · {owner_short(g['owner'])}"
                     f" · {len(g['findings'])} finding(s)"
                     f" · {first_sentence(g['findings'][0])}")
    if not lines:
        return _miss("no flagged gates in this registry")
    lines.append(f"== {len(lines)} flagged gate(s) · {total} finding(s) ==")
    return _hit_lines(lines)


def query_blast_radius(registry: dict, relpath: str) -> int:
    rel = normalize_rel(relpath)
    gates = gates_in_order(registry)
    owners = gates_owning(gates, rel)
    owned_ids = {g["id"] for g in owners}
    refs = gates_referencing(gates, owned_ids)
    seed_ids = owned_ids | set(refs)
    if not seed_ids:
        return _miss(f"no gate owns or references {relpath!r}; blast radius empty")
    cmap = consumer_index(gates)
    downstream: set[str] = set()
    for g in gates:
        if g["id"] not in seed_ids:
            continue
        produced: set[str] = set()
        for entry in g["outputs"]:
            produced |= entry_keys(entry)
        for key in produced:
            downstream |= cmap.get(key, set())
        downstream |= expand_gate_refs(fields_text(g, ("outputs", "if_changed")))
    downstream -= seed_ids
    affected = [g for g in gates if g["id"] in seed_ids or g["id"] in downstream]
    pins = sorted({node for g in affected for node in g["validated_by"]})
    lines = [f"blast-radius for {rel}: {len(affected)} gate(s) affected "
             f"({len(seed_ids)} at depth 0 — own/reference;"
             f" {len(downstream)} at depth 1 — one-hop downstream)"]
    for g in affected:
        depth = 0 if g["id"] in seed_ids else 1
        lines.append(f"d{depth} {g['id']} · {g['phase']} · {g['owner']}")
    lines.append(f"pins ({len(pins)} distinct test node id(s)): "
                 + (", ".join(pins) if pins else "none"))
    return _hit_lines(lines)


def query_csv(registry: dict, outfile) -> int:
    import csv

    rows = 0
    with Path(outfile).open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(CSV_COLUMNS)
        for g in gates_in_order(registry):
            writer.writerow([
                g["id"], g["phase"], g["order"], g["owner"],
                " | ".join(str(e) for e in g["inputs"]),
                " | ".join(str(e) for e in g["outputs"]),
                " | ".join(str(e) for e in g["transforms"]),
                " | ".join(str(e) for e in g["touched_by"]),
                " | ".join(str(e) for e in g["validated_by"]),
                " | ".join(str(e) for e in g["if_changed"]),
                " ;; ".join(str(e) for e in g["findings"]),
            ])
            rows += 1
    print(f"wrote {rows} gate row(s) + header to {outfile}")
    return QUERY_EXIT_OK if rows else QUERY_EXIT_MISS


# ------------------------------------------------- REG-TOOLING extension layer

LINE_NUM_RX = re.compile(r"^\s*\d+(?:[-–]\d+)?\s*")
IDENT_TOKEN_RX = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")

PROPOSE_ID_RX = re.compile(r"^\s*- id:\s*(PROPOSE-[A-Za-z0-9-]+)\s*(.*)$")
PROPOSE_HEADING_RX = re.compile(r"^#{2,4}\s*(PROPOSE-[A-Za-z0-9-]+)\b\s*(.*)$")
PROPOSE_BOLD_RX = re.compile(r"^-\s*\*\*(PROPOSE-[A-Za-z0-9-]+)\b\*?\*(.*)$")
PROPOSE_FIELD_RX = re.compile(
    r"^\s+(owner|inputs|outputs|transforms|touched_by|validated_by|"
    r"if_changed|phase|order):\s*(.*)$")
PROPOSE_LIST_ITEM_RX = re.compile(r"^\s+-\s+(.*)$")
PROPOSE_INLINE_SEP_RX = re.compile(
    r"(?:^|\s)·\s*(owner|inputs|outputs|transforms|touched_by|validated_by|"
    r"if_changed|phase):\s*")
PATHISH_RX = re.compile(
    r"`?([A-Za-z0-9_][A-Za-z0-9_./-]*\.(?:py|ya?ml|sh|toml|cfg|md))`?")
ALLOWED_PROPOSAL_CLASSES = ("UNIQUE", "DUP-OF-GATE", "OVERLAP-PATH",
                            "INTERNAL-DUP")


def owner_symbol_token(owner: str) -> str:
    """First identifier token after the owner path's ':' (line numbers are
    advisory per house law, so they are stripped before matching)."""
    _, sep, rest = str(owner).partition(":")
    if not sep:
        return ""
    rest = LINE_NUM_RX.sub("", rest, count=1)
    m = IDENT_TOKEN_RX.search(rest)
    return m.group(0) if m else ""


def symbol_text_tokens(text: str) -> list[str]:
    """Word tokens of a post-colon owner symbol text (ids + line numbers)."""
    return re.findall(r"[A-Za-z0-9_]+", text)


def owner_exact_hits(registry: dict, spec: str) -> list[dict]:
    """Gates whose owner path equals spec's path (exact, no suffix logic) and
    whose owner symbol token starts with spec's part after ':'."""
    path, sep, sym = spec.partition(":")
    if not sep:
        return []
    path = normalize_rel(path)
    out: list[dict] = []
    for g in gates_in_order(registry):
        owner = str(g["owner"])
        if owner_head(owner) != path:
            continue
        if any(tok.startswith(sym)
               for tok in symbol_text_tokens(owner.partition(":")[2])):
            out.append(g)
    return out


def query_symbol(registry: dict, needle: str) -> int:
    lines: list[str] = []
    for g in gates_in_order(registry):
        hay = "\n".join([str(g["owner"])] + [str(t) for t in g["transforms"]])
        if needle in hay:
            lines.append(f"{g['id']} · {g['owner']}")
    if not lines:
        return _miss(f"symbol substring {needle!r} in no owner/transforms string")
    n = len(lines)
    lines.append(f"== {n} gate(s) match symbol {needle} ==")
    return _hit_lines(lines)


def query_owner_exact(registry: dict, spec: str) -> int:
    hits = owner_exact_hits(registry, spec)
    if not hits:
        return _miss(f"no gate owner matches {spec!r} "
                     "(owner-exact needs the 'path:symbol' form)")
    lines = [f"{g['id']} · {g['owner']}" for g in hits]
    n = len(lines)
    lines.append(f"== {n} gate(s) owner-exact {spec} ==")
    return _hit_lines(lines)


def _has_hit_sheet(hits: list[dict], kind: str, thing: str) -> int:
    ids = [g["id"] for g in hits]
    lines = list(ids)
    lines.append(f"== {len(ids)} gate(s) · matched {kind} '{thing}' ==")
    return _hit_lines(lines)


def query_has(registry: dict, thing: str) -> int:
    """Unified agent boilerplate probe; first hit in the cascade wins."""
    gates = gates_in_order(registry)
    exact = [g for g in registry["gates"] if g["id"] == thing]
    if exact:
        return _has_hit_sheet(exact, "gate-id", thing)
    artifact = [g for g in gates
                if any(thing in entry_keys(e)
                       for e in list(g["inputs"]) + list(g["outputs"]))]
    if artifact:
        return _has_hit_sheet(artifact, "artifact", thing)
    pin = [g for g in gates if thing in g["validated_by"]]
    if pin:
        return _has_hit_sheet(pin, "test-pin", thing)
    library = [g for g in gates
               if any(str(e).split("@")[0].strip() == thing
                      for e in g["touched_by"])]
    if library:
        return _has_hit_sheet(library, "library", thing)
    rel = normalize_rel(thing)
    owners = gates_owning(gates, rel)
    owned_ids = {g["id"] for g in owners}
    refs = gates_referencing(gates, owned_ids)
    if owned_ids or refs:
        hits = [g for g in gates if g["id"] in owned_ids or g["id"] in refs]
        return _has_hit_sheet(hits, "file", rel)
    hits = owner_exact_hits(registry, thing)
    if hits:
        return _has_hit_sheet(hits, "owner-exact", thing)
    return _miss(f"nothing in the registry matches {thing!r} under any probe "
                 "(gate-id/artifact/test-pin/library/file/owner-exact)")


def dupe_scan(registry: dict) -> tuple[list[str], bool]:
    """SSOT hygiene scan -> (report lines, dupes_found?). Findings only —
    never auto-fixed here."""
    gates = gates_in_order(registry)

    by_owner: dict[tuple[str, str], list[str]] = {}
    for g in gates:
        owner = str(g["owner"])
        sym = owner_symbol_token(owner)
        if sym:
            by_owner.setdefault((owner_head(owner), sym), []).append(g["id"])
    collisions = {k: v for k, v in sorted(by_owner.items()) if len(v) > 1}

    produced: dict[str, set[str]] = {}
    for g in gates:
        for e in g["outputs"]:
            for key in entry_keys(e):
                produced.setdefault(key, set()).add(g["id"])
    two_writer = {k: sorted(v) for k, v in sorted(produced.items())
                  if len(v) > 1}

    sig: dict[tuple[str, frozenset[str]], set[str]] = {}
    for g in gates:
        outs = frozenset(k for e in g["outputs"] for k in entry_keys(e))
        sig.setdefault((owner_head(str(g["owner"])), outs), set()).add(g["id"])
    near = {k: sorted(v) for k, v in sig.items() if len(v) > 1}

    lines = [f"dupe-scan · {len(gates)} gate(s)"]
    lines.append(f"OWNER-COLLISIONS ({len(collisions)} identical path:symbol "
                 "claim(s)):")
    for (path, sym), ids in collisions.items():
        lines.append(f"  {path} :: {sym} -> {' + '.join(ids)}")
    if not collisions:
        lines.append("  none")
    lines.append(f"ARTIFACT-TWO-WRITER ({len(two_writer)} output artifact(s) "
                 "produced by >=2 gates):")
    for key, ids in two_writer.items():
        lines.append(f"  {key} <- {' + '.join(ids)}")
    if not two_writer:
        lines.append("  none")
    lines.append(f"NEAR-IDENTICAL ({len(near)} same-owner-path+same-outputs "
                 "group(s)):")
    for (path, outs), ids in sorted(near.items(), key=lambda kv: sorted(kv[1])):
        lines.append(f"  {path} :: outs({len(outs)}) -> {' + '.join(ids)}")
    if not near:
        lines.append("  none")
    found = bool(collisions or two_writer or near)
    if found:
        lines.append(f"verdict: DUPE(S) FOUND — {len(collisions)} owner-"
                     f"collision(s), {len(two_writer)} two-writer artifact(s), "
                     f"{len(near)} near-identical group(s)")
    else:
        lines.append("verdict: clean — no owner collisions, single-writer "
                     "artifacts, no near-identical gates")
    return lines, found


def query_dupe_registry(registry: dict, source_label) -> int:
    lines, found = dupe_scan(registry)
    lines[0] += f" · {source_label}"
    print("\n".join(lines))
    return QUERY_EXIT_MISS if found else QUERY_EXIT_OK


def _clean_field(value: str) -> str:
    return value.replace("`", "").strip().strip("'\"").strip()


def _split_list_field(value: str) -> list[str]:
    text = _clean_field(value)
    if text.startswith("[") and text.endswith("]"):
        text = text[1:-1]
    return [part.strip() for part in text.split(",") if part.strip()]


def _proposal_owner_from_body(body: str) -> str:
    m = PATHISH_RX.search(body.replace("`", ""))
    return m.group(1) if m else ""


def _parse_proposal_block(start_kind: str, start_line: str,
                          body_lines: list[str]) -> dict:
    cand: dict = {"id": "", "owner": "", "inputs": [], "outputs": []}
    start_m = (PROPOSE_ID_RX.match(start_line)
               or PROPOSE_HEADING_RX.match(start_line)
               or PROPOSE_BOLD_RX.match(start_line))
    cand["id"] = start_m.group(1) if start_m else ""
    if start_kind == "id":
        marks = [(m.group(1), m.start(1))
                 for m in PROPOSE_INLINE_SEP_RX.finditer(start_line)]
        for idx, (key, pos) in enumerate(marks):
            end = marks[idx + 1][1] if idx + 1 < len(marks) else len(start_line)
            value = start_line[pos:end].partition(":")[2]
            if key == "owner":
                cand["owner"] = _clean_field(value)
            elif key in ("inputs", "outputs"):
                cand[key] = _split_list_field(value)
        current_list: str | None = None
        for line in body_lines:
            fm = PROPOSE_FIELD_RX.match(line)
            if fm:
                key, value = fm.group(1), fm.group(2)
                current_list = key if key in ("inputs", "outputs") else None
                if key == "owner":
                    cand["owner"] = _clean_field(value)
                elif key in ("inputs", "outputs"):
                    cand[key] = _split_list_field(value)
            elif current_list:
                im = PROPOSE_LIST_ITEM_RX.match(line)
                if im:
                    cand[current_list].append(_clean_field(im.group(1)))
                else:
                    current_list = None
    else:  # heading / bold-bullet variants: best-effort owner from prose
        cand["owner"] = _proposal_owner_from_body("\n".join(body_lines))
    return cand


def parse_proposal_candidates(dirpath: Path) -> tuple[list[dict], list[str]]:
    """Candidate blocks from *.md under dirpath, scan order. Returns
    (candidates with file/id/owner/inputs/outputs, unreadable-file notes)."""
    cands: list[dict] = []
    notes: list[str] = []
    starters = (("id", PROPOSE_ID_RX), ("heading", PROPOSE_HEADING_RX),
                ("bold", PROPOSE_BOLD_RX))
    for md in sorted(dirpath.glob("*.md")):
        try:
            lines = md.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError as exc:
            notes.append(f"{md.name}: unreadable ({exc})")
            continue
        i = 0
        while i < len(lines):
            kind = next((k for k, rx in starters if rx.match(lines[i])), None)
            if kind is None:
                i += 1
                continue
            block: list[str] = []
            prev_blank = False
            j = i + 1
            while j < len(lines):
                line = lines[j]
                if any(rx.match(line) for _, rx in starters):
                    break
                if line.startswith("```"):
                    break
                if prev_blank and line.startswith("#"):
                    break
                block.append(line)
                prev_blank = line.strip() == ""
                j += 1
            cand = _parse_proposal_block(kind, lines[i], block)
            cand["file"] = md.name
            cands.append(cand)
            i = j
    return cands, notes


def classify_proposals(cands: list[dict],
                       registry: dict) -> list[tuple[dict, str, str]]:
    """-> [(candidate, CLASS, detail)] in input order. Precedence:
    INTERNAL-DUP > DUP-OF-GATE-* > OVERLAP-PATH > UNIQUE."""

    def norm_owner(cand: dict) -> str:
        return re.sub(r"\s+", " ", cand["owner"]).strip()

    gate_syms: dict[str, list[tuple[str, str]]] = {}
    registry_ids = {g["id"] for g in registry["gates"]}
    for g in gates_in_order(registry):
        owner = str(g["owner"])
        gate_syms.setdefault(owner_head(owner), []).append(
            (owner_symbol_token(owner), g["id"]))
    gate_paths = set(gate_syms)

    id_counts: dict[str, int] = {}
    owner_counts: dict[str, int] = {}
    for cand in cands:
        id_counts[cand["id"]] = id_counts.get(cand["id"], 0) + 1
        key = norm_owner(cand)
        owner_counts[key] = owner_counts.get(key, 0) + 1

    out: list[tuple[dict, str, str]] = []
    for cand in cands:
        owner_norm = norm_owner(cand)
        path_m = re.match(r"[^\s:]+", owner_norm)
        path = normalize_rel(path_m.group(0)) if path_m else ""
        sym_tok = owner_symbol_token("x:" + owner_norm.partition(":")[2])
        if id_counts[cand["id"]] > 1 or owner_counts[owner_norm] > 1:
            cls = "INTERNAL-DUP"
            detail = ("duplicate id" if id_counts[cand["id"]] > 1
                      else "identical owner within proposals")
        elif cand["id"] in registry_ids:
            cls, detail = f"DUP-OF-GATE-{cand['id']}", "id collision"
        elif path and sym_tok and any(gs.startswith(sym_tok) and gs
                                      for gs, _ in gate_syms.get(path, [])):
            gid = next(gid for gs, gid in sorted(gate_syms[path])
                       if gs.startswith(sym_tok))
            cls, detail = f"DUP-OF-GATE-{gid}", "owner-exact match"
        elif path and path in gate_paths:
            cls, detail = "OVERLAP-PATH", path
        else:
            cls, detail = "UNIQUE", ""
        out.append((cand, cls, detail))
    return out


def query_dedupe_proposals(dirpath: Path, registry: dict) -> int:
    directory = Path(dirpath)
    if not directory.is_dir():
        print(f"error: proposals dir is not a readable directory: {directory}",
              file=sys.stderr)
        return 2
    cands, notes = parse_proposal_candidates(directory)
    classified = classify_proposals(cands, registry)
    files = sorted({c["file"] for c, _, _ in classified})
    lines = [f"dedupe-proposals · {len(classified)} candidate(s) parsed from "
             f"{len(files)} file(s) · {directory}"]
    lines.extend(f"note: {n}" for n in notes)
    for cand, cls, detail in sorted(classified,
                                    key=lambda t: (t[0]["file"], t[0]["id"])):
        row = (f"{cand['file']} · {cand['id']} · {cls}"
               + (f" ({detail})" if detail else "")
               + f" · ins={len(cand['inputs'])} outs={len(cand['outputs'])}"
               + (f" · owner={owner_short(cand['owner'], limit=48)}"
                  if cand["owner"] else ""))
        lines.append(row)
    counts: dict[str, int] = {}
    for _, cls, _ in classified:
        family = "DUP-OF-GATE" if cls.startswith("DUP-OF-GATE-") else cls
        counts[family] = counts.get(family, 0) + 1
    footer = " · ".join(f"{name}={counts.get(name, 0)}"
                        for name in ALLOWED_PROPOSAL_CLASSES)
    lines.append(f"== class counts: {footer} ==")
    print("\n".join(lines))
    return QUERY_EXIT_OK


def render_digest(registry: dict, rendered_on: str | None = None) -> str:
    """Compact digest: provenance blockquote, band table, per-band bullets."""
    meta = registry["meta"]
    gates = sorted(registry["gates"], key=lambda g: g["order"])
    total = len(gates)
    flagged_total = sum(1 for g in gates if g["findings"])
    date = rendered_on or datetime.datetime.now(tz=datetime.UTC).date().isoformat()

    out: list[str] = []
    out.append("# DIGEST.md — rendered from agents/ledger/gates.yaml · NEVER HAND-EDIT ·")
    out.append("")
    out.append(f"> {total} gates · rendered {date} @ HEAD {meta['head']} · load THIS into context;")
    out.append("> gates.yaml is the sole SSOT; the retired GATES.md/gates/*.md markdown world survives in agents/ledger/arbitration/2026-08-24/surface-and-analysis-preservation.md.")
    out.append("")
    out.append("| band | phase | gates | findings |")
    out.append("|---|---|---|---|")

    current_band: str | None = None
    band_gates: list[dict] = []
    band_rows: list[str] = []
    sections: list[str] = []

    def flush_band() -> None:
        stem = band_of(band_gates[0]["id"])
        phase = band_gates[0]["phase"]
        flagged = sum(1 for g in band_gates if g["findings"])
        band_rows.append(f"| {stem} | {phase} | {len(band_gates)} | {flagged} |")
        sections.append("")
        sections.append(f"### {stem} — {phase}")
        sections.append("")
        for gate in band_gates:
            flag = " ⚠FINDING" if gate["findings"] else ""
            pins = len(gate["validated_by"]) or "none direct"
            sections.append(f"- **{gate['id']}** `{gate['owner']}`{flag}")
            ins = ", ".join(gate["inputs"])
            outs = ", ".join(gate["outputs"])
            sections.append(f"  [{ins}] → [{outs}] · pins: {pins}")
        band_gates.clear()

    for gate in gates:
        band = band_of(gate["id"])
        if current_band is not None and band != current_band:
            flush_band()
        current_band = band
        band_gates.append(gate)
    flush_band()
    out.extend(band_rows)
    out.append(f"| **total** | | **{total}** | **{flagged_total}** |")
    out.extend(sections)
    return "\n".join(out) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="render_gates.py",
        description="Render cross-indexes and DIGEST.md from agents/ledger/gates.yaml "
                    "(the machine-readable gate registry SSOT), or run read-only "
                    "queries against it. Default action (no query flag) renders "
                    "everything.",
        epilog="Query modes are mutually exclusive and never write any file; they "
               "print human-readable plain text to stdout and exit 0 on >=1 hit / 3 "
               "on zero hits (the miss note goes to stderr); argparse errors exit 2.",
    )
    parser.add_argument("--gates", type=Path, default=DEFAULT_GATES,
                        help="path to gates.yaml (default: %(default)s)")
    parser.add_argument("--index-dir", type=Path, default=DEFAULT_INDEX_DIR,
                        help="directory for the by_* YAML sidecars (default: %(default)s)")
    parser.add_argument("--digest", type=Path, default=DEFAULT_DIGEST,
                        help="path of the DIGEST.md to (re)write (default: %(default)s)")

    query = parser.add_argument_group(
        "query modes (read-only; mutually exclusive; text on stdout; "
        "exit 0 on >=1 hit, exit 3 on zero hits)")
    mutex = query.add_mutually_exclusive_group()
    mutex.add_argument("--gate", metavar="ID",
                       help="print the single full YAML registry entry for this gate id")
    mutex.add_argument("--file", metavar="RELPATH",
                       help="gates owning this repo path (exact or suffix match) plus "
                            "gates whose inputs/outputs/if_changed reference those "
                            "gates' ids (by_owner_file.yaml logic)")
    mutex.add_argument("--artifact", metavar="ID",
                       help="gates producing/consuming this ARTIFACT-/CFG-/VOCAB- id "
                            "(by_artifact.yaml key logic)")
    mutex.add_argument("--library", metavar="NAME",
                       help="gates whose touched_by names NAME (part before '@'; "
                            "by_library.yaml logic)")
    mutex.add_argument("--test", metavar="NODEID",
                       help="gates whose validated_by contains this node-id substring")
    mutex.add_argument("--findings", action="store_true",
                       help="compact risk sheet: GATE · owner-short · N finding(s) · "
                            "first sentence of first finding, + summary footer")
    mutex.add_argument("--blast-radius", metavar="RELPATH", dest="blast_radius",
                       help="LOCATE mechanized (MAIN_AGENT_CONTRACT §14 step 1): gates "
                            "owning/referencing RELPATH closed one hop over outputs → "
                            "inputs/if_changed downstream refs; depth-marked, ordered "
                            "by phase/order, final line lists the pinning test node ids")
    mutex.add_argument("--csv", type=Path, metavar="OUTFILE",
                       help="flatten every gate to one CSV row (list fields joined "
                            "with ' | ', findings with ' ;; '); writes ONLY OUTFILE")
    mutex.add_argument("--symbol", metavar="TEXT",
                       help="case-sensitive substring over owner AND transforms "
                            "strings of all gates; ids+owners one per line")
    mutex.add_argument("--owner-exact", metavar="PATH:SYMBOL", dest="owner_exact",
                       help="gates whose owner path equals PATH exactly and whose "
                            "owner symbol token starts with SYMBOL (line numbers "
                            "are advisory, never matched)")
    mutex.add_argument("--has", metavar="THING",
                       help="unified agent probe; tried in order: gate-id exact → "
                            "artifact id exact → validated_by node-id exact → "
                            "touched_by library exact → file path (--file logic) → "
                            "owner-exact 'path:symbol'; first hit wins, ids printed "
                            "compactly with the winning probe kind")
    mutex.add_argument("--dupe-registry", action="store_true", dest="dupe_registry",
                       help="SSOT hygiene scan: OWNER-COLLISIONS (identical "
                            "path:symbol claimed by >=2 gates), ARTIFACT-TWO-WRITER "
                            "(output artifact with >=2 producers, by_artifact key "
                            "logic on outputs[]), NEAR-IDENTICAL (same owner path + "
                            "same outputs set, different ids); grouped report, exit "
                            "0 clean / 3 dupes-found; findings reported, never fixed")
    mutex.add_argument("--dedupe-proposals", type=Path, metavar="DIR",
                       dest="dedupe_proposals",
                       help="parse candidate blocks from *.md in DIR ('- id: "
                            "PROPOSE-' blocks until next '- id:' or blank-line-then-"
                            "'#'; '### PROPOSE-…' headings and '- **PROPOSE-…' bullets "
                            "best-effort too), classify UNIQUE / DUP-OF-GATE-<id> / "
                            "OVERLAP-PATH / INTERNAL-DUP vs registry + within-set; "
                            "report table sorted by file then id; exit 0 unless DIR "
                            "unreadable (2)")

    args = parser.parse_args(argv)

    registry = load_registry(args.gates)

    if args.gate is not None:
        return query_gate(registry, args.gate)
    if args.file is not None:
        return query_file(registry, args.file)
    if args.artifact is not None:
        return query_artifact(registry, args.artifact)
    if args.library is not None:
        return query_library(registry, args.library)
    if args.test is not None:
        return query_test(registry, args.test)
    if args.findings:
        return query_findings(registry)
    if args.blast_radius is not None:
        return query_blast_radius(registry, args.blast_radius)
    if args.csv is not None:
        try:
            return query_csv(registry, args.csv)
        except OSError as exc:
            print(f"error: cannot write {args.csv}: {exc}", file=sys.stderr)
            return 1
    if args.symbol is not None:
        return query_symbol(registry, args.symbol)
    if args.owner_exact is not None:
        return query_owner_exact(registry, args.owner_exact)
    if args.has is not None:
        return query_has(registry, args.has)
    if args.dupe_registry:
        return query_dupe_registry(registry, args.gates)
    if args.dedupe_proposals is not None:
        return query_dedupe_proposals(args.dedupe_proposals, registry)

    args.index_dir.mkdir(parents=True, exist_ok=True)
    banner = ("# Rendered by agents/tools/render_gates.py from agents/ledger/gates.yaml"
              " — do not hand-edit.\n")
    rendered = {
        args.index_dir / "by_owner_file.yaml":
            banner + dump_yaml(build_by_owner_file(registry), sort_keys=True),
        args.index_dir / "by_artifact.yaml":
            banner + dump_yaml(build_by_artifact(registry), sort_keys=True),
        args.index_dir / "by_library.yaml":
            banner + dump_yaml(build_by_library(registry), sort_keys=True),
    }
    rendered[args.digest] = render_digest(registry)
    for path, text in rendered.items():
        path.write_text(text, encoding="utf-8")

    print(f"gates: {len(registry['gates'])} from {args.gates}")
    for path in rendered:
        print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
