# diagrams/ — canvas maps (DATA, not code)

Each map is a pair: `<name>.json` (spec) + `<name>.tldr` (rendered diagram,
opens in VS Code with the tldraw extension or the hosted dashboard
`/canvas/<name>`). Annotations save back into the `.tldr` source.

## The code (lives in `src/`, not here)

All maps are produced by builders — never hand-written JSON:

- `src/broadway/reports/canvas.py` — `new_spec` / `add_node` / `add_edge`
  primitives, `experiment_series_spec` (two-track: steps → results),
  `blast_radius_spec` (fan-out from a symbol + callers),
  `write_spec` + `emit_tldr` (renders via the tldraw skill generator).
- `scripts/blast_radius.sh <path> [symbol]` — the evidence source: gate
  ownership plus `graphify affected` callers that feed the blast maps.

## Regenerate a map

```bash
# 1. build the spec from live project surfaces (python, src builders)
# 2. render it (needs node + the tldraw skill installed)
python - <<'EOF'
from pathlib import Path
from broadway.reports import canvas
spec = canvas.experiment_series_spec({"univariate": ["01_a"]})
canvas.write_spec(spec, Path("diagrams/my_map.json"))
canvas.emit_tldr(Path("diagrams/my_map.json"), Path("diagrams/my_map.tldr"))
EOF
# 3. view at /canvas/my_map on the dashboard; annotate; save persists
```

## Current maps

- `blast_radius.*` — DatasetContract fan-out (consumers + gates).
- `results_producers.*` — every result-producing surface → dashboard.
