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

## Regenerate maps + serve (one command)

```bash
bash scripts/uv.sh run --extra dev python scripts/render_canvas_maps.py [--no-serve] [--port 8000]
```

Rebuilds every map from live project surfaces, prints the URLs, and starts
the dashboard as its final act — a running web app is the expected output.
`--no-serve` rebuilds only. Then open `/canvas/<name>`: annotate, save
persists back into the versioned `.tldr`.

## Current maps

- `blast_radius.*` — DatasetContract fan-out (consumers + gates).
- `results_producers.*` — every result-producing surface → dashboard.
