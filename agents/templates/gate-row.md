# TEMPLATE — gates.yaml registry row
#
# Add a row when a commit touches a GOVERNED SURFACE (anything another gate,
# test, or script consumes). Copy this shape into agents/ledger/gates.yaml,
# fill every field, re-render the digest, then commit.
#
#   uv run python agents/tools/render_gates.py --head $(git rev-parse HEAD)
#   # (see render_gates.py --help for exact flags)
#
# REQUIREMENTS (the registry-mapping law; audit with:
#   uv run python agents/tools/render_gates.py --blast-radius <path>):
#   R1. id        GATE-<BAND>-<NNN> unique; band from the row's phase band;
#                 mint new ids only at band-range edges (BAND_RANGES law)
#   R2. phase      the band the row belongs to (01-ingest … infra-meta)
#   R3. order      integer; dense order within phase, no gaps
#   R4. owner      the artifact + line anchors that own the law, current
#                 (re-anchor on every edit: line numbers drift)
#   R5. inputs    what the gate reads (files, refs, constants)
#   R6. outputs   every distinct verdict line the gate can emit
#   R7. transforms the law itself, one line per rule
#   R8. touched_by  other files that consume/rely on this gate
#   R9. validated_by  test node id(s) that exercise the gate (empty list =
#                 ACKNOWLEDGED coverage gap — the row's findings must say so)
#   R10. if_changed  refs/paths whose change invalidates this row
#   R11. findings  honest gaps: what is NOT tested/pinned, in plain words
#
# WORKED EXAMPLE (real row GATE-INFRA-93, trimmed):
#   - id: GATE-INFRA-93
#     phase: infra-meta
#     order: 140
#     owner: scripts/check_branch_parity.sh:71 check() (SHARED lockstep, …)
#     inputs:
#     - <the SHARED surface list>
#     - origin/main vs origin/<track> tips
#     outputs:
#     - 'PARITY OK | DRIFT: <path> differs …'
#     transforms:
#     - byte-identical git diff per SHARED path, deletions included
#     touched_by:
#     - scripts/run_local_ci.sh:30 (consumes pinned copy of THIS file)
#     validated_by:
#     - tests/test_branch_pity_scripts.py::test_parity_surface_includes_scripts
#     if_changed:
#     - any SHARED[] path
#     findings:
#     - <the honest gap>
#
# SCAFFOLD — fill every field:

- id: GATE-<BAND>-<NNN>
  phase: <band>
  order: <int>
  owner: <artifact>:<line> <function> — <what it owns>
  inputs:
  - <what it reads>
  outputs:
  - '<verdict line>'
  transforms:
  - <one rule per line>
  touched_by:
  - <consuming files>
  validated_by:
  - tests/<file>::<node>
  if_changed:
  - <paths/refs whose change invalidates this row>
  findings:
  - <honest gap in plain words>
