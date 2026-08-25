# HUMAN RESOLUTIONS — 2026-08-24 (recorded verbatim-in-substance)

Seven rulings by the human owner on the human-call queue opened by the five
senior packets (A–E). These bind all downstream lanes.

## 1 · HC-1 (SECRET-1 / stringData.DB_PASSWORD) — VOIDED

The `stringData.DB_PASSWORD` matter is disregarded entirely. No rotation, no
evidence row, no closure condition. SEC-S1 and SEC-S2's credential-urgency
framing dissolves with it; D-packet slate items touching `.env` parity,
README bootstrap, and golden approx stand unaffected (they do not reference
the k8s manifest). D's "prime exposure" judgment is struck by this ruling.

## 2 · HC-2 (propagation ordering + ledger truthing) — APPROVED

D24 single-reconciliation-pass ratified as the SOLE propagation vehicle.
Three ordered conditions:

1. **Correct the ledger first** — the `configs/secret.yaml` reference at
   FIXES.md:202 is factually false (string never existed in any referenced
   artifact; introduced during discussion). Treated as a ledger defect, not
   historical evidence; fix before any reconciliation backfill is applied.
2. **Then perform reconciliation** — execute the backfill through the D24
   reconciliation mechanism only. Multiple propagation mechanisms create
   future truth-source ambiguity.
3. **Supersede affected rows** — obsolete board rows marked superseded,
   history never silently edited; auditability preserved so future readers
   land on the corrected lineage.

Rationale of record: this was a record-truth dispute, not a policy or intent
dispute — quoted path provably nonexistent + derived board error + existing
designated mechanism ⇒ truth correction → single reconciliation pass →
supersession trail.

## 3 · NA-token alignment (packet A #9) — ASYMMETRY RATIFIED AS INTENT

Choose asymmetry absent a concrete cross-plane invariant requiring alignment.
No such invariant surfaced (divergence observable in JoinAudit; lookup-side
authored-token design deliberate). Do not normalize vocabularies for aesthetic
consistency when planes carry different contractual semantics.

## 4 · Evidence JSONs (packet C #5) — DEMOTED TO DEBUG PROVENANCE

Nothing consumes them as an enforcement contract; do not build a reader just
to justify their existence. Preserve useful diagnostics; enforcement stays at
the consumer boundary.

## 5 · decisions_dir (packet C #7) — DELETE UNLESS A PRODUCER IS NAMED

"Maybe something will write here later" is insufficient; an unproduced
subsystem creates false confidence. No producer named → dedicated deletion
item in the implementation stage.

## 6 · project/scripts (packet C #10) — EXPLICIT CLASSIFICATION REQUIRED

Governance decision, not an implementation question: if they are intentionally
record-free teaching surfaces, say so in the contract; otherwise put them
under the relevant gate.

## 7 · D11 dead EnvironmentConfig fields (packet C #25) — STRAIGHT DELETE AUTHORIZED

Evidence unusually clean; dead fields need no speculative migration
architecture (~−25 lines / 4 files, arithmetic already prepared by packet C).
Exactly the class of small deletion the bench should be allowed to make.

## Genesis provenance note

The amendment-vs-repin decision on the six void genesis event-ids awaited the
GENESIS-AUDIT evidence per human instruction ("shape passes" inadmissible as
validity). Audit concluded: **all six rows VALID-SUBSTANCE** — every claimed
authorization/amendment/ratification/verdict exists and matches in-tree today;
only the hash layer predates store-then-hash. Per the decision rule this
resolves to RECIPE AMENDMENT, landed as the Legacy-id grandfather clause in
MAIN_AGENT_CONTRACT §14 (ruled 2026-08-24): genesis ids resolve on
author+status+type checks plus byte-exact identity with their frozen STATE.md
registry row; any body edit still voids them; every event after
2026-08-24T16:19:07Z follows full recomputation, no exceptions.
