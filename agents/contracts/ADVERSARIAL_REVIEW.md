# ADVERSARIAL_REVIEW.md — every non-trivial change gets attacked before it lands

Owner directive: "from here on out always consult with another agent for adversarial opinion."

This is a standing process gate, not advice.

## When it applies

Any of:
- a new mechanism, gate, owner class, or contract;
- a removal or capability reduction (deleting a surface, dropping a model/adapter);
- a statistical claim or decision rule (metric, interval, null, threshold);
- security-relevant code (auth, secrets, filesystem paths, remote execution);
- a change that touches more than one layer.

Trivial edits (typo, one-line doc, formatting) are exempt.

## How it runs

1. The author writes the proposal **as an artifact**: what changes, where (`file:line`),
   what it replaces, and the evidence it relies on.
2. A separate agent reviews it **read-only**. It must not be the author and must not
   have authored the change. It gets the proposal, not the author's reasoning.
3. The reviewer must produce: concrete attacks grounded in the code, the failure modes
   the proposal misses, a cheaper/stronger alternative, and a verdict
   **PROCEED / REVISE (specific revision) / REJECT**.
4. The author records the verdict in the commit message (`Adversarial: PROCEED` or
   `Adversarial: REVISE — <one line>`) and lands only on PROCEED.

## What "adversarial" means

A review that agrees is a failed review. The reviewer's job is to find the case where
the proposal is wrong: the sample where it misfires, the consumer left behind, the
cost it hides, the false confidence it creates, the simpler design that would do.

## Why

Three audits found the same class repeatedly: contracts documented, enforcement
advisory, changes applied at one call site and left half-done elsewhere. The author
of a change is the worst person to find its boundary; a second agent, pointed at the
artifact, is the cheapest detector we have.
