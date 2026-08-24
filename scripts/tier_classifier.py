#!/usr/bin/env python3
"""Risk-tier classifier for the P1 review lane (TIERREV-1, ruling b983b5a5).

Every commit carries a ``Tier:`` trailer computed by the MAIN AGENT ONLY at
staging time (MAIN_AGENT_CONTRACT.md §6) — never worker-declared. FULL
commits get the full adversarial-review ceremony; CHECKLIST commits may ride
the mechanical probe battery (``tests/test_governance_probes.py``). Unknown
or mixed signals default UP to FULL.

Usage — stdin filter over a git-diff payload:

    git show <sha> | python3 scripts/tier_classifier.py      # JSON to stdout

or importable as a pure function:

    from tier_classifier import classify
    classify(["README.md"], ["some prose line"])  # {"tier": ..., "reasons": [...]}

Stdlib only, no I/O in the pure core.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import PurePosixPath

FULL = "FULL"
CHECKLIST = "CHECKLIST"

# Executed prose: editing these files IS a governance act, never checklist
# material (rule 1). Names are reserved repo-wide, not just at repo root.
NEVER_PROSE_BASENAMES = frozenset(
    {"CONTRACT_TEMPLATE.md", "WORKER_CONTRACT.md", "MAIN_AGENT_CONTRACT.md", "DECISIONS.md", "FIXES.md"}
)
NEVER_PROSE_PATHS = frozenset({"agents/ledger/STATE.md"})

# Behavior surfaces (rule 2).
BEHAVIOR_PREFIXES = ("src/", "tests/", "project/", "experiments/", "scripts/", ".github/", "k8s/")
BEHAVIOR_EXACT = frozenset({"pyproject.toml", "uv.lock"})
BEHAVIOR_SUFFIXES = (".sh",)
DOCKER_PREFIXES = ("docker", "Dockerfile")

# Descriptive-prose extensions eligible for CHECKLIST; everything outside
# {governance ∪ behavior ∪ prose-doc} is unclassifiable and defaults UP.
PROSE_SUFFIXES = frozenset({".md", ".txt", ".rst"})

SAMPLE_CAP = 3

_TRIGGERS: dict[str, re.Pattern[str]] = {
    # 8-hex sha/agent-id-like tokens (deliberately simple per spec; a hex
    # English word such as deadbeef also matches — false positives point UP).
    "sha-like-token": re.compile(r"\b[0-9a-f]{8}\b"),
    # A number adjacent (≤24 non-digit, non-newline chars) to gate vocabulary.
    "gate-number": re.compile(
        r"(?:cov\w*|floor\w*|fail[-_ ]?under|≥|>=|percent)[^\d\n]{0,24}\d"
        r"|\d[^\d\n]{0,24}(?:cov\w*|floor\w*|fail[-_ ]?under|≥|>=|percent)",
        re.IGNORECASE,
    ),
    "backticked-path": re.compile(r"`[^`\n]+\.(?:py|sh|md|ya?ml|toml)`"),
    "shell-ish-line": re.compile(r"^(?:uv run |bash |git |timeout )"),
    "fenced-code": re.compile(r"```"),
}


def _is_never_prose(path: str) -> bool:
    """Rule 1: executed-prose governance file (names reserved repo-wide)."""
    norm = path.lstrip("./")
    return norm in NEVER_PROSE_PATHS or PurePosixPath(norm).name in NEVER_PROSE_BASENAMES


def _is_behavior(path: str) -> bool:
    """Rule 2: behavior surface."""
    norm = path.lstrip("./")
    return (
        norm.startswith(BEHAVIOR_PREFIXES)
        or norm.startswith(DOCKER_PREFIXES)
        or norm in BEHAVIOR_EXACT
        or norm.endswith(BEHAVIOR_SUFFIXES)
        or (norm.startswith("configs/") and norm.endswith(".yaml"))
    )


def _is_prose_doc(path: str) -> bool:
    """Classifiable descriptive-prose document (CHECKLIST-eligible)."""
    norm = path.lstrip("./")
    return not _is_never_prose(norm) and PurePosixPath(norm).suffix.lower() in PROSE_SUFFIXES


def _content_triggers(added_lines: list[str]) -> dict[str, list[str]]:
    """Rule 3: map trigger name → up to SAMPLE_CAP distinct sample snippets."""
    hits: dict[str, list[str]] = {}
    for line in added_lines:
        for name, pattern in _TRIGGERS.items():
            m = pattern.search(line)
            if m:
                samples = hits.setdefault(name, [])
                if len(samples) < SAMPLE_CAP and m.group(0) not in samples:
                    samples.append(m.group(0)[:60])
    return hits


def classify(files: list[str], added_lines: list[str]) -> dict[str, object]:
    """Classify one change set. Returns {"tier": FULL|CHECKLIST, "reasons": [...]}.

    Tier is CHECKLIST only when every file is a classifiable prose document,
    the set is non-empty, and zero content triggers fired; any governance
    file, behavior surface, content trigger, unclassifiable path, mixed
    signal, or empty payload defaults UP to FULL.
    """
    gov = sorted({p for p in files if _is_never_prose(p)})
    beh = sorted({p for p in files if _is_behavior(p)})
    trig = _content_triggers(added_lines)
    unk = sorted({p for p in files if not _is_never_prose(p) and not _is_behavior(p) and not _is_prose_doc(p)})
    reasons: list[str] = []
    if gov:
        reasons += ["executed-prose governance file — probes + scoped adversary mandatory",
                    f"governance files: {', '.join(gov)}"]
    if beh:
        reasons += ["behavior surface", f"behavior paths: {', '.join(beh)}"]
    if trig:
        reasons += [f"content trigger(s): {', '.join(sorted(trig))}",
                    "; ".join(f"{k}={v}" for k, v in sorted(trig.items()))]
    if unk:
        reasons += ["unclassified path(s) — default UP to FULL", f"unclassified paths: {', '.join(unk)}"]
    tier = FULL
    if gov or beh or trig or unk:
        pass  # mixed signals stay visible in reasons; tier already FULL
    elif files and all(_is_prose_doc(p) for p in files):
        reasons.append("descriptive prose, zero triggers")
        tier = CHECKLIST
    else:
        reasons.append("empty payload or unclassifiable input — default UP to FULL")
    return {"tier": tier, "reasons": reasons}


def parse_diff_payload(payload: str) -> tuple[list[str], list[str]]:
    """Extract ordered unique file paths and all added lines from a diff."""
    files: list[str] = []
    added: list[str] = []
    current: str | None = None
    pending: str | None = None
    for line in payload.splitlines():
        if line.startswith("diff --git "):
            if current:
                files.append(current)
            m = re.search(r" b/(.*)$", line)
            current = pending = m.group(1).strip('"') if m else None
        elif line.startswith("rename to "):
            current = pending = line[len("rename to "):].strip('"')
        elif line.startswith("+++ "):
            side = line[4:].strip('"')
            current = side[len("b/"):] if side != "/dev/null" else pending
        elif line.startswith("+") and not line.startswith("+++"):
            added.append(line[1:])
    if current:
        files.append(current)
    return list(dict.fromkeys(files)), added


def main(argv: list[str] | None = None) -> int:
    """CLI entry: read diff payload on stdin, print classification JSON."""
    argv = sys.argv[1:] if argv is None else argv
    if argv:
        print(f"usage: {sys.argv[0]} < git-diff-payload  (no arguments expected)", file=sys.stderr)
        return 2
    files, added = parse_diff_payload(sys.stdin.read())
    print(json.dumps(classify(files, added), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
