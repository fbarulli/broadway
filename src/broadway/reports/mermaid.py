from __future__ import annotations

import logging
import re
from itertools import pairwise

from broadway.reports.results import derive_status
from broadway.timeline.models import AnalysisDecision, AnalysisStep
from broadway.timeline.sequence import WalkthroughSequence, WalkthroughStepConfig

logger = logging.getLogger(__name__)

_UNSAFE_LABEL = re.compile(r'[<>\[\]{}|#"\'`:;,`]')
_UNSAFE_ID = re.compile(r"[^a-zA-Z0-9_]")
_WS = re.compile(r"\s+")
_REPEATED_UNDERSCORE = re.compile(r"_+")

GANTT_EPOCH = "2026-01-01"


def sanitize_label(text: str) -> str:
    """Strip characters that break Mermaid syntax; keep readable words."""
    cleaned = _UNSAFE_LABEL.sub("", text)
    cleaned = _WS.sub(" ", cleaned).strip()
    return cleaned or "step"


def sanitize_id(text: str) -> str:
    """Return a Mermaid-safe node id derived from free text."""
    cleaned = _UNSAFE_ID.sub("_", text)
    cleaned = _REPEATED_UNDERSCORE.sub("_", _WS.sub("_", cleaned)).strip("_")
    if not cleaned:
        return "step"
    if cleaned[0].isdigit():
        cleaned = f"s_{cleaned}"
    return cleaned


def _ordered_statuses(
    sequence: WalkthroughSequence,
    steps: list[AnalysisStep],
    decisions: list[AnalysisDecision],
) -> list[tuple[WalkthroughStepConfig, str]]:
    steps_by_id = {s.step_id: s for s in steps}
    pairs: list[tuple[WalkthroughStepConfig, str]] = []
    all_prior_resolved = True
    for cfg in sorted(sequence.steps, key=lambda s: s.order):
        status, all_prior_resolved = derive_status(
            cfg, steps_by_id, decisions, all_prior_resolved
        )
        pairs.append((cfg, status))
    logger.debug("resolved statuses for %d steps", len(pairs))
    return pairs


def _gantt_tag(status: str) -> str:
    if status in ("completed", "completed with note"):
        return "done"
    if status == "warning":
        return "crit, done"
    if status == "failed":
        return "crit"
    if status == "awaiting decision":
        return "active"
    return ""


def _decision_method(step_id: str, decisions: list[AnalysisDecision]) -> str | None:
    expected = step_id.removeprefix("decide_")
    for decision in decisions:
        if decision.id == expected and decision.kind == expected:
            return decision.method
    return None


def render_timeline_gantt(
    analysis: str,
    sequence: WalkthroughSequence,
    steps: list[AnalysisStep],
    decisions: list[AnalysisDecision],
) -> str:
    """Render the walkthrough timeline as a Mermaid gantt diagram."""
    pairs = _ordered_statuses(sequence, steps, decisions)
    lines = [
        "gantt",
        f"    title {sanitize_label(analysis)} walkthrough",
        "    dateFormat YYYY-MM-DD",
    ]
    prev: str | None = None
    for cfg, status in pairs:
        node = sanitize_id(cfg.id)
        task = sanitize_label(f"{cfg.label} {status}")
        tag = _gantt_tag(status)
        timing = f"{GANTT_EPOCH}, 1d" if prev is None else f"after {prev}, 1d"
        if tag:
            lines.append(f"    section {sanitize_label(cfg.label)}")
            lines.append(f"    {task} :{tag}, {node}, {timing}")
        else:
            lines.append(f"    section {sanitize_label(cfg.label)}")
            lines.append(f"    {task} :{node}, {timing}")
        prev = node
    logger.debug("rendered timeline gantt steps=%d", len(pairs))
    return "\n".join(lines)


def render_decision_flowchart(
    analysis: str,
    sequence: WalkthroughSequence,
    steps: list[AnalysisStep],
    decisions: list[AnalysisDecision],
) -> str:
    """Render lineage/decision flow as a Mermaid flowchart LR diagram."""
    pairs = _ordered_statuses(sequence, steps, decisions)
    node_ids = [sanitize_id(cfg.id) for cfg, _ in pairs]
    known = set(node_ids)
    lines = ["flowchart LR", f"    %% {sanitize_label(analysis)} lineage"]
    for (cfg, status), node in zip(pairs, node_ids, strict=True):
        text = f"{sanitize_label(cfg.label)} - {sanitize_label(status)}"
        if cfg.kind == "decision":
            method = _decision_method(cfg.id, decisions)
            if method is not None:
                text += f" - {sanitize_label(method)}"
            lines.append(f'    {node}{{"{text}"}}')
        else:
            lines.append(f'    {node}["{text}"]')
    for left, right in pairwise(node_ids):
        lines.append(f"    {left} --> {right}")
    lines.extend(_parent_edges(sequence, decisions, known))
    logger.debug("rendered decision flowchart nodes=%d", len(node_ids))
    return "\n".join(lines)


def _parent_edges(
    sequence: WalkthroughSequence,
    decisions: list[AnalysisDecision],
    known: set[str],
) -> list[str]:
    targets = {
        cfg.id.removeprefix("decide_"): sanitize_id(cfg.id)
        for cfg in sequence.steps
        if cfg.kind == "decision"
    }
    edges: list[str] = []
    for decision in decisions:
        target = targets.get(decision.id)
        if target is None:
            continue
        for parent in decision.parents:
            source = sanitize_id(parent)
            if source in known and source != target:
                edges.append(f"    {source} -.-> {target}")
    return edges
