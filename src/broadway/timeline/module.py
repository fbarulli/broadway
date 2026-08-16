from __future__ import annotations

import os
from pathlib import Path

from broadway.timeline.models import AnalysisDecision, AnalysisStep

TIMELINE_DIR = Path(os.getenv("BROADWAY_TIMELINE_DIR", "artifacts/timeline"))


def steps_dir(analysis: str) -> Path:
    return TIMELINE_DIR / analysis / "steps"


def decisions_dir(analysis: str) -> Path:
    return TIMELINE_DIR / analysis / "decisions"


def save_step(step: AnalysisStep) -> None:
    d = steps_dir(step.analysis)
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{step.step_id}.json").write_text(step.model_dump_json(indent=2), encoding="utf-8")


def load_step(analysis: str, step_id: str) -> AnalysisStep | None:
    path = steps_dir(analysis) / f"{step_id}.json"
    if not path.exists():
        return None
    return AnalysisStep.model_validate_json(path.read_text(encoding="utf-8"))


def load_steps(analysis: str) -> list[AnalysisStep]:
    d = steps_dir(analysis)
    if not d.is_dir():
        return []
    steps = [
        AnalysisStep.model_validate_json(p.read_text(encoding="utf-8"))
        for p in d.glob("*.json")
    ]
    return sorted(steps, key=lambda s: s.order)


def save_decision(decision: AnalysisDecision) -> None:
    d = decisions_dir(decision.analysis)
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{decision.id}.json").write_text(decision.model_dump_json(indent=2), encoding="utf-8")


def load_decision(analysis: str, decision_id: str) -> AnalysisDecision | None:
    path = decisions_dir(analysis) / f"{decision_id}.json"
    if not path.exists():
        return None
    return AnalysisDecision.model_validate_json(path.read_text(encoding="utf-8"))


def load_decisions(analysis: str) -> list[AnalysisDecision]:
    d = decisions_dir(analysis)
    if not d.is_dir():
        return []
    return [
        AnalysisDecision.model_validate_json(p.read_text(encoding="utf-8"))
        for p in sorted(d.glob("*.json"))
    ]
