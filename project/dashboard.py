"""Project composition root for the generic experiment dashboard."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import uvicorn

from project.paths import load_project_paths

if TYPE_CHECKING:
    from broadway.reports.story import StoryConfig


def _story_config() -> StoryConfig:
    """This project's story framing — the project layer owns the domain prose."""
    from broadway.reports.story import StoryConfig

    return StoryConfig(
        dataset="olist",
        question=(
            "Which orders will end with a 1- or 2-star review — predicted only from what "
            "is known at purchase time?"
        ),
        why=(
            "Bad reviews are the cost this project predicts: an order that ends with a 1- or "
            "2-star review is what the storefront wants to avoid, and the decision may use "
            "only what was known when the order was placed."
        ),
        target_label="bad review",
        target_rule="review score ≤ 2",
        decision_moment="purchase time",
        source_name="Olist",
        source_tables=9,
        post_purchase_note=(
            "Delivery and review timestamps (post-purchase) are measured as context, never "
            "admitted as features; the feature step keeps purchase-time information only."
        ),
        excluded_note=(
            "Review score, delivery and review timestamps, and every post-purchase column "
            "are excluded — they are only known after the outcome, so using them would be "
            "leakage."
        ),
        caveats=(
            "the target is an imbalanced binary",
            "the model sees purchase-time information only",
            "this is prediction, not causation",
        ),
    )


def main() -> None:
    """Serve the generic dashboard with this project's resolved paths."""
    paths = load_project_paths()
    os.environ.setdefault("BROADWAY_EXPERIMENTS_ROOT", str(paths.experiments))
    # Project-relative observations may not exist (live verdicts live under the
    # repo-root default). Only override when the resolved dir is real.
    if paths.observations.is_dir():
        os.environ.setdefault("BROADWAY_OBSERVATIONS_DIR", str(paths.observations))
    from broadway.reports.story import configure_story

    configure_story(_story_config())
    from broadway.reports.experiments_dashboard import app

    uvicorn.run(app, host="127.0.0.1", port=8000)


if __name__ == "__main__":
    main()
