"""The story page: project framing, HTML rendering, and the HTTP routes.

The narrative and its evidence live in :mod:`broadway.reports.story`; this module
turns the sections into HTML, injects the project's :class:`StoryConfig`, and
serves the two routes. The dashboard includes :data:`router` and registers its
shared conclusions-first page frame through :func:`configure_story`. Every
dynamic value is escaped here.
"""

from __future__ import annotations

import html
import os
from collections.abc import Callable
from pathlib import Path
from urllib.parse import quote

from fastapi import APIRouter
from fastapi.responses import HTMLResponse, PlainTextResponse, Response

from broadway.reports.story import STORY_TITLE, Section, Story, StoryConfig

#: The configured framing; replaced by the project launcher through ``configure_story``.
_CONFIG: StoryConfig = StoryConfig.fallback()
#: The experiments root; the dashboard supplies a callable so late env changes are seen.
_ROOT_PROVIDER: Callable[[], Path] = lambda: Path(
    os.environ.get("BROADWAY_EXPERIMENTS_ROOT", "experiments")
)
#: The page frame; the dashboard supplies the shared conclusions-first frame.
_PAGE_RENDERER: Callable[[Story], str] = lambda builder: _default_page(builder)


def configure_story(
    config: StoryConfig | None = None,
    root_provider: Callable[[], Path] | None = None,
    page_renderer: Callable[[Story], str] | None = None,
) -> None:
    """Inject the project framing, the experiments root and the page frame."""
    global _CONFIG, _ROOT_PROVIDER, _PAGE_RENDERER
    if config is not None:
        _CONFIG = config
    if root_provider is not None:
        _ROOT_PROVIDER = root_provider
    if page_renderer is not None:
        _PAGE_RENDERER = page_renderer


def story_config() -> StoryConfig:
    """The configured story framing (the project launcher sets it)."""
    return _CONFIG


def render_section(section: Section) -> str:
    """One section: heading, conclusions, figure, numbers, then record links."""
    parts = ['<section class="story-section">', f"<h2>{html.escape(section.title)}</h2>"]
    css = "not-run" if section.not_run else "takeaway"
    parts.extend(
        f'<p class="{css}">{html.escape(conclusion)}</p>'
        for conclusion in section.conclusions
    )
    if not section.not_run:
        if section.figure:
            alt = html.escape(f"{section.title} figure")
            parts.append(
                f'<figure><img src="/story/figure/{html.escape(section.figure)}"'
                f' alt="{alt}"></figure>'
            )
        if section.numbers:
            items = "".join(
                f"<li>{html.escape(number)}</li>" for number in section.numbers
            )
            parts.append(f'<ul class="numbers">{items}</ul>')
    if section.records:
        links = " · ".join(
            f'<a href="/experiments/{quote(step)}?focus=pipeline">{html.escape(label)}</a>'
            for label, step in section.records
        )
        parts.append(f'<p class="record-link">decision record: {links}</p>')
    parts.append("</section>")
    return "\n".join(parts)


def render_body(builder: Story) -> str:
    """The story body: navigation, a dataset note, then every section."""
    parts = [
        (
            '<nav class="story-nav"><a href="/">← experiments</a> · '
            '<a href="/?focus=pipeline">pipeline evidence</a></nav>'
        )
    ]
    if not builder.has_records:
        scope = html.escape(builder.dataset or "project")
        parts.append(
            f'<p class="not-run">No {scope} decision records are published yet — every '
            "numeric stage below says “not run yet” until the pipeline lands.</p>"
        )
    parts.extend(render_section(section) for section in builder.sections())
    return "\n".join(parts)


def _default_page(builder: Story) -> str:
    """A neutral standalone page for a checkout that mounts the router alone."""
    return (
        '<!doctype html><html lang="en"><head><meta charset="utf-8">'
        f"<title>{html.escape(STORY_TITLE)}</title></head><body>"
        f"<h1>{html.escape(STORY_TITLE)}</h1>"
        f'<p class="question">{html.escape(builder.question)}</p>'
        f"{render_body(builder)}</body></html>"
    )


router = APIRouter()


@router.get("/story", response_model=None)
def story_page() -> HTMLResponse:
    """Serve the project story: conclusions first, graphs, numbers, record links."""
    builder = Story.from_experiments_root(_ROOT_PROVIDER(), _CONFIG)
    return HTMLResponse(_PAGE_RENDERER(builder))


@router.get("/story/figure/{name}", response_model=None)
def story_figure(name: str) -> Response | PlainTextResponse:
    """Serve one story graph, rendered in memory from the current records."""
    builder = Story.from_experiments_root(_ROOT_PROVIDER(), _CONFIG)
    png = builder.figure_png(name)
    if png is None:
        return PlainTextResponse("unknown story figure", status_code=404)
    return Response(content=png, media_type="image/png")
