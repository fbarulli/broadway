"""Run correlation context (LOGGING.md §5).

One ``run_id`` per invocation is shared by the log lines, the profile record,
the evidence pack, and the MLflow run. A ``ContextVar`` carries it without
threading a parameter through every layer; the placeholder keeps the log
formatter valid before an entry point mints an id.
"""

from __future__ import annotations

import uuid
from contextvars import ContextVar

RUN_ID_ENV = "BROADWAY_RUN_ID"
DEFAULT_RUN_ID = "-"

_run_id: ContextVar[str] = ContextVar("broadway_run_id", default=DEFAULT_RUN_ID)


def new_run_id() -> str:
    """Mint a fresh run id (uuid4 hex)."""
    return uuid.uuid4().hex


def current_run_id() -> str:
    """Return the run id of the current context, ``"-"`` when unset."""
    return _run_id.get()


def set_run_id(value: str) -> None:
    """Bind the run id for the current invocation/context."""
    _run_id.set(value)
