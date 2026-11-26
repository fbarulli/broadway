"""Shared lane boundary: resolve the ``module:function`` adapter reference.

Both lane entries take their model code as a reference — ``memory_probe.py``'s
``--step`` and ``hpo_kernel.py``'s ``--objective`` — so no vendor or model code
lives in the lane. The lane is path-loaded (it is not an installed package): run
the entries as ``python gpu_training/<entry>.py`` from the repo root, or
register this module in ``sys.modules`` before loading an entry by path.
"""

from __future__ import annotations

import importlib
from collections.abc import Callable
from typing import Any


def resolve_callable(reference: str, kind: str) -> Callable[..., Any]:
    """Import ``module:function`` and return the callable; every failure is loud."""
    module_name, separator, function_name = reference.partition(":")
    if not separator or not module_name or not function_name:
        raise ValueError(f"{kind} must be 'module:function', got {reference!r}")
    module = importlib.import_module(module_name)
    function = getattr(module, function_name, None)
    if function is None:
        raise ValueError(f"{kind} {reference!r}: no attribute {function_name!r} in {module_name!r}")
    if not callable(function):
        raise TypeError(f"{kind} {reference!r} is not callable")
    return function
