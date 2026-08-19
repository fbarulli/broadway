"""Project layer for the synthetic demo (main branch).

The project layer binds a dataset's loaders + constants. On main it serves
the synthetic demo (``demo/demo.csv``); the taxi branch carries its own
dataset-specific bindings under the same package name. Keeping a minimal
``project`` package on main lets the shared CI build-and-boot gate exercise
the same worker image layout on both branches.
"""

from __future__ import annotations
