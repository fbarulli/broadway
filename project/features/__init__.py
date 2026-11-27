"""Dataset-specific column plans for this project.

Each plan is a module beside this package (``project.features.amazon``) and is
referenced by an experiment config as ``feature_plan: project.features.<name>:plan``.
Keeping this file import-light lets the loader import one plan without binding
the others (and without any retired dataset plan).
"""
