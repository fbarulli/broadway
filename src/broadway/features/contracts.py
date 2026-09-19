"""Data contracts: runtime checks that a DataFrame matches the schema
expected at a given pipeline stage.

The raw-schema check lives in ``project/etl/process.py`` (contract-driven via
``build_raw_schema``); this module keeps the shared error type used across
contract boundaries.
"""


class DataContractError(Exception):
    """Raised when a DataFrame doesn't match its stage's expected schema."""
