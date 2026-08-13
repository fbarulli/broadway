"""
Data contracts: runtime checks that a DataFrame matches the schema
expected at a given pipeline stage. The generic platform validates
columns through the contract layer (``broadway.contracts.checks``)
driven by ``DatasetContract`` rather than hard-coded raw schemas.
"""


class DataContractError(Exception):
    """Raised when a DataFrame doesn't match its stage's expected schema."""
