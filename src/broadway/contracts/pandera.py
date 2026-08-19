"""Generate a Pandera schema from a :class:`DatasetContract`.

The raw schema is derived at runtime from ``contract.columns`` — one
``pa.Column`` per contract entry, so there is no hand-maintained
``DataFrameModel`` that can drift out of sync with the contract.
"""

from __future__ import annotations

import pandera as pa

from broadway.config.schema import DatasetContract

_INT_DTYPES = {
    "int8": pa.Int8,
    "int16": pa.Int16,
    "int32": pa.Int32,
    "int64": pa.Int64,
}

_FLOAT_DTYPES = {
    "float32": pa.Float32,
    "float64": pa.Float64,
}

_DATETIME_DTYPES = {"datetime64[us]", "datetime64[ns]", "datetime64"}
_STRING_DTYPES = {"object", "str", "string"}


def is_numeric_dtype(dtype: str) -> bool:
    return dtype in _INT_DTYPES or dtype in _FLOAT_DTYPES


def pandera_dtype(dtype: str) -> type[pa.DataType]:
    if dtype in _INT_DTYPES:
        return _INT_DTYPES[dtype]
    if dtype in _FLOAT_DTYPES:
        return _FLOAT_DTYPES[dtype]
    if dtype in _DATETIME_DTYPES:
        return pa.DateTime
    if dtype in _STRING_DTYPES:
        return pa.String
    return pa.Object


def build_raw_schema(contract: DatasetContract) -> pa.DataFrameSchema:
    """Build a structure-only schema for the raw columns in ``contract``.

    Columns come directly from ``contract.columns``; join-derived columns
    (e.g. ``enriched_group``) are not part of the raw contract. Dtypes are
    checked strictly (``coerce=False``) and nulls are permitted
    (``nullable=True``) — the contract's ``null_count`` is observed, not an
    invariant.
    """
    return pa.DataFrameSchema(
        {
            name: pa.Column(pandera_dtype(col.dtype), nullable=True)
            for name, col in contract.columns.items()
        }
    )
