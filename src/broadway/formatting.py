from __future__ import annotations


def humanize_float(value: float) -> str:
    return f"{float(value):.3g}"


def humanize_pvalue(value: float) -> str:
    value = float(value)
    if value < 0.001:
        return "< 0.001"
    return f"{value:.3f}"
