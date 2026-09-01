"""Euromonitor volume-extractor unit tests (v2 validated).

Pins the extractor as a fixed point for step 03: broader unit coverage
(dl/gal/qt/pt/lt), EU decimal-comma, punctuation tolerance, symmetric
word boundaries, and the bare-oz ambiguity flag. These 26 cases are the
validation record behind the 98.9% within-GTIN agreement metric.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
EXTRACTOR = REPO / "project" / "experiments" / "euromonitor" / "02_volume_normalize.py"


def _load_extractor():
    import sys

    series_dir = EXTRACTOR.parent
    if str(series_dir) not in sys.path:
        sys.path.insert(0, str(series_dir))
    src = EXTRACTOR.read_text(encoding="utf-8")
    ns: dict = {}
    exec(src.split("def main()")[0], ns)  # module-level defs only, no side effects
    return ns


@pytest.fixture(scope="module")
def extract():
    return _load_extractor()["extract_volume_ml"]


# (input, expected_ml, expected_ambiguous)
CASES: list[tuple[str, float | None, bool]] = [
    # base + US fluid ounces
    ("500ml", 500, False),
    ("0.5L", 500, False),
    ("16.9 Fl Oz", 500, False),
    ("16.9 Fl. Oz.", 500, False),
    ("16.9FlOz", 500, False),
    ("8 oz Soda Can", 235, True),
    # metric fractions + EU decimal comma
    ("50 cl", 500, False),
    ("1 Liter", 1000, False),
    ("1,5 L", 1500, False),
    ("33cl", 330, False),
    ("1,5litre", 1500, False),
    # broader units
    ("1 gallon", 3785, False),
    ("1 qt", 945, False),
    ("2 lt", 2000, False),
    ("2 ltr", 2000, False),
    ("2 pt", 945, False),
    ("2 pints", 945, False),
    ("3 dl", 300, False),
    ("5gal", 18925, False),
    # pack-count stripping
    ("6-pack of 355ml", 355, False),
    ("12 Pack, 12 fl oz", 355, False),
    ("case of 24, 500ml", 500, False),
    # ambiguity + boundary rejection
    ("8 oz bag of chips", 235, True),
    ("Protein Powder 2 lb", None, False),
    ("2 large bottles", None, False),
    # ordering: long unit wins over parenthetical
    ("RC Cola 2 lt (2.1 qt)", 2000, False),
]


@pytest.mark.parametrize("text,exp_ml,exp_ambig", CASES, ids=[c[0] for c in CASES])
def test_extract_volume_ml(extract, text, exp_ml, exp_ambig) -> None:
    ml, ambig = extract(text)
    assert ml == exp_ml, f"{text!r}: ml={ml} expected {exp_ml}"
    assert ambig == exp_ambig, f"{text!r}: ambig={ambig} expected {exp_ambig}"


def test_extract_rejects_word_embedded_units(extract) -> None:
    """Symmetric (?<![a-zA-Z])...(?![a-zA-Z]) boundaries: no match inside words."""
    for text in ["2 lb", "2 large bottles", "X500mlY"]:
        ml, _ = extract(text)
        assert ml is None, f"{text!r}: expected None, got {ml}"


def test_extract_non_string_returns_none(extract) -> None:
    assert extract(None) == (None, False)
