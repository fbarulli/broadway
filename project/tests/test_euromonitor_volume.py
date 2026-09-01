"""Euromonitor volume-extractor unit tests (v2 validated).

Pins the extractor as a fixed point for step 03: broader unit coverage
(dl/gal/qt/pt/lt), EU decimal-comma, punctuation tolerance, symmetric
word boundaries, and the bare-oz ambiguity flag. These 26 cases are the
validation record behind the 98.9% within-barcode agreement metric.
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
    # EU decimal with a space after the comma (extractor v3 fix)
    ("Highway Cola 0, 33l", 330, False),
    ("tax tropical fruits 1, 5l", 1500, False),
    # ...but count-lists with single-digit counts stay lists, not decimals
    ("case of 24, 500ml", 500, False),
    ("sterilgarda juice 100% tropical lt 1, 100cl", 1000, False),
    ("freshener little giant organic apple juice concentrate 1 + 4, 200ml", 200, False),
    ("levissima natural water levissima Issima 33 x 4, 132 cl", 1320, False),
    # a DOT decimal never takes a space: "pH 9.0 bottle 600 ml" stays 600
    ("ACTIPH alkaline water ionized pH 9.0 bottle 600 ml", 600, False),
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


# ---------------------------------------------------------------------------
# Measurement validation (02d): raw (value, unit) extraction + sanity rules.
# ---------------------------------------------------------------------------

def _load_text_module():
    import sys

    series_dir = EXTRACTOR.parent
    if str(series_dir) not in sys.path:
        sys.path.insert(0, str(series_dir))
    import _text
    return _text


@pytest.fixture(scope="module")
def text_module():
    return _load_text_module()


@pytest.mark.parametrize("text,exp_value,exp_unit,exp_ambig", [
    ("Coconut Water 350 L", 350.0, "l", False),
    ("Cola 500 ml", 500.0, "ml", False),
    ("1,5 L Juice", 1.5, "l", False),
    ("Tea 16.9 Fl. Oz.", 16.9, "fl oz", False),
    ("2 lt (2.1 qt) Cola", 2.0, "lt", False),
    ("Powder Mix 2 kg", None, None, False),  # weight units never extract
    ("8 oz chips", 8.0, "oz", True),
    (None, None, None, False),
], ids=lambda c: str(c)[:40])
def test_extract_volume_measurement(text_module, text, exp_value, exp_unit,
                                    exp_ambig) -> None:
    value, unit, ambig = text_module.extract_volume_measurement(text)
    assert (value, unit, ambig) == (exp_value, exp_unit, exp_ambig), \
        f"{text!r}: got ({value}, {unit}, {ambig})"


@pytest.mark.parametrize("value,unit,category,weight_hint,expected", [
    (None, None, "Energy Drinks", False, "Missing"),
    (350.0, "l", "Energy Drinks", False, "Flagged_Anomaly"),  # 350 L coconut water
    (350.0, "l", "Other Non-Cola Carbonates", False, "Flagged_Anomaly"),  # volume type
    (350.0, "l", "Liquid Concentrates", False, "Flagged_Anomaly"),  # volume type
    (12.0, "l", "Energy Drinks", False, "Flagged_Anomaly"),
    (10.0, "l", "Energy Drinks", False, "Valid"),  # rule is strictly > 10
    (1.5, "l", "Juice Drinks (up to 24% Juice)", False, "Valid"),
    (500.0, "ml", "Energy Drinks", False, "Valid"),
    (2.0, "oz", "Powder Concentrates", False, "Valid"),  # weight type: no liter rule
    (500.0, "ml", "Powder Concentrates", True, "Valid"),  # weight hint expected here
    (500.0, "ml", "Energy Drinks", True, "Flagged_Anomaly"),  # kg rule via hint
    (350.0, "l", "Mystery Future Category", False, "Flagged_Anomaly"),  # default volume
], ids=lambda c: str(c)[:40])
def test_validate_measurement(text_module, value, unit, category, weight_hint,
                              expected) -> None:
    result = text_module.validate_measurement(value, unit, category, weight_hint)
    assert result == expected, f"({value}, {unit}, {category}, {weight_hint}): " \
        f"got {result}, expected {expected}"


@pytest.mark.parametrize("category,expected", [
    ("Powder Concentrates", "weight"),
    ("powder concentrates", "weight"),  # normalization is case/punct-insensitive
    ("Other Non-Cola Carbonates", "volume"),
    ("Lemonade/Lime", "volume"),
    ("Energy Drinks", "volume"),
    ("RTD Coffee", "volume"),
    (None, "volume"),  # default
    ("Mystery Future Category", "volume"),  # default for unknown
])
def test_get_measurement_type(text_module, category, expected) -> None:
    assert text_module.get_measurement_type(category) == expected


def test_weight_unit_re_excludes_mg(text_module) -> None:
    """mg is a dose ('L-Carnitine 2000mg'), not net weight — no flag."""
    assert text_module.WEIGHT_UNIT_RE.search("L-Carnitine Drink 2000mg") is None
    assert text_module.WEIGHT_UNIT_RE.search("Protein Powder 2 kg") is not None
    assert text_module.WEIGHT_UNIT_RE.search("10 lb bag") is not None


def test_weight_unit_re_excludes_zero_claims(text_module) -> None:
    """'0g Added Sugar' is nutrition prose, not net weight — no flag."""
    assert text_module.WEIGHT_UNIT_RE.search("Drink 0g Added Sugar") is None
    assert text_module.WEIGHT_UNIT_RE.search("0 g fat") is None
    # EU decimal weights still match ("0,5 kg" = half a kilo)
    assert text_module.WEIGHT_UNIT_RE.search("Mix 0,5 kg") is not None
    assert text_module.WEIGHT_UNIT_RE.search("500 g powder") is not None


# ---------------------------------------------------------------------------
# TF-IDF cosine title matching (04): _matching.py primitive.
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def matching_module():
    import _matching
    return _matching


def test_identical_titles_score_1(matching_module) -> None:
    v = matching_module.build_vectorizer()
    X = v.fit_transform(["coconut water 350 ml", "coconut water 350 ml"])
    assert matching_module.score_pairs(X, [0], [1])[0] == pytest.approx(1.0)


def test_unrelated_titles_score_low(matching_module) -> None:
    v = matching_module.build_vectorizer()
    X = v.fit_transform(["coconut water 350 ml", "energy drink 250 ml"])
    assert matching_module.score_pairs(X, [0], [1])[0] < 0.2


def test_demo_pair_outscores_unrelated(matching_module) -> None:
    """The user's clearspring pair must beat an unrelated pair."""
    v = matching_module.build_vectorizer()
    X = v.fit_transform([
        "clearspring Organic King coconut 100% coconut water 350 ml",
        "coconut water ECO clearspring, 350 ml",
        "strawberry lollipop candy 50 g",
    ])
    same, unrelated = matching_module.score_pairs(X, [0, 0], [1, 2])
    assert same > unrelated
    assert same > 0.3
