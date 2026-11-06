from __future__ import annotations

from fractions import Fraction

import pytest

from certtables.enclose import enclose_gst
from certtables.enclose.continuous import quantile
from certtables.enclose.gst import Design, equal_fractions, mp_boundaries, parse_fractions


def _overlap(a, b) -> bool:
    return max(a.lo, b.lo) <= min(a.hi, b.hi)


@pytest.mark.parametrize("alpha", [0.05, 0.01])
def test_pocock_k1_is_the_normal_quantile(alpha):
    (c,) = enclose_gst(Design("pocock", alpha, equal_fractions(1)))
    z = quantile("norm", "isf", alpha / 2, {})
    assert _overlap(c, z)
    assert c.hi - c.lo < Fraction(1, 10**19)


def test_k2_constants_match_jennison_turnbull_and_mpmath():
    poc = enclose_gst(Design("pocock", 0.05, equal_fractions(2)))
    obf = enclose_gst(Design("obf", 0.05, equal_fractions(2)))
    assert abs(float(poc[0].lo) - 2.178) < 5e-4 and poc[0] == poc[1]
    assert abs(float(obf[1].lo) - 1.977) < 5e-4
    # c_1 = C_B sqrt(2) for O'Brien-Fleming
    assert abs(float(obf[0].lo) - float(obf[1].lo) * 2**0.5) < 1e-15
    ref = mp_boundaries(Design("pocock", 0.05, equal_fractions(2)), degree=6, dps=25)
    assert float(poc[0].lo) <= ref[0] * (1 + 1e-15) and ref[0] * (1 - 1e-15) <= float(poc[0].hi)


def test_spending_k2_is_tight_and_matches_reference():
    d = Design("ld-obf", 0.05, parse_fractions("3/10,1"))
    enc = enclose_gst(d)
    ref = mp_boundaries(d, degree=6, dps=25)
    for e, r in zip(enc, ref, strict=True):
        assert e.kind == "certified" and e.hi - e.lo < e.hi * Fraction(1, 10**17)
        assert abs(float(e.lo) - r) < 1e-13 * r


POWER_K5 = (3.0902, 2.7141, 2.4728, 2.2799, 2.1140)  # spotify-confidence 4.1.0, rho = 2


def test_power_family_k5_recursion_matches_published_spotify_values():
    got = mp_boundaries(Design("ld-power", 0.05, equal_fractions(5)), degree=4, dps=15)
    assert all(abs(g - w) < 6e-5 for g, w in zip(got, POWER_K5, strict=True))


@pytest.mark.slow
def test_k5_rows_are_references_not_certified():
    enc = enclose_gst(Design("ld-power", 0.05, equal_fractions(5)))
    assert all(e.kind == "reference" for e in enc)
    assert all(abs(float(e.lo) - w) < 6e-5 for e, w in zip(enc, POWER_K5, strict=True))


def test_design_validation():
    with pytest.raises(ValueError):
        Design("pocock", 0.05, parse_fractions("3/10,1"))
    with pytest.raises(ValueError):
        Design("ld-obf", 0.05, parse_fractions("1/2,1/3,1"))
    with pytest.raises(ValueError):
        Design("bonferroni", 0.05, equal_fractions(2))


@pytest.mark.slow
def test_k3_pocock_certified_by_nested_integration():
    enc = enclose_gst(Design("pocock", 0.05, equal_fractions(3)))
    assert enc[0].kind == "certified"
    assert abs(float(enc[0].lo) - 2.289) < 5e-4
    assert enc[0].hi - enc[0].lo < enc[0].hi * Fraction(1, 10**11)
    # The fixed-rule recursion behind the uncertified K = 4, 5 references lands on it too.
    ref = mp_boundaries(Design("pocock", 0.05, equal_fractions(3)), degree=6, dps=20)
    assert all(abs(float(e.lo) - r) < 1e-14 for e, r in zip(enc, ref, strict=True))
