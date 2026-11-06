from __future__ import annotations

import math
from fractions import Fraction

import pytest

from certtables.study import lre_bounds, verdict


def test_lre_hand_cases():
    one = Fraction(1)
    # 1.0000001 against exactly 1: 7 digits
    lo, hi = lre_bounds(1.0000001, one, one)
    assert lo == hi == pytest.approx(7.0, abs=1e-6)
    # inside the enclosure: best case is capped, worst case from the far endpoint
    lo, hi = lre_bounds(1.0, Fraction(999, 1000), Fraction(1001, 1000))
    assert hi == 15 and lo == pytest.approx(3.0, abs=1e-3)
    # reference interval containing 0: absolute error (LAE)
    lo, hi = lre_bounds(1e-8, Fraction(0), Fraction(0))
    assert lo == hi == pytest.approx(8.0)
    # below one digit reports 0; non-finite reports nan
    assert lre_bounds(2.0, one, one) == (0.0, 0.0)
    assert all(math.isnan(x) for x in lre_bounds(math.nan, one, one))


def test_lre_does_not_underflow_for_deep_tails():
    c = Fraction(1, 10**400)
    lo, hi = lre_bounds(0.0, c, c)
    assert lo == hi == 0.0


def test_verdicts():
    one = Fraction(1)
    assert verdict(1.0 + 1e-13, one, one, 1e-12, 0.0) == "correct"
    assert verdict(1.0 + 1e-11, one, one, 1e-12, 0.0) == "wrong"
    assert verdict(math.nan, one, one, 1e-12, 0.0) == "wrong"
    # a wide enclosure cannot decide
    assert verdict(1.0, one, one + Fraction(1, 10**6), 1e-12, 0.0) == "undecided"
    # absolute floor: 0 for a value below DBL_MIN is fine, for 1e-300 it is not
    assert verdict(0.0, Fraction(1, 10**320), Fraction(1, 10**320), 1e-12, 2.2e-308) == "correct"
    assert verdict(0.0, Fraction(1, 10**300), Fraction(1, 10**300), 1e-12, 2.2e-308) == "wrong"
