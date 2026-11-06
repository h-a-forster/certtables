from __future__ import annotations

from fractions import Fraction

import pytest
from flint import arb

from certtables.enclose.arbutil import (
    EnclosureError,
    arb_to_fraction,
    ball,
    certified_root,
    dyadic,
    fraction_to_arb,
    precision,
    to_decimal,
)


@pytest.mark.parametrize(
    ("x", "down", "up"),
    [
        (Fraction(1, 3), "3.33e-1", "3.34e-1"),
        (Fraction(-1, 3), "-3.34e-1", "-3.33e-1"),
        (Fraction(999999, 1000), "9.99e2", "1e3"),
        (Fraction(5), "5e0", "5e0"),
        (Fraction(1, 10**400), "1e-400", "1e-400"),
    ],
)
def test_to_decimal_rounds_outward(x, down, up):
    assert to_decimal(x, 3, "down") == down
    assert to_decimal(x, 3, "up") == up
    assert Fraction(down) <= x <= Fraction(up)


def test_to_decimal_rejects_bad_direction():
    with pytest.raises(ValueError):
        to_decimal(Fraction(1), 3, "nearest")


def test_dyadic_and_exact_arb_roundtrip():
    q = dyadic(Fraction(1, 3), bits=60)
    assert q.denominator & (q.denominator - 1) == 0
    assert abs(q - Fraction(1, 3)) <= Fraction(1, 3) * Fraction(1, 2**59)
    a = fraction_to_arb(q)
    assert a.is_exact()
    assert arb_to_fraction(a) == (q, q)
    with pytest.raises(ValueError):
        fraction_to_arb(Fraction(1, 3))


def test_ball_contains_interval():
    lo, hi = Fraction(1, 3), Fraction(1, 3) + Fraction(1, 10**25)
    with precision(128):
        b = ball(lo, hi)
        blo, bhi = arb_to_fraction(b)
    assert blo <= lo and hi <= bhi


def _cube_minus_two(x: arb) -> arb:
    return x * x * x - 2


@pytest.mark.parametrize("guess", [1.26, 1e-9, 1e6, -50.0, 1.2599210498948732])
def test_root_bracket_is_certified_for_any_guess(guess):
    br = certified_root(_cube_minus_two, guess, increasing=True, rel_width=1e-25)
    # lo^3 < 2 < hi^3 exactly: the bracket really contains 2^(1/3).
    assert br.lo**3 < 2 < br.hi**3
    assert br.hi - br.lo <= Fraction(1, 10**25) * 2


def test_root_decreasing_and_positive_domain():
    # G(x) = 1/x - 3 decreases on x > 0; root 1/3 (not dyadic).
    br = certified_root(lambda x: 1 / x - 3, 1e5, increasing=False, lower=Fraction(0))
    assert br.lo < Fraction(1, 3) < br.hi


def test_root_exact_zero_returns_point():
    br = certified_root(lambda x: x - arb(0.5), 0.5, increasing=True)
    assert br.lo == br.hi == Fraction(1, 2)


def test_root_x_max_stops_search():
    with pytest.raises(EnclosureError):
        certified_root(
            lambda x: x - arb(2) ** 2000, 1.0, increasing=True, x_max=Fraction(2) ** 1100
        )
