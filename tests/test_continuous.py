"""Enclosures must contain closed-form values (computed independently in Arb at 256 bits)."""

from __future__ import annotations

import math
import sys
from fractions import Fraction

import pytest
from flint import arb

from certtables.enclose.arbutil import RootOutOfRange, arb_to_fraction, precision
from certtables.enclose.continuous import (
    Enclosure,
    chi2_sf,
    clopper_pearson,
    gamma_p_series,
    quantile,
    value,
)


def _contains(enc: Enclosure, exact: arb) -> bool:
    """The (tiny) closed-form ball overlaps the enclosure."""
    lo, hi = arb_to_fraction(exact)
    return max(lo, enc.lo) <= min(hi, enc.hi)


def _tight(enc: Enclosure, rel: float = 1e-19) -> bool:
    return enc.hi - enc.lo <= Fraction(rel) * max(abs(enc.lo), abs(enc.hi))


@pytest.mark.parametrize("p", [1e-300, 1e-10, 0.025, 0.3, 0.975, 0.999])
def test_t_df1_and_df2_quantiles_closed_form(p):
    e1 = quantile("t", "ppf", p, {"df": 1.0})
    e2 = quantile("t", "ppf", p, {"df": 2.0})
    with precision(2048):  # tan near -pi/2 needs p - 1/2 to ~1000 bits
        pa = arb(p)
        t1 = (arb.pi() * (pa - arb(1) / 2)).tan()  # Cauchy
        t2 = (2 * pa - 1) / (2 * pa * (1 - pa)).sqrt()
        assert _contains(e1, t1) and _contains(e2, t2)
    assert _tight(e1) and _tight(e2)


@pytest.mark.parametrize("p", [1e-200, 1e-5, 0.5, 0.9])
def test_chi2_df2_is_exponential(p):
    e = quantile("chi2", "ppf", p, {"df": 2.0})
    with precision(256):
        assert _contains(e, -2 * (1 - arb(p)).log())
    q = quantile("chi2", "isf", p, {"df": 2.0})
    with precision(256):
        assert _contains(q, -2 * arb(p).log())


@pytest.mark.parametrize("x", [1e-3, 0.7, 5.0, 60.0])
def test_chi2_df1_cdf_and_f22_cdf(x):
    with precision(256):
        assert _contains(value("chi2", "cdf", x, {"df": 1.0}), (arb(x) / 2).sqrt().erf())
        assert _contains(value("f", "cdf", x, {"dfn": 2.0, "dfd": 2.0}), arb(x) / (1 + arb(x)))


def test_normal_tails_and_symmetry():
    # The double 0.975 is 0.97499999999999997780..., so its quantile rounds to
    # 1.9599639845400538, one ulp below the familiar 1.959963984540054 = z_{0.975}.
    e = quantile("norm", "ppf", 0.975, {})
    assert float(e.lo) == float(e.hi) == 1.9599639845400538
    deep = value("norm", "sf", 37.0, {})
    assert deep.lo > 0 and _tight(deep, 1e-21)
    assert quantile("norm", "ppf", 0.5, {}).hi == 0


def test_tiny_values_become_bounds():
    e = value("t", "cdf", -1e10, {"df": 1e6})
    assert e.lo == 0 and e.hi == Fraction(1, 2**1100) and "below" in e.method


@pytest.mark.parametrize(("n", "alpha"), [(1, 0.05), (7, 0.01), (10000, 0.05)])
def test_clopper_pearson_closed_forms(n, alpha):
    # k = n: lower bound solves p^n = alpha/2; k = 0: upper bound solves (1-p)^n = alpha/2.
    lo = clopper_pearson(n, n, alpha, "lower")
    up = clopper_pearson(0, n, alpha, "upper")
    with precision(256):
        r = (arb(alpha / 2).log() / n).exp()
        assert _contains(lo, r) and _contains(up, 1 - r)
    assert clopper_pearson(n, n, alpha, "upper").lo == 1
    assert clopper_pearson(0, n, alpha, "lower").hi == 0


def test_gamma_series_agrees_with_upper_gamma_complement():
    with precision(128):
        for s, z in [(2.5, 1.0), (150.0, 140.0), (0.05, 0.04), (5000.0, 4990.0)]:
            p = gamma_p_series(arb(s), arb(z))
            q = arb(z).gamma_upper(arb(s), regularized=1)
            assert (p + q - 1).contains(0)
            assert p.rad() < abs(p.mid()) * arb(2) ** -100


def test_large_df_chi2_cdf_is_tight():
    # python-flint's gamma_lower returns a non-finite ball here; the series must not.
    e = value("chi2", "cdf", 999000.0, {"df": 1e6})
    assert _tight(e, 1e-21) and 0.2 < float(e.lo) < 0.3
    with precision(128):
        assert chi2_sf(arb(999000.0), arb(1e6)).is_finite()


def test_bad_inputs_fail_loudly():
    with pytest.raises(ValueError):
        quantile("t", "ppf", 1.0, {"df": 3.0})
    with pytest.raises(ValueError):
        value("t", "cdf", 1.0, {"df": -1.0})
    with pytest.raises(ValueError):
        value("t", "pdf", 1.0, {"df": 1.0})
    with pytest.raises(ValueError):
        clopper_pearson(5, 3, 0.05, "lower")
    assert math.isfinite(float(value("f", "sf", 2.0, {"dfn": 3.0, "dfd": 7.5}).lo))


@pytest.mark.parametrize(
    ("family", "fn", "x", "params"),
    [
        ("t", "cdf", -3.5, {"df": 7.5}),
        ("t", "sf", 40.0, {"df": 3.0}),
        ("chi2", "sf", 250.0, {"df": 100.0}),
        ("chi2", "cdf", 0.02, {"df": 4.0}),
        ("f", "sf", 6.0, {"dfn": 3.0, "dfd": 7.5}),
        ("f", "cdf", 0.1, {"dfn": 10.0, "dfd": 100.0}),
    ],
)
def test_values_agree_with_independent_mpmath(family, fn, x, params):
    """An independent implementation (mpmath at 60 digits) lands inside the enclosure
    widened by 1e-40 relative (its own error)."""
    import mpmath

    enc = value(family, fn, x, params)
    with mpmath.workdps(60):
        mx = mpmath.mpf(x)
        if family == "t":
            df = mpmath.mpf(params["df"])
            tail = mpmath.betainc(df / 2, 0.5, 0, df / (df + mx**2), regularized=True) / 2
            v = tail if (x > 0) == (fn == "sf") else 1 - tail
        elif family == "chi2":
            k = mpmath.mpf(params["df"])
            v = (
                mpmath.gammainc(k / 2, mx / 2, mpmath.inf, regularized=True)
                if fn == "sf"
                else mpmath.gammainc(k / 2, 0, mx / 2, regularized=True)
            )
        else:
            d1, d2 = mpmath.mpf(params["dfn"]), mpmath.mpf(params["dfd"])
            v = (
                mpmath.betainc(d2 / 2, d1 / 2, 0, d2 / (d2 + d1 * mx), regularized=True)
                if fn == "sf"
                else mpmath.betainc(d1 / 2, d2 / 2, 0, d1 * mx / (d1 * mx + d2), regularized=True)
            )
        vq = Fraction(mpmath.nstr(v, 55, min_fixed=1, max_fixed=0))
    slack = abs(vq) * Fraction(1, 10**40)
    assert enc.lo - slack <= vq <= enc.hi + slack


def test_cdf_falls_back_to_the_complement_when_beta_lower_returns_nan():
    # beta_lower gives nan for this point up to 1024 bits; 1 - SF is rigorous and tight.
    enc = value("f", "cdf", 4666.36, {"dfn": 1.81417, "dfd": 772.678})
    assert enc.hi == 1 and 1 - enc.lo < Fraction(1, 2**1000)


@pytest.mark.parametrize(
    ("family", "function", "p", "params"),
    [
        ("t", "ppf", 1e-300, {"df": 0.1}),  # |x| ~ 1e3000
        ("f", "isf", 1e-300, {"dfn": 1.0, "dfd": 0.1}),
        ("chi2", "ppf", 1e-300, {"df": 0.1}),  # x ~ 1e-3000
    ],
)
def test_quantile_outside_the_search_range_is_proved_not_just_missed(family, function, p, params):
    with pytest.raises(RootOutOfRange):
        quantile(family, function, p, params)


def test_reference_digits_do_not_depend_on_an_implementation_under_test(monkeypatch):
    first = quantile("t", "ppf", 1e-10, {"df": 2.5})
    monkeypatch.setitem(sys.modules, "scipy", None)  # importing scipy now raises ImportError
    assert quantile("t", "ppf", 1e-10, {"df": 2.5}) == first


def test_quantile_just_above_the_lower_limit_is_enclosed():
    # About 1.6e-320, between 2^-1100 and the last power 2^(-2^k) of the geometric search.
    e = quantile("chi2", "ppf", 1e-160, {"df": 1.0})
    assert 1e-320 < float(e.lo) <= float(e.hi) < 2e-320
