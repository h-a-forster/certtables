"""Certified CDFs, survival functions and quantiles of the normal, t, chi-square, F and beta
distributions, and Clopper-Pearson bounds (docs/DESIGN.md sections 2.2-2.4).

Inputs are the exact binary64 values a caller passes to a double-precision implementation,
so ``x = 0.1`` means the double nearest 0.1. Every value is an Arb ball from Arb's rigorous
special functions:

- normal: ``erfc``; t, F and beta: the regularised lower incomplete beta function
  (``arb.beta_lower(..., regularized=1)``); chi-square: the regularised upper incomplete
  gamma function (``gamma_upper``) and, for the lower tail, a series with an explicit
  remainder bound (:func:`gamma_p_series`).

Quantiles are enclosed by :func:`certtables.enclose.arbutil.certified_root` on
``CDF(x) - p`` (increasing) or ``SF(x) - q`` (decreasing).
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from fractions import Fraction

from flint import arb, ctx

from certtables.enclose.arbutil import (
    BASE_PREC,
    MAX_PREC,
    EnclosureError,
    arb_to_fraction,
    certified_root,
    precision,
)

#: Relative radius at which a function value counts as tight enough.
VALUE_REL_WIDTH = 1e-22
#: Relative width at which a quantile bracket counts as tight enough.
ROOT_REL_WIDTH = 1e-20
#: Quantiles beyond this magnitude (above the binary64 range) are not enclosed.
X_MAX = Fraction(2) ** 1100
#: Positive quantiles below this (under the binary64 range) are not enclosed, and function
#: values certainly below it are stored as the enclosure [0, TINY]: every double rounds such
#: a value to 0 and the exact digits would cost megabit rationals (DESIGN 3.3).
TINY = Fraction(1, 2**1100)


@dataclass(frozen=True)
class Enclosure:
    """A certified (or, for ``kind='reference'``, uncertified) interval for one value."""

    lo: Fraction
    hi: Fraction
    method: str
    prec: int
    kind: str = "certified"  # certified | exact | reference


# --------------------------------------------------------------------------- functions
# Each takes exact Arb arguments and returns a ball. Parameters are positive.


def _half() -> arb:
    return arb(1) / 2


def norm_cdf(x: arb) -> arb:
    return (-x / arb(2).sqrt()).erfc() / 2


def norm_sf(x: arb) -> arb:
    return (x / arb(2).sqrt()).erfc() / 2


def _t_tail(x: arb, df: arb) -> arb:
    """P(T > x) for x > 0: I_{df/(df+x^2)}(df/2, 1/2) / 2."""
    z = df / (df + x * x)
    return z.beta_lower(df / 2, _half(), regularized=1) / 2


def t_sf(x: arb, df: arb) -> arb:
    if x.is_zero():
        return _half()
    if x > 0:
        return _t_tail(x, df)
    return 1 - _t_tail(-x, df)


def t_cdf(x: arb, df: arb) -> arb:
    return t_sf(-x, df)


def gamma_p_series(s: arb, z: arb) -> arb:
    """Regularised lower incomplete gamma P(s, z) for 0 < z <= s + 1 from the series
    P = z^s e^-z / Gamma(s+1) * sum_k z^k / ((s+1)...(s+k)), with a rigorous tail bound.

    Needed because ``arb.gamma_lower`` in python-flint 0.8.0 returns non-finite or very wide
    balls (and takes about a second) for large s near z = s (DESIGN 2.2). After N terms the
    term ratios are at most r = z/(s+N+1) < 1, so the remainder is at most t_N / (1 - r).
    """
    if not (z > 0 and z <= s + 1):
        raise ValueError("series used outside 0 < z <= s + 1")
    total = arb(1)
    term = arb(1)
    eps = arb(2) ** (-ctx.prec - 10)
    k = 0
    while True:
        k += 1
        term = term * z / (s + k)
        total += term
        r = z / (s + k + 1)
        if r < 1:
            tail = term * r / (1 - r)
            if tail < total * eps:
                break
        if k > 10_000_000:  # pragma: no cover - guard against a non-terminating loop
            raise EnclosureError("gamma series did not converge")
    total += arb(0, tail.upper())
    return (s * z.log() - z - (s + 1).lgamma()).exp() * total


def chi2_cdf(x: arb, k: arb) -> arb:
    if not x > 0:
        return arb(0)
    z, s = x / 2, k / 2
    if z <= s + 1:
        return gamma_p_series(s, z)
    return 1 - z.gamma_upper(s, regularized=1)


def chi2_sf(x: arb, k: arb) -> arb:
    if not x > 0:
        return arb(1)
    return (x / 2).gamma_upper(k / 2, regularized=1)


def f_cdf(x: arb, d1: arb, d2: arb) -> arb:
    if not x > 0:
        return arb(0)
    w = d1 * x / (d1 * x + d2)
    return w.beta_lower(d1 / 2, d2 / 2, regularized=1)


def f_sf(x: arb, d1: arb, d2: arb) -> arb:
    if not x > 0:
        return arb(1)
    w = d2 / (d2 + d1 * x)
    return w.beta_lower(d2 / 2, d1 / 2, regularized=1)


def beta_cdf(x: arb, a: arb, b: arb) -> arb:
    if not x > 0:
        return arb(0)
    if not x < 1:
        return arb(1)
    return x.beta_lower(a, b, regularized=1)


def beta_sf(x: arb, a: arb, b: arb) -> arb:
    if not x > 0:
        return arb(1)
    if not x < 1:
        return arb(0)
    return (1 - x).beta_lower(b, a, regularized=1)


#: family -> (cdf, sf, parameter names, method label, lower end of the support)
FAMILIES: dict[str, tuple[Callable[..., arb], Callable[..., arb], tuple[str, ...], str, float]] = {
    "norm": (norm_cdf, norm_sf, (), "erfc", -math.inf),
    "t": (t_cdf, t_sf, ("df",), "beta_lower", -math.inf),
    "chi2": (chi2_cdf, chi2_sf, ("df",), "gamma_upper/series", 0.0),
    "f": (f_cdf, f_sf, ("dfn", "dfd"), "beta_lower", 0.0),
    "beta": (beta_cdf, beta_sf, ("a", "b"), "beta_lower", 0.0),
}


def _check_params(family: str, params: dict[str, float]) -> list[float]:
    if family not in FAMILIES:
        raise ValueError(f"unknown family {family!r}")
    names = FAMILIES[family][2]
    if set(params) - set(names) - {"x", "p", "q"}:
        raise ValueError(f"unexpected parameters for {family}: {sorted(params)}")
    vals = [float(params[n]) for n in names]
    for n, v in zip(names, vals, strict=True):
        if not (math.isfinite(v) and v > 0):
            raise ValueError(f"{family}: {n} must be finite and positive, got {v!r}")
    return vals


def value(family: str, function: str, x: float, params: dict[str, float]) -> Enclosure:
    """Enclosure of ``cdf`` or ``sf`` at the double ``x``, escalating precision until the
    relative radius is at most :data:`VALUE_REL_WIDTH`."""
    cdf, sf, _, label, _ = FAMILIES[family]
    vals = _check_params(family, params)
    if function not in ("cdf", "sf"):
        raise ValueError(f"function must be cdf or sf, got {function!r}")
    if not math.isfinite(x):
        raise ValueError(f"x must be finite, got {x!r}")
    fn = cdf if function == "cdf" else sf
    prec = BASE_PREC
    last = None
    while prec <= MAX_PREC:
        with precision(prec):
            v = fn(arb(x), *[arb(p) for p in vals])
            if v.is_finite():
                last = (v, prec)
                if v.is_exact() or v.rad() <= abs(v.mid()) * arb(VALUE_REL_WIDTH):
                    break
        prec *= 2
    if last is None:
        # beta_lower can return nan up to MAX_PREC close to 1 (DESIGN 2.2): use the complement.
        other = sf if function == "cdf" else cdf
        with precision(MAX_PREC):
            v = 1 - other(arb(x), *[arb(p) for p in vals])
            if v.is_finite() and v.rad() <= abs(v.mid()) * arb(VALUE_REL_WIDTH):
                last = (v, MAX_PREC)
    if last is None:
        raise EnclosureError(f"{family}.{function}({x}, {vals}): no finite ball")
    v, prec = last
    with precision(prec):
        if v.upper() < arb(2) ** -1100:
            return Enclosure(Fraction(0), TINY, f"arb:{label};below-2^-1100", prec)
        lo, hi = arb_to_fraction(v)
    return Enclosure(max(lo, Fraction(0)), min(hi, Fraction(1)), f"arb:{label}", prec)


def quantile(family: str, function: str, p: float, params: dict[str, float]) -> Enclosure:
    """Enclosure of ``ppf(p)`` (root of ``CDF(x) - p``) or ``isf(p)`` (root of ``SF(x) - p``)
    at the double ``p`` in (0, 1). The search starts at 1 on purpose: a starting guess from
    SciPy would make the digits of the reference depend on an implementation under test."""
    cdf, sf, _, label, support_lo = FAMILIES[family]
    vals = _check_params(family, params)
    if function not in ("ppf", "isf"):
        raise ValueError(f"function must be ppf or isf, got {function!r}")
    if not (0.0 < p < 1.0):
        raise ValueError(f"p must be in (0, 1), got {p!r}")
    pa = arb(p)

    if function == "ppf":

        def G(x: arb) -> arb:
            return cdf(x, *[arb(v) for v in vals]) - pa
    else:

        def G(x: arb) -> arb:
            return sf(x, *[arb(v) for v in vals]) - pa

    lower = None if support_lo == -math.inf else Fraction(support_lo)
    br = certified_root(
        G,
        1.0,
        increasing=function == "ppf",
        rel_width=ROOT_REL_WIDTH,
        lower=lower,
        x_max=X_MAX,
        x_min=TINY if lower is not None else None,
    )
    return Enclosure(br.lo, br.hi, f"arb-root:{label}", br.prec)


def clopper_pearson(k: int, n: int, alpha: float, side: str) -> Enclosure:
    """Exact (Clopper-Pearson) bound for ``k`` successes in ``n`` trials at level
    ``1 - alpha``: ``lower`` solves ``P(X >= k | p) = alpha/2`` (beta ``ppf(alpha/2; k,
    n-k+1)``), ``upper`` solves ``P(X <= k | p) = alpha/2`` (beta ``isf(alpha/2; k+1, n-k)``).
    ``alpha/2`` is the double ``alpha / 2`` (exact halving)."""
    if not (isinstance(k, int) and isinstance(n, int) and 0 <= k <= n and n >= 1):
        raise ValueError(f"need integers 0 <= k <= n, n >= 1; got k={k!r}, n={n!r}")
    if not (0.0 < alpha < 1.0):
        raise ValueError(f"alpha must be in (0, 1), got {alpha!r}")
    if side == "lower":
        if k == 0:
            return Enclosure(Fraction(0), Fraction(0), "exact:boundary", 0, "exact")
        return quantile("beta", "ppf", alpha / 2, {"a": float(k), "b": float(n - k + 1)})
    if side == "upper":
        if k == n:
            return Enclosure(Fraction(1), Fraction(1), "exact:boundary", 0, "exact")
        return quantile("beta", "isf", alpha / 2, {"a": float(k + 1), "b": float(n - k)})
    raise ValueError(f"side must be 'lower' or 'upper', got {side!r}")
