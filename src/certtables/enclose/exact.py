"""Exact rational null distributions and p-values (docs/DESIGN.md section 2.5).

Everything here is a :class:`fractions.Fraction` computed with integer arithmetic (FLINT's
``fmpq_mat`` for the Kolmogorov-Smirnov matrix power); nothing is rounded. Probabilities
given as doubles (the binomial ``p``, the KS ``d``) are taken at their exact binary value.
"""

from __future__ import annotations

import math
from fractions import Fraction
from functools import lru_cache

# ------------------------------------------------------------------------------ binomial


def binom_cdf(k: int, n: int, p: float) -> Fraction:
    """P(X <= k) for X ~ Binomial(n, p), ``p`` the exact value of a double in [0, 1]."""
    _check_n(n)
    pf = Fraction(p)
    if not 0 <= pf <= 1:
        raise ValueError(f"p must be in [0, 1], got {p!r}")
    if k < 0:
        return Fraction(0)
    if k >= n:
        return Fraction(1)
    num, den = pf.numerator, pf.denominator
    q = den - num
    total = sum(math.comb(n, i) * num**i * q ** (n - i) for i in range(k + 1))
    return Fraction(total, den**n)


def binom_sf(k: int, n: int, p: float) -> Fraction:
    """P(X > k)."""
    return 1 - binom_cdf(k, n, p)


# -------------------------------------------------------------------------------- Fisher


def _check_n(n: int) -> None:
    if not (isinstance(n, int) and n >= 1):
        raise ValueError(f"n must be a positive integer, got {n!r}")


def fisher_pvalue(a: int, b: int, c: int, d: int, alternative: str) -> Fraction:
    """Fisher's exact test on the 2x2 table ``[[a, b], [c, d]]``, conditioning on both
    margins. ``less``: P(X <= a); ``greater``: P(X >= a); ``two-sided``: the sum of the
    probabilities of all tables with probability <= that of the observed table (exact
    comparison, no tolerance)."""
    if min(a, b, c, d) < 0 or not all(isinstance(v, int) for v in (a, b, c, d)):
        raise ValueError("table entries must be non-negative integers")
    r1, c1, n = a + b, a + c, a + b + c + d
    lo, hi = max(0, c1 - (n - r1)), min(r1, c1)
    weights = {x: math.comb(r1, x) * math.comb(n - r1, c1 - x) for x in range(lo, hi + 1)}
    total = math.comb(n, c1)
    if alternative == "less":
        s = sum(w for x, w in weights.items() if x <= a)
    elif alternative == "greater":
        s = sum(w for x, w in weights.items() if x >= a)
    elif alternative == "two-sided":
        s = sum(w for w in weights.values() if w <= weights[a])
    else:
        raise ValueError(f"unknown alternative {alternative!r}")
    return Fraction(s, total)


# --------------------------------------------------------------------- Wilcoxon, Mann-Whitney


@lru_cache(maxsize=64)
def wilcoxon_counts(n: int) -> tuple[int, ...]:
    """Number of subsets of {1..n} with each sum w = 0..n(n+1)/2 (W+ null counts)."""
    _check_n(n)
    c = [1] + [0] * (n * (n + 1) // 2)
    top = 0
    for i in range(1, n + 1):
        top += i
        for w in range(top, i - 1, -1):
            c[w] += c[w - i]
    return tuple(c)


def _tail_pvalue(counts: tuple[int, ...], stat: int, alternative: str) -> Fraction:
    total = sum(counts)
    stat = int(stat)
    if not 0 <= stat < len(counts):
        raise ValueError(f"statistic {stat} outside the support 0..{len(counts) - 1}")
    le = Fraction(sum(counts[: stat + 1]), total)
    ge = Fraction(sum(counts[stat:]), total)
    if alternative == "less":
        return le
    if alternative == "greater":
        return ge
    if alternative == "two-sided":
        return min(Fraction(1), 2 * min(le, ge))
    raise ValueError(f"unknown alternative {alternative!r}")


def wilcoxon_pvalue(n: int, w_plus: int, alternative: str) -> Fraction:
    """Exact signed-rank p-value for the sum of positive ranks ``w_plus`` (no ties, no
    zeros). ``greater``: P(W+ >= w); ``less``: P(W+ <= w); two-sided: min(1, 2 min)."""
    return _tail_pvalue(wilcoxon_counts(n), w_plus, alternative)


@lru_cache(maxsize=256)
def mannwhitney_counts(m: int, n: int) -> tuple[int, ...]:
    """Null counts of U = #{(i, j): x_i > y_j} for sample sizes m and n (no ties): the
    coefficients of the Gaussian binomial [m+n choose m]_q."""
    _check_n(m)
    _check_n(n)
    size = m * n + 1
    c = [1] + [0] * (size - 1)
    for i in range(1, m + 1):
        # multiply by (1 - q^(n+i))
        s = n + i
        for u in range(size - 1, s - 1, -1):
            c[u] -= c[u - s]
        # divide by (1 - q^i): running sum with stride i
        for u in range(i, size):
            c[u] += c[u - i]
    if sum(c) != math.comb(m + n, m) or min(c) < 0:
        raise AssertionError("Gaussian binomial recursion failed")  # pragma: no cover
    return tuple(c)


def mannwhitney_pvalue(m: int, n: int, u: int, alternative: str) -> Fraction:
    """Exact Mann-Whitney p-value for U of the first sample (size ``m``); ``greater``:
    P(U >= u), ``less``: P(U <= u), two-sided: min(1, 2 min)."""
    return _tail_pvalue(mannwhitney_counts(m, n), u, alternative)


# --------------------------------------------------------------------- Kolmogorov-Smirnov


def ks_cdf(n: int, d: float) -> Fraction:
    """P(D_n <= d) for the two-sided one-sample KS statistic, ``d`` the exact value of a
    double. Durbin's matrix formula (Marsaglia, Tsang and Wang 2003) in exact rational
    arithmetic for d < 1/2; for d >= 1/2 the two one-sided tails are disjoint and
    P(D_n >= d) = 2 P(D_n^+ >= d) with the Birnbaum-Tingey sum."""
    _check_n(n)
    df = Fraction(d)
    if df <= Fraction(1, 2 * n):
        return Fraction(0)
    if df >= 1:
        return Fraction(1)
    if df >= Fraction(1, 2):
        return 1 - 2 * _smirnov_sf(n, df)
    return _durbin(n, df)


def ks_sf(n: int, d: float) -> Fraction:
    """P(D_n > d)."""
    return 1 - ks_cdf(n, d)


#: Largest Durbin matrix dimension 2k - 1 that :func:`ks_cdf` accepts.
DURBIN_MAX_DIM = 121


def _durbin(n: int, d: Fraction) -> Fraction:
    from flint import fmpq, fmpq_mat

    nd = n * d
    k = math.floor(nd) + 1
    m = 2 * k - 1
    if m > DURBIN_MAX_DIM:
        raise ValueError(f"Durbin matrix of dimension {m} exceeds {DURBIN_MAX_DIM}")
    h = k - nd  # in (0, 1]
    H = [[Fraction(1) if i - j + 1 >= 0 else Fraction(0) for j in range(m)] for i in range(m)]
    for i in range(m):
        H[i][0] -= h ** (i + 1)
        H[m - 1][i] -= h ** (m - i)
    if 2 * h - 1 > 0:
        H[m - 1][0] += (2 * h - 1) ** m
    for i in range(m):
        for j in range(m):
            if i - j + 1 > 0:
                H[i][j] /= math.factorial(i - j + 1)
    M = fmpq_mat(m, m, [fmpq(x.numerator, x.denominator) for row in H for x in row])
    P = _matpow(M, n)
    s = P[k - 1, k - 1]
    val = Fraction(int(s.p), int(s.q))
    return val * math.factorial(n) / Fraction(n) ** n


def _matpow(M, e: int):
    result = None
    base = M
    while e:
        if e & 1:
            result = base if result is None else result * base
        e >>= 1
        if e:
            base = base * base
    return result


def _smirnov_sf(n: int, d: Fraction) -> Fraction:
    """P(D_n^+ >= d), Birnbaum and Tingey (1951), exact for rational d in (0, 1)."""
    total = Fraction(0)
    for j in range(math.floor(n * (1 - d)) + 1):
        a = 1 - d - Fraction(j, n)
        b = d + Fraction(j, n)
        total += math.comb(n, j) * a ** (n - j) * b ** (j - 1)
    return d * total
