"""Exact distributions: total mass 1 exactly, brute-force agreement, two formulas agree."""

from __future__ import annotations

import itertools
import math
from fractions import Fraction

import pytest

from certtables.enclose import exact as E


@pytest.mark.parametrize("n", [1, 4, 9])
def test_wilcoxon_counts_match_enumeration(n):
    brute = [0] * (n * (n + 1) // 2 + 1)
    for signs in itertools.product((0, 1), repeat=n):
        brute[sum(r for r, s in zip(range(1, n + 1), signs, strict=True) if s)] += 1
    assert list(E.wilcoxon_counts(n)) == brute
    assert sum(E.wilcoxon_counts(30)) == 2**30


@pytest.mark.parametrize(("m", "n"), [(1, 1), (3, 4), (5, 3)])
def test_mannwhitney_counts_match_enumeration(m, n):
    brute = [0] * (m * n + 1)
    for xpos in itertools.combinations(range(m + n), m):
        xs = set(xpos)
        # U = number of (x, y) pairs with x after y in the pooled order
        u = sum(1 for i in xs for j in range(m + n) if j not in xs and j < i)
        brute[u] += 1
    assert list(E.mannwhitney_counts(m, n)) == brute
    assert sum(E.mannwhitney_counts(50, 50)) == math.comb(100, 50)


def test_binomial_mass_and_brute_force():
    p = 0.1
    pf = Fraction(p)
    assert E.binom_cdf(10, 10, p) == 1
    brute = sum(math.comb(10, i) * pf**i * (1 - pf) ** (10 - i) for i in range(4))
    assert E.binom_cdf(3, 10, p) == brute
    assert E.binom_cdf(3, 10, p) + E.binom_sf(3, 10, p) == 1


def test_fisher_tails_and_mass():
    a, b, c, d = 8, 2, 1, 5
    less, greater = E.fisher_pvalue(a, b, c, d, "less"), E.fisher_pvalue(a, b, c, d, "greater")
    pa = Fraction(math.comb(10, 8) * math.comb(6, 1), math.comb(16, 9))
    assert less + greater - pa == 1
    # two-sided: tables at most as likely as the observed one, exact comparison
    assert E.fisher_pvalue(3, 1, 1, 3, "two-sided") == Fraction(17, 35)


def test_tail_pvalues_two_sided_definition():
    counts = E.wilcoxon_counts(5)
    le = Fraction(sum(counts[:3]), 32)
    assert E.wilcoxon_pvalue(5, 2, "less") == le
    assert E.wilcoxon_pvalue(5, 2, "two-sided") == 2 * le
    assert E.wilcoxon_pvalue(5, 7, "two-sided") == 1  # capped


def test_ks_small_n_closed_form_and_formula_agreement():
    # n = 1: P(D_1 <= d) = 2d - 1 on [1/2, 1]
    assert E.ks_cdf(1, 0.75) == Fraction(1, 2)
    # Durbin's matrix and the Birnbaum-Tingey sum agree exactly where both apply (d >= 1/2).
    for n, d in [(3, 0.5), (5, 0.625), (10, 0.75), (20, 0.5)]:
        assert E._durbin(n, Fraction(d)) == 1 - 2 * E._smirnov_sf(n, Fraction(d))
    assert E.ks_cdf(10, 0.01) == 0 and E.ks_cdf(10, 1.0) == 1


def test_ks_known_value():
    # Marsaglia, Tsang and Wang (2003) quote P(D_10 < 0.274) = 0.6284796154565043
    assert abs(float(E.ks_cdf(10, 0.274)) - 0.6284796154565043) < 1e-15


def test_bad_inputs_fail_loudly():
    with pytest.raises(ValueError):
        E.fisher_pvalue(1, 2, 3, 4, "both")
    with pytest.raises(ValueError):
        E.wilcoxon_pvalue(5, 16, "less")
    with pytest.raises(ValueError):
        E.binom_cdf(1, 0, 0.5)
