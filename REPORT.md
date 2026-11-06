# certtables report: a digits-correct census of the Python stats stack

2026-11-06. Every number comes from `data/`, `results/census-grid.csv`,
`results/census-random.csv`, `results/wrong.csv`, `results/upstream-checks.json` or
`results/crosscheck.json`, all committed; `results/census.md` has the full target x family
tables. Commands are at the end. Method and proofs: [docs/DESIGN.md](docs/DESIGN.md).

## Summary

- **Reference.** 16,203 rows: 14,726 certified Arb enclosures, 1,414 exact rationals and 63
  uncertified references (K = 4, 5 group-sequential boundaries). The relative width of a
  certified row is at most 3.6e-15 (581 rows are instead the bound `[0, 2^-1100]`), and
  nothing scored came out undecided.
- **SciPy 1.18.1 is right where most people use it.** On the deterministic grid of 6,365
  scored points, the normal functions, Clopper-Pearson bounds, binomial tails, Fisher,
  Wilcoxon and Mann-Whitney had no certified-wrong point (1,593 points). 167 points were
  wrong (2.6%), and of the 6,165 points with a normal-range reference, 127 had fewer than 6
  correct digits, 151 fewer than 10 and 190 fewer than 13.
- **The failures are far-tail quantiles.** On log-uniform draws over each family's domain,
  `F` failed at **23.1%** (21.2 to 25.1, 95% Clopper-Pearson) and `t` at **3.6%** (2.8 to
  4.6); chi-square at 0.05% (0.00 to 0.29), normal and Clopper-Pearson at 0 of 2,000 each.
  Every F or t failure is a quantile; their CDFs and survival functions passed.
- **One small defect drives most of it.** `stats.f.isf` is `ppf(1 - q)`: 551 of 778 points are
  wrong, 490 of them `inf`. A one-function change fixes 533 of those and breaks none of the
  227 correct ones.
- **Two silent t-quantile failures appear unreported.** `t.ppf(1e-200, 3)` returns half the
  true value, with `t.cdf` of the result about `8e-200`, and the issue on this region
  (scipy#25968) calls the same kind of value fine. At small df the result sticks near 1e154.
- **statsmodels 0.15.0**: 0 wrong of 1,114 grid and 2,000 random points. **JAX 0.12.0.1**: 0
  wrong on the normal, 1.2% (0.6 to 2.1) on chi-square, all with at least 9.59 digits right. **spotify-confidence
  4.1.0**: boundaries about 1e-5 low after the first look, harmless in practice. **All 16
  printed Jennison and Turnbull constants** checked (K = 2 to 5) agree to three decimals at
  the 36 looks compared.

## What was measured

Inputs are the exact binary64 values an implementation receives. A value is **certified
wrong** when its distance from the enclosure exceeds the largest tolerance over it, and
**correct** when even the farthest endpoint is within the smallest tolerance. Both are proofs
about the declared tolerance:

| Target | Relative tolerance | Absolute floor |
|---|---|---|
| SciPy, statsmodels, JAX, mpmath at 15 digits | 1e-12 | `DBL_MIN` for probabilities |
| spotify-confidence (declares none) | 1e-6 | none |
| printed constants (three decimals) | none | 5e-4 |

The tolerance is a choice: a function good to 11.9 digits is "wrong" here. The root causes
below separate those from real defects.

- **Grid** (`data/certtables-v1.csv`, 6,481 rows of 6,519 points): a declared population, so
  counts need no interval.
- **Random draws** (`data/certtables-v1-random.csv`, 9,722 rows of 10,000 points; seed
  20261030): 2,000 draws per family, parameters and tail probabilities log-uniform, `p` down
  to 1e-300. Rates are under this measure, which is deliberately heavy in the tails; they are
  not rates in typical use.
- **Left out:** 316 quantiles (38 grid, 278 random) outside `[2^-1100, 2^1100]`, each proved
  to lie there by a certified sign, and 11 random rows out of the normal double range. They
  have no scoreable finite double answer, so what targets return for them was not scored.

## Grid census

Exact counts; LRE columns count points with fewer than that many correct digits. Points with a
reference below `DBL_MIN` are scored by verdict only, so `LRE pts` can be below `scored`.

| Target | Family | Scored | Wrong | Undecided | LRE pts | LRE < 6 | LRE < 10 | LRE < 13 |
|---|---|---|---|---|---|---|---|---|
| SciPy | normal | 98 | 0 | 0 | 96 | 0 | 0 | 0 |
| SciPy | t | 1948 | 22 | 0 | 1902 | 22 | 22 | 22 |
| SciPy | chi-square | 1416 | 2 | 0 | 1353 | 0 | 0 | 5 |
| SciPy | F | 1186 | 140 | 0 | 1156 | 103 | 127 | 157 |
| SciPy | Clopper-Pearson | 522 | 0 | 0 | 522 | 0 | 0 | 0 |
| SciPy | binomial | 592 | 0 | 0 | 565 | 0 | 0 | 0 |
| SciPy | Fisher | 45 | 0 | 0 | 43 | 0 | 0 | 0 |
| SciPy | Wilcoxon | 168 | 0 | 0 | 168 | 0 | 0 | 0 |
| SciPy | Mann-Whitney | 168 | 0 | 0 | 168 | 0 | 0 | 0 |
| SciPy | Kolmogorov-Smirnov | 222 | 3 | 0 | 192 | 2 | 2 | 6 |
| statsmodels | Clopper-Pearson | 522 | 0 | 0 | 522 | 0 | 0 | 0 |
| statsmodels | binomial | 592 | 0 | 0 | 565 | 0 | 0 | 0 |
| JAX | normal | 98 | 0 | 0 | 96 | 0 | 0 | 0 |
| JAX | chi-square | 612 | 11 | 0 | 549 | 0 | 2 | 15 |
| mpmath (15 digits) | normal | 50 | 0 | 0 | 48 | 0 | 0 | 0 |
| mpmath (15 digits) | t | 984 | 64 | 0 | 946 | 0 | 26 | 88 |
| mpmath (15 digits) | chi-square | 607 | 0 | 0 | 544 | 0 | 0 | 0 |
| mpmath (15 digits) | F | 462 | 4 | 0 | 438 | 0 | 1 | 14 |
| spotify-confidence | boundaries, certified | 11 | 6 | 0 | 11 | 6 | 6 | 6 |
| printed | boundaries, certified | 14 | 0 | 0 | 14 | 14 | 14 | 14 |

The printed row's LRE columns only restate that three decimals are about 4 digits. On
uncertified K = 4, 5 references, spotify-confidence agreed at 2 points and disagreed at 7;
the printed constants agreed at all 22. mpmath also had 39 grid points that timed out or
raised `NoConvergence`.

## Random draws

| Target | Family | Scored | Wrong | Rate | 95% CI |
|---|---|---|---|---|---|
| SciPy | normal | 2000 | 0 | 0.0000 | 0.0000 to 0.0019 |
| SciPy | t | 1909 | 69 | 0.0361 | 0.0282 to 0.0456 |
| SciPy | chi-square | 1929 | 1 | 0.0005 | 0.0000 to 0.0029 |
| SciPy | F | 1873 | 433 | 0.2312 | 0.2122 to 0.2510 |
| SciPy | Clopper-Pearson | 2000 | 0 | 0.0000 | 0.0000 to 0.0019 |
| statsmodels | Clopper-Pearson | 2000 | 0 | 0.0000 | 0.0000 to 0.0019 |
| JAX | normal | 2000 | 0 | 0.0000 | 0.0000 to 0.0019 |
| JAX | chi-square | 979 | 12 | 0.0123 | 0.0063 to 0.0214 |
| mpmath (15 digits) | normal | 987 | 0 | 0.0000 | 0.0000 to 0.0038 |
| mpmath (15 digits) | t | 984 | 204 | 0.2073 | 0.1823 to 0.2341 |
| mpmath (15 digits) | chi-square | 959 | 0 | 0.0000 | 0.0000 to 0.0039 |
| mpmath (15 digits) | F | 964 | 56 | 0.0581 | 0.0441 to 0.0748 |

mpmath scores CDFs and survival functions only (81 more points timed out or raised
`NoConvergence`), so its rates are not comparable with SciPy's. Intervals are per family,
with no multiplicity adjustment.

## Root causes

All 670 SciPy points wrong over the grid and the draws fall into these groups. Drafts with
reproductions are in [docs/upstream/](docs/upstream/); none has been filed.

| Cause | Points | Share of 670 | Status upstream |
|---|---|---|---|
| `f.isf` is `ppf(1 - q)` | 551 | 82% | no dedicated issue |
| `t` quantile: stuck near `sqrt(df / DBL_MIN)` | 47 | 7% | unreported |
| `t` quantile: infinite with the wrong sign | 29 | 4% | scipy#25968; Boost fix after 1.18.1 |
| `t` quantile: finite, off by a factor of 1.6 to 2.8 | 15 | 2% | unreported |
| `f.ppf` `nan` or lost digits at tiny `p` | 22 | 3% | same inverse as scipy#26531 |
| Marginal or edge cases (chi-square 3, Kolmogorov-Smirnov 3) | 6 | 1% | no report warranted |

- **`f.isf`.** `f_gen` has no `_isf`, so it inherits `_ppf(1.0 - q)`; all 778 scored points
  equal `f.ppf(1 - q)` bit for bit. `1 - q` is exactly 1 below 1.1e-16, hence `inf`, and at
  `q = 1e-10` only 7.78 digits survive. The largest wrong `q` is 1e-5.
- **`t` quantiles.** `stats.t.cdf(stats.t.ppf(p, df), df)` is about `8 p` for df between
  about 2.05 and 4.5 and `p` below a df-dependent onset (about 1e-160 at df = 3), with no
  warning. Finite wrong values at small df stick near `sqrt(df / DBL_MIN)`, the point where
  a variable like `df / (df + x^2)` underflows, although the true quantile is in range.
- **Marginal.** The three chi-square points have LRE 11.25 to 11.97 and one Kolmogorov-Smirnov
  survival point 11.09, just under the 12-digit tolerance. The two Kolmogorov-Smirnov CDF
  points return 0 at `d` one rounding above `1/(2n)`, where the exact values are 2e-83 and
  1e-166. They count as wrong only because inputs are exact doubles: at `1/(2n)` itself the
  value is exactly 0, which is what SciPy returns, so they are not SciPy defects.

Other targets: JAX's 23 chi-square points have LRE from 9.59, at large df; spotify-confidence's
13 later-look boundaries are low by 6e-6 to 2.8e-5 relative, consistent with its fixed-step
trapezoid rule; mpmath's losses at 15 digits are mostly at large df (LRE from 6.93), and 15
digits is a working precision there, not a promise.

## Is the reference right?

Arb proves a formula's value at an exact input, so the live risk is a wrong formula or a wrong
use of Arb. The checks:

- **Closed forms and brute force** in the test suite: t with df = 1, 2; chi-square with
  df = 1, 2; F(2, 2); Clopper-Pearson at `k = 0` and `k = n`; Pocock K = 1; exact distributions
  sum to 1 exactly and match enumeration.
- **Agreement.** SciPy, which shares no code with Arb, is within 1e-12 of the enclosure on
  15,406 of 16,076 scored rows. A formula error would show up as a block of disagreements; the
  670 that remain have the causes above.
- **Independent check of the disagreements.** mpmath at 80 digits, a separate implementation
  of the same beta and gamma identities, tested the 1,016 continuous rows behind a wrong
  verdict for any target and a seeded sample of 600 others (`results/crosscheck.json`). 963 and
  584 passed and none failed; 67 ran past the 10-second limit (52 of them `f.isf` rows) and 2
  raised `NoConvergence`, so those stand on Arb alone. This catches a wrong use of Arb, not a
  wrong identity; the identities rest on the closed forms and the agreement above.
- **Printed tables.** All 16 Jennison and Turnbull constants checked agree to three decimals.
- **K = 3 against the uncertified recursion.** A slow test checks that the mpmath recursion
  behind the K = 4, 5 references lands on the certified K = 3 value to 1e-14.

## Limits

- The tolerance and the sampling measure are choices, stated above. A failure rate here is
  not a rate in everyday use: the 23% for F comes from sampling `q` far below any usual
  significance level.
- The census covers these functions at these points on one Windows x86-64 machine. SciPy
  wheels on other platforms were not run.
- `lo` and `hi` of group-sequential rows depend on the mpmath version through the starting
  point, not in validity. Other rows are independent of every target.
- The printed constants were not re-read against a physical copy of the book. Agreement with
  certified values to three decimals makes a transcription error unlikely.
- K = 4, 5 boundaries are references, not proofs, and are not in the certified census.
- R packages (gsDesign, rpact, ldbounds) were not run; the table can test them, since it is
  plain CSV.

## Reproduce

```sh
uv sync --group dev --extra spotify --extra jax
uv run certtables build --max-minutes 12    # rerun until exit 0; about 30 CPU minutes
uv run certtables verify
uv run certtables check --target scipy,statsmodels,spotify,printed,jax,mpmath   # about 8 minutes
uv run certtables report
uv run python scripts/upstream_checks.py
uv run python scripts/crosscheck_mpmath.py  # rerun to continue; 12 minutes per call
uv run pytest --runslow                     # three rebuilds, about 30 minutes in all
```
