# Upstream drafts

Drafts only. Nothing here has been filed; file one only with the owner's go-ahead. Each
draft was checked against the project's tracker on 2026-11-06 (GitHub search and the issues
named in it). Counts come from `results/upstream-checks.json`, `results/wrong.csv` and the
files in `results/targets/`.

`uv run certtables repro '<id>' --target <name>` reproduces any row of `results/wrong.csv`:
it evaluates the target and prints the enclosure, the verdict and the LRE bounds.

| Draft | Target | Points | State upstream |
|---|---|---|---|
| [scipy-f-isf.md](scipy-f-isf.md) | `scipy.stats.f.isf` | 551 | no dedicated issue; `f` is listed without `_isf` in the tracking issue scipy#17832 |
| [scipy-t-quantile.md](scipy-t-quantile.md) | `stats.t.ppf`, `stats.t.isf` | 91 | one of three failure modes known (scipy#25968); the other two are new |
| [scipy-f-ppf.md](scipy-f-ppf.md) | `stats.f.ppf` | 22 | same Boost inverse as scipy#26531; not reported for `f.ppf` |
| [spotify-confidence-bounds.md](spotify-confidence-bounds.md) | `sequential_bounds` | 13 | no issue found |

## Certified wrong, no draft

These are wrong by the declared tolerance, but not worth a report.

- **`chi2.sf` (2 points), `chi2.ppf` (1 point), SciPy.** LRE 11.25 to 11.97 at df from 6,192
  to 1e6, against a tolerance of 12 digits; consistent with evaluating a power and an
  exponential with a large exponent in double precision.
- **`kstwo.sf`, SciPy, 1 point** (`n = 100`, `d = 0.2`): LRE 11.09.
- **`kstwo.cdf`, SciPy, 2 points** (`n = 5, d = 0.1`; `n = 10, d = 0.05`). SciPy returns 0;
  the exact values are 2.02e-83 and 1.01e-166. The double `d` lies just above `1/(2n)`, the
  edge of the support, while `n * d` rounds to exactly 1/2. A rounding of the input's
  position, not a defect in any practical sense.
- **`jax.scipy.stats.chi2`, 23 points**: LRE down to 9.59 for large df, from
  double-precision `gammainc`.
- **mpmath at `dps = 15`, 328 points**: 15 digits is the working precision, not a promise on
  the result of `betainc`, and the losses (LRE down to 6.93) are mostly at large df. 120 more
  points ran past the 2-second limit or raised `NoConvergence`. Raise `mp.dps` before relying
  on it there.

## Already fixed upstream

`jax.scipy.stats.norm.isf` returned `inf` for small `q` through 0.11.2 (jax#41262, computed
as `ppf(1 - q)`, the same mistake as `f.isf`); the census runs JAX 0.12.0.1, where it is
fixed.
