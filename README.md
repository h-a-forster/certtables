# certtables

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](pyproject.toml)

How many digits of your p-value, quantile or stopping boundary are right?

certtables publishes reference values for statistical functions as intervals `[lo, hi]`
that provably contain the true value, and scores implementations against them. The tables
hold 16,203 rows: normal, t, chi-square and F CDFs, survival functions and quantiles into
deep tails (values down to 1e-300 and beyond), Clopper-Pearson bounds up to n = 10^4,
binomial, Fisher, Wilcoxon, Mann-Whitney and Kolmogorov-Smirnov exact tests, and
group-sequential boundaries.

Python 3.11+, depends on python-flint (Arb) and mpmath. No model calls, no network.

## Findings

From [REPORT.md](REPORT.md), with SciPy 1.18.1, statsmodels 0.15.0, JAX 0.12.0.1 and
spotify-confidence 4.1.0. "Wrong" means certified: the value lies outside the enclosure by
more than a relative 1e-12 (1e-6 for spotify-confidence).

| Target | Family | Grid: wrong / scored | Random draws: failure rate (95% CI) |
|---|---|---|---|
| SciPy | normal | 0 / 98 | 0.00% (0.00 to 0.19) |
| SciPy | t | 22 / 1948 | 3.61% (2.82 to 4.56) |
| SciPy | chi-square | 2 / 1416 | 0.05% (0.00 to 0.29) |
| SciPy | F | 140 / 1186 | 23.12% (21.22 to 25.10) |
| SciPy | Clopper-Pearson | 0 / 522 | 0.00% (0.00 to 0.19) |
| SciPy | binomial, Fisher, Wilcoxon, Mann-Whitney | 0 / 973 | not drawn |
| SciPy | Kolmogorov-Smirnov | 3 / 222 | not drawn |
| statsmodels | Clopper-Pearson, binomial | 0 / 1114 | 0.00% (0.00 to 0.19) |
| JAX | normal | 0 / 98 | 0.00% (0.00 to 0.19) |
| JAX | chi-square | 11 / 612 | 1.23% (0.63 to 2.14) |
| spotify-confidence | certified boundaries | 6 / 11 | not drawn |

- **The stack is right where most people use it.** SciPy's normal functions, exact tests and
  Clopper-Pearson bounds, and statsmodels, had no certified failure. Nothing was undecided.
- **`scipy.stats.f.isf` is the main defect.** It is `ppf(1 - q)`, so it returns `inf` below
  q = 1.1e-16 and loses digits well above that: 551 of 778 points were wrong, and a
  one-function fix corrects 533 of them without breaking any correct point.
- **Student t quantiles fail silently in the far tail.** Besides the known wrong-signed `inf`
  (scipy#25968), SciPy returns finite values off by a factor of 1.6 to 2.8 for df between
  about 2 and 4.5 (for example `t.ppf(1e-200, 3)`, half the true value), and values stuck near
  1e154 at small df. These two modes appear unreported.
- **Printed constants hold up.** All 36 Jennison and Turnbull Table 2.1 values checked agree
  to their three decimals (14 against certified rows, 22 against references).

These are rates under a stated log-uniform measure that reaches far into the tails, not
rates in "typical use". Drafts for maintainers are in [docs/upstream/](docs/upstream/);
nothing has been filed.

## What is certified and what is only referenced

| Kind | Rows | Meaning |
|---|---|---|
| `exact` | 1,414 | a rational computed with integers, no rounding; stored rounded outward to 25 digits with a hash of the exact value |
| `certified` | 14,726 | an Arb enclosure: a theorem about the formula at the exact binary64 input, or a certified sign change for quantiles and boundaries |
| `reference` | 63 | K = 4, 5 group-sequential boundaries from mpmath, NOT certified; scored as agree or disagree only |

Arb proves the value of a formula. That the formula is the right one rests on tests against
closed forms and brute force, and on a cross-check with mpmath at 80 digits (see
[REPORT.md](REPORT.md)). 316 quantiles lie outside `[2^-1100, 2^1100]`: each is proved to be
there, listed in `data/MANIFEST.json`, and has no row.

## Test your own implementation

`data/certtables-v1.csv` (grid) and `data/certtables-v1-random.csv` (random draws) are plain
CSV. `lo` and `hi` are decimal strings rounded outward. Inputs are exact doubles: `p=0.975`
means the double nearest 0.975. Parameters are in the `id`, for example
`t.ppf(df=2.5;p=1e-10)`. Rows of kind `reference` are not proofs.

```python
import csv
from fractions import Fraction

from scipy import stats  # your implementation here

with open("data/certtables-v1.csv", newline="") as fh:
    rows = [r for r in csv.DictReader(fh) if r["id"].startswith("norm.isf(")]
for r in rows:
    p = float(r["id"].split("p=")[1].rstrip(")"))
    v = stats.norm.isf(p)
    lo, hi = Fraction(r["lo"]), Fraction(r["hi"])
    print(r["id"], v, "inside" if lo <= Fraction(v) <= hi else float(Fraction(v) / lo - 1))
```

The `f64` column gives the correctly rounded double when `lo` and `hi` round to the same one.
To score a Python library with the full census, add a target
([CONTRIBUTING.md](CONTRIBUTING.md)).

## Install and run

```sh
git clone https://github.com/h-a-forster/certtables
cd certtables
uv sync --group dev --extra spotify      # add --extra jax for that target
uv run certtables verify                 # row hashes, file hashes, grid hash
uv run certtables check --target scipy   # also statsmodels, mpmath, spotify, jax, printed
uv run certtables report                 # results/census*.csv, results/census.md, wrong.csv
uv run certtables repro 'f.isf(dfn=10.0;dfd=10.0;p=1e-20)' --target scipy
uv run certtables build --max-minutes 12 # rebuild; rerun to resume, about 30 minutes in all
```

## Documentation

- [REPORT.md](REPORT.md): the census, root causes and limits.
- [docs/DESIGN.md](docs/DESIGN.md): why the enclosures are proofs, scoring, scope, prior work.
- [docs/upstream/](docs/upstream/): drafts for maintainers, not filed.
- [results/census.md](results/census.md): every target x family.
- [docs/release-notes-0.1.0.md](docs/release-notes-0.1.0.md).

## Limits

Python targets only. Bugs in Arb itself would break the proofs; none is known in the functions
used, and two regimes where python-flint returned useless balls are replaced or guarded
([DESIGN 2.2](docs/DESIGN.md)). Simulation-defined tables (Model Confidence Set, Acerbi-Szekely)
are out of scope: there is nothing to enclose.

## License

MIT. See [LICENSE](LICENSE).
