# certtables 0.1.0 release notes

First release. Numbers come from [REPORT.md](../REPORT.md) and trace to `results/` and `data/`.

## What it does

certtables publishes interval-certified reference values for statistical functions and scores
implementations against them. Python 3.11+, python-flint 0.8.0 and mpmath; no model or network
calls.

## In this release

- `data/certtables-v1.csv` (6,481 rows) and `data/certtables-v1-random.csv` (9,722 rows): normal,
  t, chi-square and F CDFs, survival functions and quantiles; Clopper-Pearson bounds to
  n = 10^4; exact binomial, Fisher, Wilcoxon, Mann-Whitney and Kolmogorov-Smirnov results;
  Pocock, O'Brien-Fleming and Lan-DeMets boundaries, certified for K <= 3 by nested
  `acb.integral` and referenced (not certified) for K = 4, 5.
- Targets: SciPy, statsmodels, mpmath at 15 digits, spotify-confidence, JAX and printed
  constants from Jennison and Turnbull.
- CLI: `build` (checkpointed, resumable), `check`, `report`, `verify`, `repro`, `env`.
- Scripts: `upstream_checks.py` (root causes), `crosscheck_mpmath.py` (independent check).
- Drafts for maintainers in `docs/upstream/`, not filed.

## Findings

SciPy 1.18.1: 167 of 6,365 grid points certified wrong; random-draw failure rates of 23.1% for
F and 3.6% for t, all in far-tail quantiles; none for normal functions, Clopper-Pearson bounds
or exact tests. `stats.f.isf` accounts for 551 of the 670 wrong points. Two silent
`t` quantile failures appear unreported.

## Known limits

Windows x86-64 only for the census. K = 4, 5 boundaries are not certified. Printed constants
were not re-read against a physical copy. mpmath's `error` count depends slightly on machine
speed (2-second limit per point).
