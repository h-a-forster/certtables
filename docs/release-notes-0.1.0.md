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

## Pre-release audit, 2026-11-06

Two independent audits, a statistics review and an engineering review, ran on this release.
Each finding was checked against the code. No table row, score or census number changed.

Fixed:

- `build`: a failing group-sequential design now records every look in `errors.jsonl`, and so in
  `MANIFEST.json`; before, only look 1 was recorded and the rest vanished. No shipped design fails.
- `verify`: a manifest missing an entry now fails instead of passing, and each declared point
  must have exactly one row or `not_enclosed` entry.
- `build --max-minutes` is one budget for the call, not one per set.
- `check` refuses an unknown target before scoring any, and its help names the targets.
- `report` read `data/` from the working directory, and outside the repository wrote empty
  census files and exited 0. It now follows the manifest, and `check`, `report`, `verify` and
  `repro` refuse to run without `data/`.
- Tests now cover the census: `tests/test_report.py` runs scoring through to the census files
  on hand-worked rows, and checks that the committed census is what the committed scores give.
- CI swaps Python versions so the Linux leg replays the recorded SciPy adapter (the lock gives
  3.11 an older SciPy) and Windows keeps the 3.11 floor; skips are now printed. CI has not run.
- Wording: the "wrong" rule in the README omitted the `DBL_MIN` floor; "36 constants" was 36
  comparisons of 16 constants; the crosscheck timeouts were 52 `f.isf` rows, not 51; the crosscheck
  was described as testing the formulas, though it evaluates the same beta and gamma identities;
  `f64` was described as depending on the stored `lo` and `hi`; the README snippet's "outside" read
  as "wrong" for correctly rounded doubles.
- Citation: O'Brien-Fleming `C_B` comes from Jennison and Turnbull's Table 2.3, not 2.1. This
  rests on Jennison's own citation of Table 2.3 for `C_B` = 2.04; the book itself was not checked.

Rejected, with reasons:

- Dropping the two Kolmogorov-Smirnov CDF points from the 3 wrong. They are an artifact of the
  exact-double input, but that convention (DESIGN 2.1) applies to every row, and an exception for
  two rows would make the count unreproducible. They stay counted and are explained beside it.
- Changing `f64` for the 4 values beyond the double range or the 2 Clopper-Pearson midpoints. The
  values are correctly rounded from the exact enclosure; the description was what was wrong.
- A self-contained wheel. The tables belong to the repository, and shipping them in the wheel adds
  nothing to a study run from a checkout; the requirement is now documented and enforced. The
  1.4 MB sdist, which carries `data/` and `results/`, stays.
- Replaying the SciPy adapter on 3.11. The recorded values are for SciPy 1.18.1, which needs 3.12;
  a skip there is correct.
