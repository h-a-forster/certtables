# AGENTS.md

Guide for coding agents (and humans) working on certtables. Read `docs/DESIGN.md` before
changing any enclosure, grid, tolerance or scoring rule: it holds the input convention, the
proofs, the estimands and the scope this file points to.

## What this is

A Python package, CLI and study. It computes interval-certified reference values (Arb ball
arithmetic via python-flint) and exact rationals for statistical CDFs, survival functions,
quantiles, Clopper-Pearson bounds, exact-test null distributions and group-sequential
boundaries, publishes them as `data/certtables-v1.csv`, and scores SciPy, statsmodels,
mpmath (15 digits), spotify-confidence, JAX and printed constants against them.

## Conventions

- uv + hatchling, `src/` layout, Python >= 3.11 (local dev 3.13 via `.python-version`),
  typed code, `ruff check` + `ruff format` (line length 100), pytest. `[tool.uv]
  exclude-newer` keeps the lock on releases at least two weeks old; move it deliberately.
- Runtime dependencies: python-flint (pinned, 0.8.0) and mpmath. Targets are extras
  (`targets`, `spotify`, `jax`) and import their library inside `evaluate`.
- Inputs are exact doubles (DESIGN 2.1). A certified row must be an enclosure: never store
  a midpoint estimate as `certified`; uncertified values are `kind = reference`. Nothing in
  `enclose/` may read a target: the table must not depend on an implementation under test.
- Every enclosure is either an Arb ball from a rigorous Arb function, a certified sign
  change (`arbutil.certified_root`), or an exact rational. When an Arb function misbehaves
  in a regime, replace it there by a series with an explicit remainder bound and document it
  in DESIGN 2.2 (as for the chi-square lower tail).
- Long jobs: `certtables build` checkpoints per family under `runs/build/<set>/` (append +
  fsync per row, skip finished ids, `progress.log`, `errors.jsonl`) and takes
  `--max-minutes`; keep each call under about 12 minutes and rerun to resume. A full build
  takes about 30 CPU minutes (grid and random draws can run side by side). `uv run pytest
  --runslow` runs three rebuilds (the longest took about 19 minutes with all three side by
  side); select one with `-k` to keep a job short.
- Changing the grid changes its hash: publish a new table version rather than editing v1.
- No model or network calls in the package. CPU only.
- Scratch files go in `tmp/` (gitignored). Do not create `.bak` copies; use git.
- The repository is public; CI runs on pushes to main, pull requests and manual dispatch.
  Commits and PRs carry no AI attribution trailers.
- Release hygiene: no personal names, local user paths or machine details in committed
  files. Upstream drafts in `docs/upstream/` are drafts only; never file them without the
  owner's go-ahead.
- README and REPORT tables and summary figures are copies of `results/published.md`
  (`certtables report` writes it); change `published.py`, rerun, and copy, never type a number.
  `tests/test_published.py` and CI hold the documents to it.
- Honesty: "certified" only for Arb/exact rows; K = 4, 5 boundaries and mpmath values are
  references. A target is "certified wrong" only against its declared tolerance; say which.

## Architecture map

| Path | Role |
|---|---|
| `src/certtables/enclose/arbutil.py` | exact Arb/rational/decimal conversions, `certified_root` |
| `src/certtables/enclose/continuous.py` | normal, t, chi-square, F, beta CDF/SF/quantiles; Clopper-Pearson; gamma series |
| `src/certtables/enclose/exact.py` | binomial, Fisher, Wilcoxon, Mann-Whitney, Kolmogorov-Smirnov (exact rationals) |
| `src/certtables/enclose/gst.py` | group-sequential boundaries: certified K <= 3 (`acb.integral`), mpmath references |
| `src/certtables/grid.py` | point ids, deterministic grid, seeded random draws, grid hash |
| `src/certtables/table.py` | CSV rows, hashes, checkpointed build |
| `src/certtables/targets/` | adapters: scipy, statsmodels, mpmath, spotify, jax, printed |
| `src/certtables/lre.py` | strdcheck's LRE module, vendored verbatim |
| `src/certtables/study.py` | LRE bounds, verdicts, `check_target` |
| `src/certtables/report.py` | census tables (`results/census*.csv`, `census.md`, `wrong.csv`) |
| `src/certtables/published.py` | `results/published.md`: every README/REPORT results table and the figures their prose quotes |
| `src/certtables/cli.py` | `build / check / report / verify / repro / env` |
| `scripts/record_adapters.py` | regenerate `tests/data/recorded.json` |
| `scripts/upstream_checks.py` | root-cause counts behind `docs/upstream/` |
| `scripts/crosscheck_mpmath.py` | independent check of rows with mpmath at 80 digits |
| `data/` | the published tables and `MANIFEST.json` |
| `results/` | per-target scores and census (committed) |
| `docs/upstream/` | repro drafts for certified-wrong points; `docs/upstream.md` lists what was filed and its outcome |
| `studies/libs-2026-11-09/` | PyTorch, JAX, TFP, Distributions.jl census: scripts as they ran, locks, `census.md`, `claims.csv` |

## How to test

```sh
uv sync --group dev --extra spotify --extra jax
uv run pytest                    # fast suite; --runslow adds K = 3 and the rebuilds
uv run ruff check . && uv run ruff format --check . && uv lock --check
uv run certtables verify
uv run certtables check --target scipy,statsmodels,spotify,printed,jax,mpmath
uv run certtables report
```

## Status and next steps

Updated: 2026-11-06. Version 0.1.0: tables built, six targets scored, `REPORT.md` written,
upstream drafts in `docs/upstream/`, filed reports in `docs/upstream.md`. Open work is in `backlog.md`.

Rerun `check`, `report` and `scripts/upstream_checks.py` when SciPy ships a newer Boost: the
`t` and `f.ppf` findings should change. The lock pins SciPy 1.18.1 and JAX 0.12.0.1.
