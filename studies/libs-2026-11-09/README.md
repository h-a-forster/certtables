# Four more libraries: PyTorch, JAX, TFP, Distributions.jl (2026-11-09)

The certtables table v1 applied to four libraries outside the Python stats stack. Results are
summarised in [REPORT.md](../../REPORT.md) ("Four more libraries"); what was filed upstream and
what became of it is in [docs/upstream.md](../../docs/upstream.md).

## Versions

| Library | Version | Environment |
|---|---|---|
| PyTorch | 2.15.0+cpu | [torch/requirements.lock](torch/requirements.lock) |
| JAX | 0.12.0.1, CPU | [jax/requirements.lock](jax/requirements.lock) |
| TensorFlow Probability | 0.25.0 with tensorflow-cpu 2.21.0 | [tfp/requirements.lock](tfp/requirements.lock) |
| Distributions.jl | 0.25.134, StatsFuns 2.2.1, SpecialFunctions 2.9.1, Julia 1.13.3 | [distributionsjl/proj/](distributionsjl/proj/) (Manifest and the Julia zip's SHA-256) |
| reference and scoring | certtables at the commit that added this folder | [ref/requirements.lock](ref/requirements.lock) plus `src/` on `PYTHONPATH` |

One Windows x86-64 machine, CPU builds, scalar calls, about an hour of wall time.

## What is committed

- [data/census.md](data/census.md): per library, precision, function and API: points scored,
  wrong, errors, out of range, and the random-draw rate with a 95% Clopper-Pearson interval;
  then the root-cause classes.
- [data/claims.csv](data/claims.csv): every wrong value, with the version, exact input,
  returned value, 25-digit enclosure, ulps, LRE and root-cause class.
- `scripts/`: the scripts as they ran; `scripts/repro/` holds the small repros and root-cause
  checks behind the upstream reports (`jax_isolate.py` splits the JAX error into prefactor
  and continued fraction, the point of jax#42458).

The intermediate files (point lists, fresh Arb enclosures, raw library output, scored files;
about 12 MB) are not committed. They are rebuilt by the steps below into `data/`, which is
ignored except for the two files above.

## Rebuilding

Run from this folder. Each library gets its own virtual environment built from its lock
(`uv venv <lib>/.venv && uv pip install -r <lib>/requirements.lock`); the reference
environment also needs the repository's `src/` on `PYTHONPATH`.

```sh
# reference environment
python scripts/make_points.py ../.. data          # points-f64, points-f32, points-ext
python scripts/make_ks_sweep.py                   # points-ks (KS Durbin branch)
python scripts/make_refs.py data/points-f32.tsv data/refs-f32-w0.csv 0 2   # and worker 1; same for ext, ks
# each library environment; appends, skips finished points, marks a killed point HANG
python scripts/run_lib.py torch f64 data/points-f64.tsv data/torch-f64.tsv  # torch/jax/tfp x f64/f32/ext
bash scripts/run_julia_watch.sh data/points-f64.tsv data/distributionsjl-f64.tsv logs/julia-f64.log 180 5
# reference environment again
python scripts/score.py torch f64                 # writes data/<lib>-<mode>-scored.csv
python scripts/consolidate.py                     # data/claims.csv and data/census.md
```

`run_julia_watch.sh` expects Julia 1.13.3 unpacked under `distributionsjl/jl/` and uses
`distributionsjl/depot` as its depot; check the zip against `julia-1.13.3.sha256`.
`scripts/repro/torch_igam_pi.py` reads `aten/src/ATen/native/Math.h` from PyTorch's v2.15.0 tag.

## Limits

Single precision is scored against fresh Arb enclosures at the exact binary32 input, with the
same number of ulps as the double rule (relative 5.4e-4); table v1 itself says nothing about
binary32. The beta rows are an extension outside table v1, with 55 large-shape points skipped
and about 97 unfinished. The t and F root-cause tests use local copies of the formulas, not
patched packages.
