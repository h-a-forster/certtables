# Contributing

Bug reports, new families, new targets and documentation fixes are welcome. Read
[docs/DESIGN.md](docs/DESIGN.md) first: it fixes the input convention (exact doubles), why
the enclosures are proofs, the scoring rules and what is out of scope.

The most valuable report is a wrong reference value. Open an issue with the row `id`, the
value you believe is right and how you computed it. A certified row that is wrong means a
bug in Arb, python-flint or this package's use of them, and is treated as a release blocker.

## Setup

You need [uv](https://docs.astral.sh/uv/) and git. From a clone:

```sh
uv sync --group dev                # add --extra spotify or --extra jax for those targets
uv run certtables --version
```

## Tests and lint

```sh
uv run pytest              # fast suite, offline (a few minutes)
uv run pytest --runslow    # plus certified K = 3 boundaries and full grid rebuilds
uv run ruff check .
uv run ruff format --check .
uv run certtables verify   # row hashes and MANIFEST.json
```

CI runs lint, the fast suite and `certtables verify` on Ubuntu (Python 3.11) and Windows
(Python 3.13). Tests must run offline, write only under pytest's `tmp_path` and pass on
Windows. A bug fix comes with a test that fails without it.

## Adding a family

1. Write the enclosure in `src/certtables/enclose/`: either Arb balls from rigorous Arb
   functions (`continuous.py`), a certified sign change through
   `arbutil.certified_root` for anything defined as the root of a monotone function, or
   exact rationals (`exact.py`). If an Arb function returns non-finite or wide balls in some
   regime, replace it there by a series with an explicit remainder bound, as
   `continuous.gamma_p_series` does, and say so in DESIGN 2.2.
2. Dispatch it in `enclose/__init__.py:enclose_point`.
3. Add its points to `grid.deterministic_grid` (and `random_points` if it has a continuous
   parameter domain). Changing the grid changes the grid hash; bump the table version
   (`certtables-v2.csv`) rather than editing v1 rows in place.
4. Tests: the enclosure must contain closed-form values where they exist, exact
   distributions must sum to 1 exactly and match brute-force enumeration for small sizes.
5. `uv run certtables build`, then `certtables verify`.

## Adding a target

1. Subclass `targets.base.Target` in `src/certtables/targets/`: `supports`, `evaluate`
   (import the library inside the method), `version`, and `tolerance` if the default
   (relative 1e-12, absolute `DBL_MIN` for probabilities) is not what the library declares.
   Document the declared tolerance in DESIGN 4.2.
2. Register it in `targets/__init__.py` and add an optional extra in `pyproject.toml`.
3. Add points to `scripts/record_adapters.py`, run it, and make sure any constructed input
   carries the intended statistic (see `tests/test_targets.py`), so a disagreement is the
   target's and not the adapter's.
4. `uv run certtables check --target <name>` and `uv run certtables report`.

## Code

- Python 3.11+, typed. Runtime dependencies: python-flint (pinned) and mpmath. Targets are
  optional extras.
- Validate inputs before computing and fail with a clear error; never return a reference
  that is not an enclosure. Points that cannot be enclosed are recorded, not dropped.
- Long builds are checkpointed per family (`table.build`); keep it that way.
- Line length 100; ruff enforces the rest. Write text files as UTF-8.
