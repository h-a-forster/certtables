# Upstream reports and outcomes

Issues filed from certtables findings. State read from GitHub on 2026-11-13.

| Project | Issue | Filed | State | Outcome |
|---|---|---|---|---|
| SciPy | [scipy#26692](https://github.com/scipy/scipy/issues/26692): `stats.f.isf` loses the upper tail | 2026-11-12 | open | labelled `defect`, `scipy.stats`; from [upstream/scipy-f-isf.md](upstream/scipy-f-isf.md) |
| Distributions.jl | [Distributions.jl#2111](https://github.com/JuliaStats/Distributions.jl/issues/2111): `KSDist` cdf missing a power in `cdf_durbin` | 2026-11-09 | closed, completed | fixed by [PR #2112](https://github.com/JuliaStats/Distributions.jl/pull/2112), merged 2026-11-09, 2 h 35 min after filing; not yet in a release (latest v0.25.134 predates it) |
| TensorFlow Probability | [tfp#2040](https://github.com/tensorflow/probability/issues/2040): Chi2, Gamma, Beta and Binomial `survival_function` fall back to `1 - cdf` | 2026-11-09 | open | no response yet |
| PyTorch | [pytorch#204127](https://github.com/pytorch/pytorch/issues/204127): `gammainc`/`gammaincc` use single-precision pi in the double asymptotic series | 2026-11-09 | open | a contributor offered on 2026-11-10 to write the fix |
| JAX | [jax#42458](https://github.com/jax-ml/jax/issues/42458): `gammainc`/`gammaincc` lose accuracy for large `a` (prefactor cancellation) | 2026-11-09 | open | a maintainer asked whether it duplicates jax#41201 and assigned the issue to themselves on 2026-11-10; no reply from us yet ([isolation script](../studies/libs-2026-11-09/scripts/repro/jax_isolate.py)) |

The last four come from the 2026-11-09 study of four more libraries
([studies/libs-2026-11-09/](../studies/libs-2026-11-09/), REPORT.md "Four more libraries").

Not filed, with the reason:

- [upstream/scipy-t-quantile.md](upstream/scipy-t-quantile.md): the two new `t.ppf` modes were not
  tested on Boost develop, which SciPy maintainers asked for after large incomplete-beta
  changes; rerun the census after SciPy's next Boost update first.
- [upstream/scipy-f-ppf.md](upstream/scipy-f-ppf.md): covered by scipy#26531, where a maintainer
  found the Boost inverse already fixed on develop.
- [upstream/spotify-confidence-bounds.md](upstream/spotify-confidence-bounds.md): real but
  harmless (about 5e-6 in tail probability).
