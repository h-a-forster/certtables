"""JAX (``jax.scipy.stats``, CPU, float64) adapter; optional extra ``certtables[jax]``."""

from __future__ import annotations

from certtables.grid import Point
from certtables.targets.base import Target

_FUNCS = {
    ("norm", "cdf"),
    ("norm", "sf"),
    ("norm", "ppf"),
    ("norm", "isf"),
    ("chi2", "cdf"),
    ("chi2", "sf"),
}


class JaxTarget(Target):
    name = "jax"
    description = "jax.scipy.stats norm.cdf/sf/ppf/isf, chi2.cdf/sf (x64 enabled, CPU)"

    def version(self) -> str:
        import jax

        return jax.__version__

    def supports(self, pt: Point) -> bool:
        return (pt.family, pt.function) in _FUNCS

    def evaluate(self, pt: Point) -> float:
        import jax

        jax.config.update("jax_enable_x64", True)
        from jax.scipy import stats

        p = pt.p
        dist = getattr(stats, pt.family)
        arg = float(p["x"] if pt.function in ("cdf", "sf") else p["p"])
        shape = [float(p["df"])] if "df" in p else []
        return float(getattr(dist, pt.function)(arg, *shape))
