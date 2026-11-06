"""SciPy (``scipy.stats``) adapter."""

from __future__ import annotations

import warnings

from certtables.grid import Point
from certtables.targets.base import Target, mannwhitney_samples, wilcoxon_differences

_CONT = {"norm", "t", "chi2", "f"}


class SciPyTarget(Target):
    name = "scipy"
    description = (
        "scipy.stats distributions, binomtest CI, fisher_exact, wilcoxon, mannwhitneyu, kstwo"
    )

    def version(self) -> str:
        import scipy

        return scipy.__version__

    def supports(self, pt: Point) -> bool:
        return pt.family in _CONT | {"cp", "binom", "fisher", "wilcoxon", "mannwhitney", "ks"}

    def evaluate(self, pt: Point) -> float:
        from scipy import stats

        p = pt.p
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if pt.family in _CONT:
                dist = getattr(stats, pt.family)
                shape = [float(p[k]) for k in ("df", "dfn", "dfd") if k in p]
                arg = float(p["x"] if pt.function in ("cdf", "sf") else p["p"])
                return float(getattr(dist, pt.function)(arg, *shape))
            if pt.family == "cp":
                n, k, alpha = int(p["n"]), int(p["k"]), float(p["alpha"])
                ci = stats.binomtest(k, n).proportion_ci(confidence_level=1 - alpha, method="exact")
                return float(ci.low if pt.function == "lower" else ci.high)
            if pt.family == "binom":
                f = stats.binom.cdf if pt.function == "cdf" else stats.binom.sf
                return float(f(int(p["k"]), int(p["n"]), float(p["p"])))
            if pt.family == "fisher":
                a, b, c, d = (int(v) for v in str(p["table"]).split(","))
                res = stats.fisher_exact([[a, b], [c, d]], alternative=str(p["alternative"]))
                return float(res.pvalue)
            if pt.family == "wilcoxon":
                d = wilcoxon_differences(int(p["n"]), int(p["w"]))
                res = stats.wilcoxon(d, alternative=str(p["alternative"]), method="exact")
                return float(res.pvalue)
            if pt.family == "mannwhitney":
                x, y = mannwhitney_samples(int(p["m"]), int(p["n"]), int(p["u"]))
                res = stats.mannwhitneyu(x, y, alternative=str(p["alternative"]), method="exact")
                return float(res.pvalue)
            if pt.family == "ks":
                f = stats.kstwo.cdf if pt.function == "cdf" else stats.kstwo.sf
                return float(f(float(p["d"]), int(p["n"])))
        raise ValueError(f"unsupported point {pt.id}")
