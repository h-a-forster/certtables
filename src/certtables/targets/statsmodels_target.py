"""statsmodels adapter: ``proportion_confint(method='beta')`` and ``binom_test`` tails."""

from __future__ import annotations

import warnings

from certtables.grid import Point
from certtables.targets.base import Target


class StatsmodelsTarget(Target):
    name = "statsmodels"
    description = (
        "statsmodels.stats.proportion: proportion_confint(method='beta'), "
        "binom_test(alternative='smaller'/'larger')"
    )

    def version(self) -> str:
        import statsmodels

        return statsmodels.__version__

    def supports(self, pt: Point) -> bool:
        return pt.family in ("cp", "binom")

    def evaluate(self, pt: Point) -> float:
        from statsmodels.stats.proportion import binom_test, proportion_confint

        p = pt.p
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if pt.family == "cp":
                lo, hi = proportion_confint(
                    int(p["k"]), int(p["n"]), alpha=float(p["alpha"]), method="beta"
                )
                return float(lo if pt.function == "lower" else hi)
            k, n, prob = int(p["k"]), int(p["n"]), float(p["p"])
            if pt.function == "cdf":  # P(X <= k)
                return float(binom_test(k, n, prob, alternative="smaller"))
            return float(binom_test(k + 1, n, prob, alternative="larger"))  # P(X >= k+1)
