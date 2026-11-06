"""spotify-confidence 4.1.0: ``sequential_bounds`` (Lan-DeMets power family, rho = 2).

``sequential_bounds(t, alpha, sides=2)`` calls ``bounds(t, alpha, rho=2, ztrun=8,
sides=2, max_nints=1000)``, a port of R ldbounds that spends ``alpha/2 * t**2`` per side
and stops its search at a probability tolerance of 1e-7 (same mapping as peekaudit's
adapter). It declares no accuracy, so it is scored against relative 1e-6.
"""

from __future__ import annotations

import warnings
from fractions import Fraction

from certtables.grid import Point
from certtables.targets.base import Target

REL_GST = 1e-6


class SpotifyTarget(Target):
    name = "spotify"
    description = "spotify-confidence sequential_bounds (power family rho = 2, two-sided)"

    def version(self) -> str:
        from importlib.metadata import version

        return version("spotify-confidence")

    def supports(self, pt: Point) -> bool:
        return pt.family == "gst" and pt.p["design"] == "ld-power"

    def tolerance(self, pt: Point) -> tuple[float, float]:
        return REL_GST, 0.0

    def evaluate(self, pt: Point) -> float:
        return float(_bounds(str(pt.p["t"]), float(pt.p["alpha"]))[int(pt.p["look"]) - 1])


_CACHE: dict[tuple[str, float], list[float]] = {}


def _bounds(sched: str, alpha: float) -> list[float]:
    key = (sched, alpha)
    if key not in _CACHE:
        import numpy as np
        from spotify_confidence.analysis.frequentist.confidence_computers.z_test_computer import (
            sequential_bounds,
        )

        t = np.array([float(Fraction(v)) for v in sched.split(",")], dtype=np.float64)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)  # pandas deprecation inside it
            res = sequential_bounds(t, alpha=alpha, sides=2)
        zb = res.df["zb"].to_numpy(dtype=np.float64)
        _CACHE[key] = [float(v) for v in zb]
    return _CACHE[key]
