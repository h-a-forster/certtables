"""Implementations under test. Each adapter imports its target lazily."""

from __future__ import annotations

from certtables.targets.base import Target
from certtables.targets.jax_target import JaxTarget
from certtables.targets.mpmath_target import MpmathTarget
from certtables.targets.printed import PrintedTarget
from certtables.targets.scipy_target import SciPyTarget
from certtables.targets.spotify_target import SpotifyTarget
from certtables.targets.statsmodels_target import StatsmodelsTarget

TARGETS: dict[str, type[Target]] = {
    t.name: t
    for t in (SciPyTarget, StatsmodelsTarget, MpmathTarget, SpotifyTarget, JaxTarget, PrintedTarget)
}


def get_target(name: str) -> Target:
    if name not in TARGETS:
        raise SystemExit(f"unknown target {name!r}; choose from {sorted(TARGETS)}")
    return TARGETS[name]()
