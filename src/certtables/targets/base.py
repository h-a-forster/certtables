"""Target adapter interface and tolerances (docs/DESIGN.md section 4).

A target evaluates a point and returns a float. ``supports`` says which points it covers;
``tolerance`` is the declared accuracy it is scored against: ``(rel, abs)`` meaning the
target is acceptable at a true value ``c`` when ``|v - c| <= max(rel * |c|, abs)``.
"""

from __future__ import annotations

import sys
from abc import ABC, abstractmethod

from certtables.grid import Point

#: Default for double-precision functions.
REL_DOUBLE = 1e-12
#: Absolute floor for probability-valued outputs: below the smallest normal double, values
#: are subnormal and lose relative precision legitimately (gradual underflow).
ABS_PROBABILITY = sys.float_info.min


class Target(ABC):
    name: str = ""
    description: str = ""

    @abstractmethod
    def supports(self, pt: Point) -> bool: ...

    @abstractmethod
    def evaluate(self, pt: Point) -> float: ...

    def tolerance(self, pt: Point) -> tuple[float, float]:
        return REL_DOUBLE, (ABS_PROBABILITY if pt.is_probability else 0.0)

    def version(self) -> str:
        return ""


def wilcoxon_differences(n: int, w: int) -> list[float]:
    """Differences with |d| = 1..n whose positive ranks sum to ``w`` (greedy from the top;
    every w in 0..n(n+1)/2 is reachable)."""
    top = n * (n + 1) // 2
    if not 0 <= w <= top:
        raise ValueError(f"w={w} outside 0..{top}")
    signs = [-1] * (n + 1)
    rem = w
    for r in range(n, 0, -1):
        if r <= rem:
            signs[r] = 1
            rem -= r
    if rem:
        raise AssertionError("greedy subset sum failed")  # pragma: no cover
    return [float(signs[r] * r) for r in range(1, n + 1)]


def mannwhitney_samples(m: int, n: int, u: int) -> tuple[list[float], list[float]]:
    """Samples x (size m) and y (size n), no ties, with U = #{x_i > y_j} = u."""
    if not 0 <= u <= m * n:
        raise ValueError(f"u={u} outside 0..{m * n}")
    y = [float(j + 1) for j in range(n)]  # 1..n
    x = []
    rem = u
    for i in range(m):
        r = min(n, rem)  # number of y values below x_i
        rem -= r
        x.append(r + 0.5 + i * 1e-3)
    return x, y
