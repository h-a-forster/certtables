"""Points, the declared deterministic grid and the seeded random draws
(docs/DESIGN.md section 3).

A point is (family, function, params). Its id is ``family.function(k=v;...)`` with floats
written by ``repr`` (round-trips to the same double) and ints by ``str``; ids are the
primary key of ``data/certtables-v1.csv``.
"""

from __future__ import annotations

import hashlib
import math
import random
from collections.abc import Iterator
from dataclasses import dataclass

CONTINUOUS = ("norm", "t", "chi2", "f")
EXACT = ("binom", "fisher", "wilcoxon", "mannwhitney", "ks")
FAMILIES = (*CONTINUOUS, "cp", *EXACT, "gst")
#: Families of the random-draw census (log-uniform over the parameter domain).
RANDOM_FAMILIES = (*CONTINUOUS, "cp")
#: Functions whose value is a probability (scored with an absolute floor; DESIGN 4.2).
PROBABILITY_FUNCTIONS = {"cdf", "sf", "pvalue"}


def _fmt(v: object) -> str:
    if isinstance(v, bool):
        raise TypeError("bool parameters are not allowed")
    if isinstance(v, int):
        return str(v)
    if isinstance(v, float):
        if not math.isfinite(v):
            raise ValueError(f"non-finite parameter {v!r}")
        return repr(v)
    if isinstance(v, str):
        if any(ch in v for ch in ";=()"):
            raise ValueError(f"reserved character in {v!r}")
        return v
    raise TypeError(f"unsupported parameter type {type(v).__name__}")


@dataclass(frozen=True)
class Point:
    family: str
    function: str
    params: tuple[tuple[str, int | float | str], ...]

    @property
    def id(self) -> str:
        inner = ";".join(f"{k}={_fmt(v)}" for k, v in self.params)
        return f"{self.family}.{self.function}({inner})"

    @property
    def p(self) -> dict[str, int | float | str]:
        return dict(self.params)

    @property
    def is_probability(self) -> bool:
        return self.function in PROBABILITY_FUNCTIONS


def parse_id(pid: str) -> Point:
    """Inverse of :attr:`Point.id`."""
    head, _, rest = pid.partition("(")
    family, _, function = head.partition(".")
    if not rest.endswith(")") or not family or not function:
        raise ValueError(f"malformed point id {pid!r}")
    params: list[tuple[str, int | float | str]] = []
    body = rest[:-1]
    for item in body.split(";") if body else []:
        k, _, v = item.partition("=")
        params.append((k, _parse_value(v)))
    pt = Point(family, function, tuple(params))
    if pt.id != pid:
        raise ValueError(f"point id does not round-trip: {pid!r}")
    return pt


def _parse_value(v: str) -> int | float | str:
    try:
        return int(v)
    except ValueError:
        pass
    try:
        return float(v)
    except ValueError:
        return v


# --------------------------------------------------------------------------- the grid

P_GRID = (
    1e-300, 1e-200, 1e-100, 1e-50, 1e-30, 1e-20, 1e-15, 1e-10, 1e-7, 1e-5, 1e-3, 0.01,
    0.025, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 0.95, 0.975, 0.99, 0.999,
)  # fmt: skip
NORM_X = (
    -38.0, -30.0, -20.0, -10.0, -5.0, -3.0, -2.0, -1.5, -1.0, -0.5, -0.1, -1e-5, 0.0, 1e-5,
    0.1, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0, 20.0, 30.0, 38.0,
)  # fmt: skip
T_DF = (
    0.1, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 7.5, 10.0, 20.0, 30.0, 50.0, 100.0, 1e3,
    1e4, 1e5, 1e6, 1e8,
)  # fmt: skip
T_X = (
    -1e10, -1e5, -1000.0, -100.0, -30.0, -10.0, -5.0, -3.0, -2.0, -1.0, -0.5, -0.1, 0.0,
    0.1, 0.5, 1.0, 2.0, 3.0, 5.0, 10.0, 30.0, 100.0, 1000.0, 1e5, 1e10,
)  # fmt: skip
CHI2_DF = (0.1, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 10.0, 20.0, 50.0, 100.0, 300.0, 1e3, 1e4,
           1e5, 1e6)  # fmt: skip
CHI2_REL_X = (1e-6, 1e-3, 0.01, 0.1, 0.5, 0.9, 1.0, 1.1, 1.5, 2.0, 3.0, 5.0, 10.0, 100.0)
CHI2_ABS_X = (1e-10, 1e-3, 1.0, 10.0, 100.0, 1000.0)
F_DF = (
    (1.0, 1.0), (1.0, 10.0), (2.0, 2.0), (3.0, 7.5), (5.0, 2.0), (10.0, 10.0), (10.0, 100.0),
    (0.5, 0.5), (1.0, 1e3), (100.0, 100.0), (1e3, 1e3), (1e4, 10.0), (10.0, 1e4),
    (1e5, 1e5), (2.5, 3.5),
)  # fmt: skip
F_X = (1e-10, 1e-5, 1e-3, 0.01, 0.1, 0.5, 0.8, 1.0, 1.25, 2.0, 3.0, 5.0, 10.0, 100.0, 1e4,
       1e8)  # fmt: skip
CP_N = (1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000)
CP_ALPHA = (0.1, 0.05, 0.01)
BINOM_N = (1, 5, 10, 20, 50, 100, 200, 500, 1000)
BINOM_P = (1e-3, 0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99)
FISHER_TABLES = (
    (3, 1, 1, 3), (1, 9, 11, 3), (8, 2, 1, 5), (0, 5, 5, 0), (10, 10, 10, 10), (12, 5, 3, 20),
    (100, 50, 50, 100), (500, 600, 700, 450), (1, 1000, 1000, 1), (0, 10, 10, 10),
    (2, 7, 8, 2), (20, 0, 0, 20), (50, 60, 45, 70), (5, 0, 1, 4), (300, 10, 10, 300),
)  # fmt: skip
ALTERNATIVES = ("two-sided", "less", "greater")
WILCOXON_N = (5, 10, 15, 20, 25, 30, 40, 50)
MWU_SIZES = ((3, 4), (5, 5), (5, 10), (10, 10), (10, 20), (20, 20), (25, 30), (50, 50))
KS_N = (1, 2, 3, 5, 10, 20, 50, 100)
KS_D = (0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.75, 0.9, 0.99)
#: KS points with n*d above this use Durbin matrices that are too slow; d >= 1/2 is exempt.
KS_MAX_ND = 30
#: Group-sequential designs on the grid: (kind, alpha, schedules).
GST_SCHEDULES_EQUAL = ("1/2,1", "1/3,2/3,1", "1/4,1/2,3/4,1", "1/5,2/5,3/5,4/5,1")
GST_SCHEDULES_UNEQUAL = ("1/4,1", "3/10,1", "7/10,1")
GST_DESIGNS = (
    ("pocock", 0.05), ("pocock", 0.01), ("obf", 0.05), ("obf", 0.01),
    ("ld-obf", 0.05), ("ld-pocock", 0.05), ("ld-power", 0.05),
)  # fmt: skip


def _cont(family: str, dfs: list[dict[str, float]], xs) -> Iterator[Point]:
    for params in dfs:
        items = tuple(params.items())
        xgrid = xs(params) if callable(xs) else xs
        for fn in ("cdf", "sf"):
            for x in xgrid:
                yield Point(family, fn, (*items, ("x", float(x))))
        for fn in ("ppf", "isf"):
            for p in P_GRID:
                yield Point(family, fn, (*items, ("p", float(p))))


def _chi2_x(params: dict[str, float]) -> list[float]:
    df = params["df"]
    return sorted({df * r for r in CHI2_REL_X} | set(CHI2_ABS_X))


def _k_values(n: int) -> list[int]:
    return sorted({k for k in (0, 1, 2, n // 10, n // 4, n // 2, n - 1, n) if 0 <= k <= n})


def deterministic_grid() -> list[Point]:
    """The declared grid, in a fixed order."""
    pts: list[Point] = []
    pts += _cont("norm", [{}], NORM_X)
    pts += _cont("t", [{"df": d} for d in T_DF], T_X)
    pts += _cont("chi2", [{"df": d} for d in CHI2_DF], _chi2_x)
    pts += _cont("f", [{"dfn": a, "dfd": b} for a, b in F_DF], F_X)
    for n in CP_N:
        for k in _k_values(n):
            for alpha in CP_ALPHA:
                for side in ("lower", "upper"):
                    pts.append(Point("cp", side, (("n", n), ("k", k), ("alpha", alpha))))
    for n in BINOM_N:
        for p in BINOM_P:
            for k in sorted({0, 1, round(n * p), n // 2, n - 1} - {n}):
                for fn in ("cdf", "sf"):
                    pts.append(Point("binom", fn, (("n", n), ("p", p), ("k", k))))
    for tab in FISHER_TABLES:
        table = ",".join(map(str, tab))
        for alt in ALTERNATIVES:
            pts.append(Point("fisher", "pvalue", (("table", table), ("alternative", alt))))
    for n in WILCOXON_N:
        top = n * (n + 1) // 2
        for w in sorted({0, 1, top // 4, top // 2, 3 * top // 4, top - 1, top}):
            for alt in ALTERNATIVES:
                pts.append(Point("wilcoxon", "pvalue", (("n", n), ("w", w), ("alternative", alt))))
    for m, n in MWU_SIZES:
        top = m * n
        for u in sorted({0, 1, top // 4, top // 2, 3 * top // 4, top - 1, top}):
            for alt in ALTERNATIVES:
                pts.append(
                    Point(
                        "mannwhitney",
                        "pvalue",
                        (("m", m), ("n", n), ("u", u), ("alternative", alt)),
                    )
                )
    for n in KS_N:
        for d in KS_D:
            if n * d <= KS_MAX_ND or d >= 0.5:
                for fn in ("cdf", "sf"):
                    pts.append(Point("ks", fn, (("n", n), ("d", d))))
    pts += gst_points()
    return pts


def gst_points() -> list[Point]:
    pts = []
    for kind, alpha in GST_DESIGNS:
        scheds = GST_SCHEDULES_EQUAL
        if kind.startswith("ld-"):
            scheds = GST_SCHEDULES_EQUAL + GST_SCHEDULES_UNEQUAL
        for s in scheds:
            K = s.count(",") + 1
            for look in range(1, K + 1):
                pts.append(
                    Point(
                        "gst",
                        "bound",
                        (("design", kind), ("alpha", alpha), ("t", s), ("look", look)),
                    )
                )
    return pts


def grid_hash(points: list[Point]) -> str:
    """SHA-256 of the newline-joined point ids (order-sensitive)."""
    return hashlib.sha256("\n".join(p.id for p in points).encode()).hexdigest()


# --------------------------------------------------------------------------- random draws

RANDOM_SEED = 20261030
RANDOM_PER_FAMILY = 2000


def _logu(rng: random.Random, lo: float, hi: float) -> float:
    return float(f"{10 ** rng.uniform(math.log10(lo), math.log10(hi)):.6g}")


def random_points(family: str, n: int = RANDOM_PER_FAMILY, seed: int = RANDOM_SEED) -> list[Point]:
    """``n`` seeded draws for ``family`` (DESIGN 3.2 gives the measures). Duplicates are
    dropped, so a family can have slightly fewer than ``n`` points."""
    if family not in RANDOM_FAMILIES:
        raise ValueError(f"no random draw for {family!r}")
    rng = random.Random(f"{seed}:{family}")
    out: dict[str, Point] = {}
    for _ in range(n):
        if family == "cp":
            nn = max(1, round(_logu(rng, 1, 1e4)))
            k = rng.randint(0, nn)
            alpha = _logu(rng, 1e-4, 0.2)
            side = rng.choice(("lower", "upper"))
            pt = Point("cp", side, (("n", nn), ("k", k), ("alpha", alpha)))
        else:
            fn = rng.choice(("cdf", "sf", "ppf", "isf"))
            params: tuple = ()
            if family in ("t", "chi2"):
                params = (("df", _logu(rng, 0.1, 1e6)),)
            elif family == "f":
                params = (("dfn", _logu(rng, 0.1, 1e6)), ("dfd", _logu(rng, 0.1, 1e6)))
            if fn in ("ppf", "isf"):
                last = ("p", _logu(rng, 1e-300, 0.5))
            elif family == "norm":
                last = ("x", rng.choice((-1, 1)) * _logu(rng, 1e-3, 38.0))
            elif family == "t":
                last = ("x", rng.choice((-1, 1)) * _logu(rng, 1e-3, 1e3))
            elif family == "chi2":
                last = ("x", float(f"{params[0][1] * 10 ** rng.uniform(-3, 2):.6g}"))
            else:
                last = ("x", _logu(rng, 1e-4, 1e4))
            pt = Point(family, fn, (*params, last))
        out.setdefault(pt.id, pt)
    return list(out.values())
