"""Write the shared point lists the library runners read.

Run in ref/venv with certtables/src on PYTHONPATH.

data/points-f64.tsv   table v1 points the Distribution-style APIs can express.
data/points-f32.tsv   the same points with every float input rounded to binary32. The table's
                      enclosures are theorems about the exact binary64 input, so they do not
                      apply to a rounded input: refs-f32.csv holds fresh Arb enclosures.
data/points-ext.tsv   beta cdf/sf/ppf/isf, not in table v1 (fresh Arb enclosures, refs-ext.csv).

Line format: set <TAB> id. Runners parse the id; floats in ids round-trip by repr.
"""

import csv
import math
import random
import struct
import sys
from pathlib import Path

from certtables.grid import Point, parse_id

csv.field_size_limit(10**7)
REPO = Path(sys.argv[1])
OUT = Path(sys.argv[2])
FAMS = {"norm", "t", "chi2", "f", "cp", "binom", "fisher", "ks"}


def f32(x: float) -> float:
    """Nearest binary32 as the exact double. inf on overflow."""
    try:
        return struct.unpack("f", struct.pack("f", x))[0]
    except OverflowError:
        return math.inf if x > 0 else -math.inf


def f32_point(pt: Point):
    """pt with float inputs rounded to binary32, or None when an input does not survive
    (underflows to 0 or subnormal, overflows, or a probability rounds to 1)."""
    out = []
    for k, v in pt.params:
        if isinstance(v, float):
            w = f32(v)
            if not math.isfinite(w):
                return None
            if v != 0.0 and abs(w) < 2.0**-126:
                return None
            if k == "p" and not 0.0 < w < 1.0:
                return None
            out.append((k, w))
        else:
            out.append((k, v))
    return Point(pt.family, pt.function, tuple(out))


def main() -> None:
    f64_lines, f32_lines = [], {}
    dropped = 0
    for label, name in (("grid", "certtables-v1.csv"), ("random", "certtables-v1-random.csv")):
        with open(REPO / "data" / name, newline="") as fh:
            for row in csv.DictReader(fh):
                if row["family"] not in FAMS:
                    continue
                pt = parse_id(row["id"])
                if row["family"] == "fisher" and pt.p["alternative"] == "two-sided":
                    continue
                f64_lines.append(f"{label}\t{row['id']}\n")
                if row["family"] in ("norm", "chi2", "t", "f"):
                    q = f32_point(pt)
                    if q is None:
                        dropped += 1
                    else:
                        f32_lines.setdefault(q.id, label)
    (OUT / "points-f64.tsv").write_text("".join(f64_lines))
    (OUT / "points-f32.tsv").write_text("".join(f"{v}\t{k}\n" for k, v in f32_lines.items()))
    print("f64", len(f64_lines), "f32", len(f32_lines), "f32 inputs not representable", dropped)

    # Beta: seeded log-uniform draws plus a small fixed grid.
    rng = random.Random("20261109:beta")
    ext = {}

    def lu(lo, hi):
        return float(f"{10 ** rng.uniform(math.log10(lo), math.log10(hi)):.6g}")

    for a, b in [(0.5, 0.5), (1.0, 3.0), (2.0, 2.0), (2.5, 3.5), (10.0, 1.0), (10.0, 10.0),
                 (50.0, 200.0), (100.0, 100.0), (1000.0, 1000.0), (1e4, 1e4), (1e5, 1e5)]:  # fmt: skip
        for x in (1e-10, 1e-3, 0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99, 0.999):
            for fn in ("cdf", "sf"):
                pt = Point("beta", fn, (("a", a), ("b", b), ("x", x)))
                ext[pt.id] = "grid"
        for p in (1e-300, 1e-100, 1e-30, 1e-10, 1e-3, 0.05, 0.5, 0.95):
            for fn in ("ppf", "isf"):
                pt = Point("beta", fn, (("a", a), ("b", b), ("p", p)))
                ext[pt.id] = "grid"
    for _ in range(1200):
        a, b = lu(0.1, 1e5), lu(0.1, 1e5)
        fn = rng.choice(("cdf", "sf", "ppf", "isf"))
        if fn in ("cdf", "sf"):
            m = a / (a + b)
            x = float(f"{min(0.999999, m * 10 ** rng.uniform(-2, 0.3)):.6g}")
            pt = Point("beta", fn, (("a", a), ("b", b), ("x", x)))
        else:
            pt = Point("beta", fn, (("a", a), ("b", b), ("p", lu(1e-300, 0.5))))
        ext.setdefault(pt.id, "random")
    (OUT / "points-ext.tsv").write_text("".join(f"{v}\t{k}\n" for k, v in ext.items()))
    print("ext", len(ext))


if __name__ == "__main__":
    main()
