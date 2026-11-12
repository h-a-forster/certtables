"""Score runner output against certified rows, with certtables' own verdict and LRE code.

usage: score.py <lib> <f64|f32|ext> [<api-sets>]  (paths fixed under this study folder)
Writes data/<lib>-<mode>-scored.csv and prints the census per api, split grid and random.

Binary64 uses the table v1 rules unchanged (rel 1e-12, DBL_MIN floor for probabilities). Binary32
uses the same number of ulps: rel 1e-12 * 2^29 = 5.4e-4, FLT_MIN floor for probabilities, and
references outside the normal binary32 range are out-of-range. Rows are never read for anything
but lo and hi. ulps = distance from the value to the nearest end of the enclosure, in units of
last place of the reference in the target's format.
"""

import csv
import math
import sys
from collections import defaultdict
from fractions import Fraction
from pathlib import Path

from certtables.grid import parse_id
from certtables.study import lre_bounds, verdict

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
REPO = ROOT.parents[1]  # the certtables repository
csv.field_size_limit(10**7)

FMT = {
    "f64": dict(rel=1e-12, floor=sys.float_info.min, prec=53, emin=-1022, emax=1024),
    "f32": dict(rel=1e-12 * 2**29, floor=2.0**-126, prec=24, emin=-126, emax=128),
}


def sf(x: Fraction) -> str:
    """Decimal for the csv, for a bound beyond the double range too."""
    try:
        return f"{float(x):.17g}"
    except OverflowError:
        return f"{float(x / 2**1100):.17g}*2^1100"


def ulp(c: Fraction, prec: int, emin: int) -> Fraction:
    """Unit in the last place at |c| for the format."""
    c = abs(c)
    if c == 0:
        return Fraction(2) ** (emin - prec + 1)
    e = c.numerator.bit_length() - c.denominator.bit_length()
    if Fraction(2) ** e > c:
        e -= 1
    elif Fraction(2) ** (e + 1) <= c:
        e += 1
    return Fraction(2) ** (max(e, emin) - prec + 1)


def miss_ulps(v: float, lo: Fraction, hi: Fraction, prec: int, emin: int) -> float:
    if not math.isfinite(v):
        return math.inf
    q = Fraction(v)
    if lo <= q <= hi:
        return 0.0
    near, end = (lo - q, lo) if q < lo else (q - hi, hi)
    r = near / ulp(end, prec, emin)
    try:
        return float(r)
    except OverflowError:
        return math.inf


def load_refs(mode: str) -> dict[str, tuple[Fraction, Fraction, str]]:
    refs: dict[str, tuple[Fraction, Fraction, str]] = {}
    if mode == "f64":
        for name in ("certtables-v1.csv", "certtables-v1-random.csv"):
            with open(REPO / "data" / name, newline="") as fh:
                for r in csv.DictReader(fh):
                    refs[r["id"]] = (Fraction(r["lo"]), Fraction(r["hi"]), r["kind"])
        return refs
    pat = f"refs-{mode}-w*.csv"
    for path in DATA.glob(pat):
        with open(path, newline="") as fh:
            for r in csv.reader(fh):
                refs[r[0]] = (Fraction(r[3]), Fraction(r[4]), r[1])
    return refs


def main() -> None:
    lib, mode = sys.argv[1], sys.argv[2]
    fmt = FMT["f32" if mode == "f32" else "f64"]
    points = f"points-{mode}.tsv"
    sets = {}
    for ln in open(DATA / points):
        s, pid = ln.rstrip("\n").split("\t")
        sets[pid] = s
    refs = load_refs(mode)
    in_file = DATA / f"{lib}-{mode}.tsv"
    out = DATA / f"{lib}-{mode}-scored.csv"
    census: dict[tuple, dict] = defaultdict(lambda: defaultdict(int))
    dmax, dmin = Fraction(2) ** fmt["emax"], Fraction(2) ** fmt["emin"]
    with open(in_file, encoding="utf-8") as fh, open(out, "w", newline="") as oh:
        w = csv.writer(oh, lineterminator="\n")
        w.writerow("set id api value lo hi verdict lre_lo ulps".split())
        for ln in fh:
            pid, api, raw = ln.rstrip("\n").split("\t")
            pt = parse_id(pid)
            key = (api, sets.get(pid, "?"))
            c = census[key]
            ref = refs.get(pid)
            if ref is None:
                c["no-ref"] += 1
                w.writerow([sets.get(pid), pid, api, raw, "", "", "no-ref", "", ""])
                continue
            lo, hi, kind = ref
            if pt.is_probability is False:
                mx, mn = max(abs(lo), abs(hi)), min(abs(lo), abs(hi))
                if mx != 0 and (mx > dmax or mn < dmin):
                    c["out-of-range"] += 1
                    w.writerow([sets.get(pid), pid, api, raw, sf(lo), sf(hi), "out-of-range", "", ""])
                    continue
            if raw == "HANG" or raw.startswith("ERR"):
                c["error"] += 1
                w.writerow([sets.get(pid), pid, api, raw, sf(lo), sf(hi), "error", "", ""])
                continue
            v = float(raw)
            abs_ = fmt["floor"] if pt.is_probability else 0.0
            vd = verdict(v, lo, hi, fmt["rel"], abs_)
            lre, _ = lre_bounds(v, lo, hi)
            u = miss_ulps(v, lo, hi, fmt["prec"], fmt["emin"])
            c["scored"] += 1
            c[vd] += 1
            if not math.isfinite(v):
                c["nonfinite"] += 1
            below = hi < Fraction(fmt["floor"]) and pt.is_probability
            if not below:
                c["ulp_max"] = max(c["ulp_max"], u) if u != math.inf else c["ulp_max"]
                if not math.isnan(lre):
                    for t in (6, 10, 13):
                        if lre < t:
                            c[f"lre<{t}"] += 1
            w.writerow([sets.get(pid), pid, api, raw, sf(lo), sf(hi), vd,
                        f"{lre:.2f}", f"{u:.3g}"])  # fmt: skip
    cols = "scored correct wrong undecided nonfinite error out-of-range no-ref lre<6 lre<10 lre<13 ulp_max".split()
    print("api,set," + ",".join(cols))
    for key in sorted(census):
        c = census[key]
        print(",".join(key) + "," + ",".join(f"{c[k]:.3g}" if k == "ulp_max" else str(c[k]) for k in cols))


if __name__ == "__main__":
    main()
