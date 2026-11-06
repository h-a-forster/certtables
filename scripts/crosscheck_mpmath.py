"""Independent check of table rows with mpmath at 80 digits (evidence, not a proof).

Arb proves a value of the formula it is given; this checks the formulas against a separate
implementation. It covers every certified row some target scored as wrong, plus a seeded
sample of the other continuous rows. A CDF or SF row passes when mpmath's value lies in the
enclosure widened by 1e-40 relative; a quantile row passes when mpmath's CDF - p (or SF - p)
changes sign across the enclosure, each end widened by 1e-15 of the enclosure's own width
plus 1e-30 relative (an exact root must give |CDF - p| below 1e-60). Points run in a worker
process with a time limit; a point that runs out of time is counted as `timeout`, never as a
pass.

Output: results/crosscheck.json (counts) and results/crosscheck.csv (one line per point).
Rerunning skips the points already in the CSV.

    uv run python scripts/crosscheck_mpmath.py [--sample 150] [--limit 10] [--minutes 12]
"""

from __future__ import annotations

import argparse
import csv
import json
import multiprocessing as mp_
import queue
import random
import time
from collections import Counter
from fractions import Fraction
from pathlib import Path

import mpmath

from certtables.grid import parse_id
from certtables.table import read_rows

DPS = 80
ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = ROOT / "results" / "crosscheck.csv"
JSON_PATH = ROOT / "results" / "crosscheck.json"
COLUMNS = ("id", "origin", "outcome", "seconds")


def _mpf(x: Fraction):
    return mpmath.mpf(x.numerator) / mpmath.mpf(x.denominator)


def _values(pt, x):
    """(cdf, sf) of the family at the mpf ``x``; the small tail is computed directly."""
    p = pt.p
    if pt.family == "norm":
        z = x / mpmath.sqrt(2)
        return mpmath.erfc(-z) / 2, mpmath.erfc(z) / 2
    if pt.family == "t":
        df = mpmath.mpf(float(p["df"]))
        tail = mpmath.betainc(df / 2, 0.5, 0, df / (df + x * x), regularized=True) / 2
        return (tail, 1 - tail) if x < 0 else (1 - tail, tail)
    if pt.family == "chi2":
        k = mpmath.mpf(float(p["df"]))
        cdf = mpmath.gammainc(k / 2, 0, x / 2, regularized=True)
        return cdf, mpmath.gammainc(k / 2, x / 2, mpmath.inf, regularized=True)
    d1, d2 = mpmath.mpf(float(p["dfn"])), mpmath.mpf(float(p["dfd"]))
    cdf = mpmath.betainc(d1 / 2, d2 / 2, 0, d1 * x / (d1 * x + d2), regularized=True)
    return cdf, mpmath.betainc(d2 / 2, d1 / 2, 0, d2 / (d2 + d1 * x), regularized=True)


def check(pid: str, lo: str, hi: str) -> str:
    pt = parse_id(pid)
    lo_q, hi_q = Fraction(lo), Fraction(hi)
    with mpmath.workdps(DPS):
        if pt.function in ("cdf", "sf"):
            cdf, sf = _values(pt, mpmath.mpf(float(pt.p["x"])))
            v = cdf if pt.function == "cdf" else sf
            slack = abs(v) * mpmath.mpf(10) ** -40
            return "pass" if _mpf(lo_q) - slack <= v <= _mpf(hi_q) + slack else "FAIL"
        target = mpmath.mpf(float(pt.p["p"]))
        slack = (hi_q - lo_q) * Fraction(1, 10**15) + max(abs(lo_q), abs(hi_q)) / 10**30
        below = _values(pt, _mpf(lo_q - slack))
        above = _values(pt, _mpf(hi_q + slack))
        i = 0 if pt.function == "ppf" else 1
        sign = 1 if pt.function == "ppf" else -1
        if lo_q == hi_q:  # an exact root, such as a median at 0
            return "pass" if abs(below[i] - target) < mpmath.mpf(10) ** -60 else "FAIL"
        ok = sign * (below[i] - target) < 0 < sign * (above[i] - target)
        return "pass" if ok else "FAIL"


def _worker(inbox, outbox) -> None:
    while True:
        pid, lo, hi = inbox.get()
        try:
            outbox.put((pid, check(pid, lo, hi)))
        except Exception as exc:  # recorded as an outcome, not a pass
            outbox.put((pid, f"error:{type(exc).__name__}"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, default=150, help="sampled rows per family")
    ap.add_argument("--limit", type=float, default=10.0, help="seconds per point")
    ap.add_argument("--minutes", type=float, default=12.0, help="stop after this long")
    args = ap.parse_args()

    rows = {}
    for name in ("certtables-v1.csv", "certtables-v1-random.csv"):
        rows.update({r.id: r for r in read_rows(ROOT / "data" / name)})
    with open(ROOT / "results" / "wrong.csv", newline="", encoding="utf-8") as fh:
        wrong = sorted({r["id"] for r in csv.DictReader(fh) if r["verdict"] == "wrong"})
    fams = ("norm", "t", "chi2", "f")
    usable = {pid for pid, r in rows.items() if r.family in fams and "below-2^" not in r.method}
    todo = {pid: "wrong" for pid in wrong if pid in usable}
    rng = random.Random(20261106)
    for fam in fams:
        pool = sorted(pid for pid in usable if rows[pid].family == fam and pid not in todo)
        todo.update({pid: "sample" for pid in rng.sample(pool, args.sample)})

    done: dict[str, dict[str, str]] = {}
    if CSV_PATH.exists():
        with open(CSV_PATH, newline="", encoding="utf-8") as fh:
            done = {r["id"]: r for r in csv.DictReader(fh) if r["outcome"] != "FAIL"}
    else:
        CSV_PATH.write_text(",".join(COLUMNS) + "\n", encoding="utf-8")

    def spawn():
        inbox, outbox = mp_.Queue(), mp_.Queue()
        proc = mp_.Process(target=_worker, args=(inbox, outbox), daemon=True)
        proc.start()
        return proc, inbox, outbox

    proc, inbox, outbox = spawn()
    start = time.monotonic()
    for pid, origin in todo.items():
        if pid in done:
            continue
        if time.monotonic() - start > 60 * args.minutes:
            print("time limit reached; rerun to continue")
            break
        t0 = time.monotonic()
        inbox.put((pid, rows[pid].lo, rows[pid].hi))
        try:
            _, outcome = outbox.get(timeout=args.limit)
        except queue.Empty:
            proc.terminate()
            proc.join()
            proc, inbox, outbox = spawn()
            outcome = "timeout"
        rec = {
            "id": pid,
            "origin": origin,
            "outcome": outcome,
            "seconds": f"{time.monotonic() - t0:.2f}",
        }
        done[pid] = rec
        with open(CSV_PATH, "a", newline="", encoding="utf-8") as fh:
            csv.DictWriter(fh, COLUMNS, lineterminator="\n").writerow(rec)
    proc.terminate()

    # Only the points of the current selection count (wrong.csv may have shrunk since).
    kept = [done[pid] for pid in todo if pid in done]
    with open(CSV_PATH, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, COLUMNS, lineterminator="\n")
        w.writeheader()
        w.writerows(kept)
    counts = Counter((r["origin"], r["outcome"]) for r in kept)
    summary = {
        "dps": DPS,
        "seconds_per_point": args.limit,
        "selected": len(todo),
        "checked": sum(counts.values()),
        "counts": {f"{o}/{r}": n for (o, r), n in sorted(counts.items())},
    }
    JSON_PATH.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
