"""Fresh Arb enclosures at exact inputs the table does not hold (binary32 inputs, beta).

Uses certtables' own enclose functions, so the proofs are the ones DESIGN section 2 describes.
Appends one row per point and fsyncs; finished ids are skipped on rerun. Failures go to
<out>.errors with the reason: RootOutOfRange means proved outside [2^-1100, 2^1100].

usage: make_refs.py points.tsv out.csv [worker_index worker_count] [--fams norm,chi2]
"""

import csv
import os
import sys
import time
from fractions import Fraction
from pathlib import Path

from certtables.enclose import continuous, enclose_point
from certtables.enclose.arbutil import RootOutOfRange, to_decimal
from certtables.grid import parse_id


def enclose(pt):
    if pt.family == "beta":
        p = pt.p
        params = {"a": float(p["a"]), "b": float(p["b"])}
        if pt.function in ("cdf", "sf"):
            return continuous.value("beta", pt.function, float(p["x"]), params)
        return continuous.quantile("beta", pt.function, float(p["p"]), params)
    return enclose_point(pt)


def main() -> None:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    fams = None
    for a in sys.argv[1:]:
        if a.startswith("--fams="):
            fams = set(a.split("=")[1].split(","))
    src, out = Path(args[0]), Path(args[1])
    w, n = (int(args[2]), int(args[3])) if len(args) > 2 else (0, 1)
    out = out.with_name(f"{out.stem}-w{w}{out.suffix}")
    done = set()
    for path in (out, out.with_suffix(".errors")):
        if path.exists():
            with open(path, newline="") as fh:
                for r in csv.reader(fh):
                    if len(r) >= 2:
                        done.add(r[0])
    ids = [ln.rstrip("\n").split("\t")[1] for ln in open(src)]
    ids = [i for j, i in enumerate(ids) if j % n == w and i not in done]
    if fams:
        ids = [i for i in ids if i.split(".")[0] in fams]
    print(f"worker {w}: {len(ids)} to do, {len(done)} done", flush=True)
    t0 = time.time()
    with open(out, "a", newline="") as fh, open(out.with_suffix(".errors"), "a", newline="") as eh:
        cw, ce = csv.writer(fh, lineterminator="\n"), csv.writer(eh, lineterminator="\n")
        for c, pid in enumerate(ids, 1):
            pt = parse_id(pid)
            t1 = time.time()
            try:
                enc = enclose(pt)
                if enc.lo > enc.hi:
                    raise ValueError("lo > hi")
                cw.writerow([pid, enc.kind if hasattr(enc, "kind") else "certified", enc.method,
                             to_decimal(Fraction(enc.lo), 25, "down"),
                             to_decimal(Fraction(enc.hi), 25, "up"), f"{time.time() - t1:.3f}"])  # fmt: skip
                fh.flush()
                os.fsync(fh.fileno())
            except RootOutOfRange as exc:
                ce.writerow([pid, "out-of-range", str(exc)[:200]])
                eh.flush()
            except Exception as exc:  # recorded, not dropped
                ce.writerow([pid, "error", f"{type(exc).__name__}: {exc}"[:300]])
                eh.flush()
            if c % 200 == 0:
                print(f"worker {w}: {c}/{len(ids)} {time.time() - t0:.0f}s", flush=True)
    print(f"worker {w}: finished {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
