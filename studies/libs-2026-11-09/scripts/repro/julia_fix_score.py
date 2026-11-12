import csv, math
from collections import Counter
from fractions import Fraction
from pathlib import Path
REPO_DATA = Path(__file__).resolve().parents[4] / "data"  # certtables/data
from certtables.study import verdict
s = {}
for name in ("certtables-v1.csv", "certtables-v1-random.csv"):
    for r in csv.DictReader(open(REPO_DATA / name, newline="")):
        s[r["id"]] = (Fraction(r["lo"]), Fraction(r["hi"]))
c = Counter(); bad = {}
for ln in open("data/_jfix.out"):
    pid, v = ln.rstrip("\n").split("\t"); v = float(v)
    lo, hi = s[pid]
    fam = pid.split("(")[0]
    prob = fam.endswith("cdf") or fam.endswith("sf")
    vd = verdict(v, lo, hi, 1e-12, 2.2250738585072014e-308 if prob else 0.0)
    c[fam, vd] += 1
    if vd != "correct": bad.setdefault(fam, []).append((pid, v, float(lo) if hi < 1e300 else "big"))
for k in sorted(c): print(k, c[k])
for k, v in bad.items(): print(k, v[:3])
