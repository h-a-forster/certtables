"""Durbin branch: shipped KSDist and the one-line fix against the exact rationals."""
import csv, re
from fractions import Fraction
refs = {r[0]: (Fraction(r[3]), Fraction(r[4])) for r in csv.reader(open("data/refs-ks-w0.csv"))}
n = sw = fw = 0; sh = sl = 0; worst = (0, ""); fworst = 0.0; rows=[]
for ln in open("data/ks-fixed.tsv"):
    pid, a, b = ln.rstrip("\n").split("\t")
    a, b = float(a), float(b)
    lo, hi = refs[pid]
    N = int(re.search(r"n=(\d+)", pid)[1]); d = float(re.search(r"d=([0-9.e-]+)", pid)[1])
    t = Fraction(d) * N; k = t.__ceil__(); h = k - t
    rel = lambda v: 0.0 if lo <= Fraction(v) <= hi else float(abs(Fraction(v) - lo) / lo)
    ra, rb = rel(a), rel(b)
    n += 1
    if h > Fraction(1, 2):
        sh += 1
        if ra > 1e-12: sw += 1
    else:
        sl += 1
        assert ra <= 1e-10, (pid, ra)
    if rb > 1e-12: fw += 1
    fworst = max(fworst, rb)
    if ra > worst[0]: worst = (ra, pid, a, float(lo))
    rows.append((ra, pid, a, float(lo), b, float(h)))
print(f"points {n}; h>1/2: {sh}, wrong in shipped {sw}; h<=1/2: {sl}; wrong with the fix {fw}; worst rel with fix {fworst:.2e}; worst shipped {worst}")
for r in sorted(rows, reverse=True)[:6]: print(r)
big = [r for r in rows if r[0] > 1e-3]; print("rel error >1e-3:", len(big), "of", n)
print("with n=100:", [r for r in sorted(rows, reverse=True) if "n=100;" in r[1]][:1], "n=50:", [r for r in sorted(rows, reverse=True) if "n=50;" in r[1]][:1], "n=20:", [r for r in sorted(rows, reverse=True) if "n=20;" in r[1]][:1])
