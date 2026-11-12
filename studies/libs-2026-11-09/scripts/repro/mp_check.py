"""mpmath at 80 digits on the enclosures the drafts cite. Evidence about the use of Arb, not a proof."""
import csv, mpmath as mp
from fractions import Fraction
from pathlib import Path
REPO_DATA = Path(__file__).resolve().parents[4] / "data"  # certtables/data
mp.mp.dps = 80
rows = {}
for name in ("certtables-v1.csv", "certtables-v1-random.csv"):
    for r in csv.DictReader(open(REPO_DATA / name, newline="")):
        rows[r["id"]] = (r["lo"], r["hi"])
def chi(df, x, up):
    v = mp.gammainc(mp.mpf(df) / 2, mp.mpf(x) / 2, mp.inf if up else 0, regularized=True) if up else mp.gammainc(mp.mpf(df) / 2, 0, mp.mpf(x) / 2, regularized=True)
    return v
cases = [("chi2.cdf(df=50.0;x=50.0)", lambda: chi(50.0, 50.0, False)), ("chi2.sf(df=50.0;x=50.0)", lambda: chi(50.0, 50.0, True)),
         ("chi2.cdf(df=300.0;x=300.0)", lambda: chi(300.0, 300.0, False)), ("chi2.sf(df=20.0;x=100.0)", lambda: chi(20.0, 100.0, True)),
         ("chi2.sf(df=1.80006;x=90.5982)", lambda: chi(1.80006, 90.5982, True)), ("chi2.cdf(df=100000.0;x=90000.0)", lambda: chi(1e5, 9e4, False)),
         ("chi2.sf(df=10000.0;x=15000.0)", lambda: chi(1e4, 1.5e4, True)), ("chi2.cdf(df=1000000.0;x=1000000.0)", lambda: chi(1e6, 1e6, False)),
         ("chi2.sf(df=1000000.0;x=1000000.0)", lambda: chi(1e6, 1e6, True)),
         ("norm.ppf(p=1e-15)", lambda: mp.findroot(lambda z: mp.ncdf(z) - mp.mpf(1e-15), -7.94)),
         ("binom.sf(n=50;p=0.01;k=25)", lambda: sum(mp.binomial(50, i) * mp.mpf(0.01) ** i * (1 - mp.mpf(0.01)) ** (50 - i) for i in range(26, 51)))]
for pid, fn in cases:
    v = fn(); lo, hi = mp.mpf(rows[pid][0]), mp.mpf(rows[pid][1])
    print(pid, mp.nstr(v, 24), "inside" if lo <= v <= hi else f"OUTSIDE [{rows[pid][0]},{rows[pid][1]}]")
