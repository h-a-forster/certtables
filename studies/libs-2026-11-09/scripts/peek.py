"""Print wrong values for one scored file. usage: peek.py <scored.csv> <api-substring> [N] [--worst]"""

import csv
import sys

path, sub = sys.argv[1], sys.argv[2]
n = int(sys.argv[3]) if len(sys.argv) > 3 else 15
rows = [r for r in csv.DictReader(open(path)) if sub in r["api"] and r["verdict"] in ("wrong", "error")]
rows.sort(key=lambda r: -(float(r["ulps"]) if r["ulps"] else 1e999))
step = max(1, len(rows) // n)
print(len(rows), "rows")
for r in rows[::step][:n]:
    print(r["set"][0], r["id"], r["api"], "v=" + r["value"][:40], "[" + r["lo"][:22] + "]", "lre", r["lre_lo"], "ulp", r["ulps"])
