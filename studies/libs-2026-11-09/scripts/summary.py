"""One line per library, precision, family, function: scored and wrong by set, with 95% CP on random."""
import csv, sys
from collections import defaultdict
from certtables.enclose.continuous import clopper_pearson
for lib, mode in [("torch","f64"),("torch","f32"),("jax","f64"),("jax","f32"),("tfp","f64"),("tfp","f32"),("distributionsjl","f64"),("jax","ext"),("tfp","ext"),("distributionsjl","ext")]:
    try: rows = list(csv.DictReader(open(f"data/{lib}-{mode}-scored.csv")))
    except FileNotFoundError: continue
    per = defaultdict(lambda: defaultdict(int)); seen = 0
    for r in rows:
        fam = r["id"].split(".")[0]; api = r["api"]
        p = per[fam]
        if r["verdict"] in ("correct","wrong","undecided"):
            p[r["set"]+"_n"] += 1
            if r["verdict"] == "wrong": p[r["set"]+"_w"] += 1
        if r["verdict"] == "error": p["err"] += 1
        if r["verdict"] == "undecided": p["und"] += 1
        if r["verdict"] == "out-of-range": p["oor"] += 1
    print(f"## {lib} {mode}: evaluations {len(rows)}")
    for fam, p in sorted(per.items()):
        n, w = p["random_n"], p["random_w"]
        rate = ""
        if n:
            lo = float(clopper_pearson(w, n, 0.05, "lower").lo); hi = float(clopper_pearson(w, n, 0.05, "upper").hi)
            rate = f"{w}/{n} = {w/n:.2%} ({lo:.2%} to {hi:.2%})"
        print(f"| {fam} | {p['grid_w']}/{p['grid_n']} | {rate or 'not drawn'} | {p['err']} | {p['oor']} | {p['und']} |")
