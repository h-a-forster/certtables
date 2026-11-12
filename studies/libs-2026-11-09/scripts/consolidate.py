"""data/claims.csv (every wrong value, with 25-digit bounds and a root-cause class) and
data/census.md (counts and random-draw rates). Run in ref/venv after score.py."""

import csv
import math
import re
from collections import defaultdict
from fractions import Fraction
from pathlib import Path

from certtables.enclose.continuous import clopper_pearson

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
REPO = ROOT.parents[1]  # the certtables repository
VER = {
    "torch": "torch 2.15.0+cpu",
    "jax": "jax 0.12.0.1 (CPU)",
    "tfp": "tensorflow-probability 0.25.0, tensorflow-cpu 2.21.0",
    "distributionsjl": "Distributions.jl 0.25.134, StatsFuns 2.2.1, SpecialFunctions 2.9.1, Julia 1.13.3",
}
csv.field_size_limit(10**7)


def strings(mode):
    out = {}
    if mode == "f64":
        for name in ("certtables-v1.csv", "certtables-v1-random.csv"):
            for r in csv.DictReader(open(REPO / "data" / name, newline="")):
                out[r["id"]] = (r["lo"], r["hi"])
    else:
        for p in DATA.glob(f"refs-{mode}-w*.csv"):
            for r in csv.reader(open(p, newline="")):
                out[r[0]] = (r[3], r[4])
    return out


def temme(a, x):
    ratio = abs(x - a) / a
    return (20 < a < 200 and ratio < 0.3) or (a > 200 and ratio < 4.5 / math.sqrt(a))


def klass(lib, r, lre):
    pid, api = r["id"], r["api"]
    fam = pid.split(".")[0]
    v = r["value"]
    if lib == "torch":
        if "Normal.cdf" in api or "ndtr" == api[-4:]:
            return "T-cancel: 1+erf cancels for x<0 (known: pytorch#192295, PR #189234)"
        if "icdf" in api:
            return "T-icdf: erfinv(2p-1) loses small p"
        m = re.search(r"df=([^;]+);x=([^)]+)", pid)
        a, x = float(m[1]) / 2, float(m[2]) / 2
        if temme(a, x):
            return "T-pi: single-precision pi in the double igamma series (Math.h)"
        return "T-marginal: igamma outside that series, LRE >= 11"
    if lib == "jax":
        if fam == "beta":
            return "J-betainc: stats.beta cdf/sf through betainc (known: jax#41678)"
        return "J-large-a: gammainc/gammaincc at large a (known in part: jax#41201, PR #41255)"
    if lib == "tfp":
        if "survival_function" in api and fam in ("chi2", "binom", "beta"):
            return "P-sf: survival_function falls back to 1 - cdf"
        if fam == "chi2":
            return "P-igamma: tf igamma/quantile at large df"
        if fam == "t" and "quantile" in api:
            return "P-t-quantile: far-tail StudentT.quantile, inf or digits lost"
        if fam == "t":
            return "P-t32: StudentT.cdf/sf NaN in binary32 at large df or |x|"
        if fam == "beta" and "quantile" in api:
            return "P-beta-quantile: Beta.quantile wrong at p=0.5 (known: tfp#2034) and at tiny p or NaN"
        if lre != "nan" and float(lre) >= 10:
            return "P-marginal: LRE >= 10, betainc-based"
        return "P-beta-binom: betainc-based, see claims"
    if lib == "distributionsjl":
        if fam == "ks":
            m = re.search(r"n=(\d+);d=([0-9.e-]+)", pid)
            n, d = int(m[1]), float(m[2])
            t = Fraction(d) * n
            h = t.__ceil__() - t
            if float(t) <= 1 and t <= Fraction(1, 2) + Fraction(1, 2**40):
                return "D-ks-convention: d one rounding above 1/(2n), not a defect"
            if h > Fraction(1, 2) and t > 1 and float(t * Fraction(d)) < 4:
                return "D-ks-durbin: (2h-1) without ^m in cdf_durbin"
            return "D-ks-marginal: LRE about 11, Miller or Durbin rounding"
        if fam == "t" and ("uantile" in api):
            return "D-t-quantile: t through fdistinvccdf (open PR StatsFuns#229)"
        if fam == "t":
            return "D-t-cdf: inv(1+nu/x^2) collapses to 1 (noted in StatsFuns#229)"
        if fam == "f" and "uantile" in api:
            return "D-f-quantile: x/(1-x) after beta_inc_inv"
        if fam == "f":
            return "D-f-tail: beta argument near 1 in fdist(c)cdf"
        if fam == "chi2":
            return "D-chi2: gamma_inc/gamma_inc_inv failure at extreme p (SpecialFunctions)"
        if fam == "beta":
            return "D-beta-quantile: beta_inc_inv sticks near 1e-154 at tiny p"
        return "D-other"
    return "?"


def main():
    claims = open(DATA / "claims.csv", "w", newline="")
    cw = csv.writer(claims, lineterminator="\n")
    cw.writerow("library version precision set function input returned lo hi ulps lre class".split())
    lines = []
    classes = defaultdict(lambda: [0, ""])
    for lib, mode in (("torch", "f64"), ("torch", "f32"), ("jax", "f64"), ("jax", "f32"), ("tfp", "f64"),
                      ("tfp", "f32"), ("tfp", "ext"), ("jax", "ext"), ("distributionsjl", "f64"),
                      ("distributionsjl", "ks"), ("distributionsjl", "ext")):  # fmt: skip
        path = DATA / f"{lib}-{mode}-scored.csv"
        if not path.exists():
            continue
        s = strings(mode)
        per = defaultdict(lambda: defaultdict(int))
        for r in csv.DictReader(open(path)):
            key = (r["id"].split(".")[0] + "." + r["id"].split(".")[1].split("(")[0], r["api"], r["set"])
            c = per[key]
            c[r["verdict"]] += 1
            if r["verdict"] in ("wrong", "error"):
                kl = klass(lib, r, r["lre_lo"])
                if r["verdict"] == "error":
                    kl = "E " + kl
                lo, hi = s.get(r["id"], ("", ""))
                cw.writerow([lib, VER[lib], {"f32": "binary32"}.get(mode, "binary64"), r["set"], r["api"], r["id"],
                             r["value"], lo, hi, r["ulps"], r["lre_lo"], kl])  # fmt: skip
                k = (lib, mode, kl)
                classes[k][0] += 1
                u = float(r["ulps"]) if r["ulps"] else math.inf
                if not classes[k][1] or u > classes[k][2]:
                    classes[k][1:] = [f"{r['id']} {r['api']} -> {r['value'][:26]}", u]
        lines.append(f"\n### {lib} {mode}\n\n| function | api | set | scored | wrong | error | out of range | rate (95% CP) |\n|---|---|---|---|---|---|---|---|")
        for (fn, api, st), c in sorted(per.items()):
            sc = c["correct"] + c["wrong"] + c["undecided"]
            rate = ""
            if st == "random" and sc:
                lo = clopper_pearson(c["wrong"], sc, 0.05, "lower")
                hi = clopper_pearson(c["wrong"], sc, 0.05, "upper")
                rate = f"{c['wrong'] / sc:.2%} ({float(lo.lo):.2%} to {float(hi.hi):.2%})"
            lines.append(f"| {fn} | {api} | {st} | {sc} | {c['wrong']} | {c['error']} | {c['out-of-range']} | {rate} |")
    claims.close()
    lines.append("\n### Classes\n\n| library | run | class | count | worst |\n|---|---|---|---|---|")
    for (lib, mode, kl), v in sorted(classes.items()):
        lines.append(f"| {lib} | {mode} | {kl} | {v[0]} | {v[1]} |")
    (DATA / "census.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines[-len(classes) - 2:]))


if __name__ == "__main__":
    main()
