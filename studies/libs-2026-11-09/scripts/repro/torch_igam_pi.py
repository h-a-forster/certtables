"""Replicate torch's _igam_helper_asymptotic_series in plain Python, table parsed from Math.h,
once with single-precision pi (as shipped) and once with double pi, against torch's output and
the enclosure. Run in ref/venv. usage: torch_igam_pi.py Math.h torch-f64-scored.csv"""
import csv, math, re, sys
src = open(sys.argv[1]).read()
tab = src[src.index("static constexpr scalar_t d[25][25]"):]
tab = tab[tab.index("{{"):tab.index("};")]
vals = [float(v) for v in re.findall(r"-?\d\.\d+e[-+]?\d+|-?\d\.\d+|\b0\b", tab)]
assert len(vals) == 625, len(vals)
d = [vals[i * 25:(i + 1) * 25] for i in range(25)]
PI32 = 3.1415927410125732
EPS = 2.220446049250313e-16

def series(a, x, igam, pi):
    lam = x / a
    sigma = (x - a) / a
    if igam:
        sgn = -1.0 if lam > 1 else 1.0
        sgnx = 1.0 if lam > 1 else -1.0
    absol = math.inf
    eta = 0.0
    if lam > 1:
        eta = math.sqrt(-2 * (math.log1p(sigma) - sigma))
    elif lam < 1:
        eta = -math.sqrt(-2 * (math.log1p(sigma) - sigma))
    sgn = (-1.0 if igam else 1.0)
    if lam < 1 and igam: sgn = sgn
    res = 0.5 * math.erfc(sgn * eta * math.sqrt(a / 2))
    etapow = [1.0] + [0.0] * 24
    maxpow = 0
    s = 0.0
    afac = 1.0
    for k in range(25):
        ck = d[k][0]
        for n in range(1, 25):
            if n > maxpow:
                etapow[n] = eta * etapow[n - 1]
                maxpow += 1
            t = d[k][n] * etapow[n]
            ck += t
            if abs(t) < EPS * abs(ck):
                break
        term = ck * afac
        if abs(term) > absol:
            break
        s += term
        if abs(term) < EPS * abs(s):
            break
        absol = abs(term)
        afac /= a
    return res + sgn * math.exp(-0.5 * a * eta * eta) * s / math.sqrt(2 * pi * a)

n = bad32 = bad64 = in64 = 0
worst64 = 0.0
same = 0
for r in csv.DictReader(open(sys.argv[2])):
    if not ("Chi2" in r["api"] or "gamma" in r["api"]) or r["verdict"] not in ("wrong", "correct"):
        continue
    m = re.search(r"df=([^;]+);x=([^)]+)", r["id"])
    a, x = float(m[1]) / 2, float(m[2]) / 2
    ratio = abs(x - a) / a
    if not ((20 < a < 200 and ratio < 0.3) or (a > 200 and ratio < 4.5 / math.sqrt(a))):
        continue
    igam = ".cdf(" in r["id"]
    v32, v64 = series(a, x, igam, PI32), series(a, x, igam, math.pi)
    lo, hi, tv = float(r["lo"]), float(r["hi"]), float(r["value"])
    ref = (lo + hi) / 2
    n += 1
    same += abs(v32 - tv) <= 4e-15 * abs(tv) + 1e-320
    bad32 += abs(v32 - ref) > 1e-12 * ref
    bad64 += abs(v64 - ref) > 1e-12 * ref
    worst64 = max(worst64, abs(v64 - ref) / ref)
print(f"Temme-branch points {n}; replica with pi32 equals torch to 4e-15 rel on {same}; wrong with pi32 {bad32}; wrong with double pi {bad64}; worst rel error with double pi {worst64:.2e}")
