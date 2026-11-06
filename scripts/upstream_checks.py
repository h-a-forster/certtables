"""Numbers behind docs/upstream/: root causes of SciPy's certified-wrong quantiles.

Reads the table and results/targets/scipy-*.csv, calls SciPy, and writes
results/upstream-checks.json. Nothing here is certified except the table it compares with.

    uv run python scripts/upstream_checks.py
"""

from __future__ import annotations

import csv
import json
import math
import warnings
from collections import Counter
from pathlib import Path

import scipy
import scipy.special as sc
from scipy import stats

from certtables.grid import parse_id
from certtables.study import verdict
from certtables.table import read_rows
from certtables.targets.base import REL_DOUBLE

ROOT = Path(__file__).resolve().parents[1]


def _results() -> dict[str, dict[str, str]]:
    out = {}
    for label in ("grid", "random"):
        with open(ROOT / "results" / "targets" / f"scipy-{label}.csv", newline="") as fh:
            out.update({r["id"]: r for r in csv.DictReader(fh)})
    return out


def _table():
    rows = []
    for name in ("certtables-v1.csv", "certtables-v1-random.csv"):
        rows += read_rows(ROOT / "data" / name)
    return rows


def f_isf_candidate(q: float, d1: float, d2: float) -> float:
    """isf through the inverse of the small tail: SF(x) = I_z(d2/2, d1/2), z = d2/(d2 + d1 x)
    (SciPy's current path above q = 0.01, where 1/z - 1 can cancel and 1 - q does not)."""
    if q >= 0.01:
        return float(stats.f.ppf(1.0 - q, d1, d2))
    z = sc.betaincinv(d2 / 2, d1 / 2, q)
    return float(d2 / d1 * (1 / z - 1))


def scored_rows(rows, res, family, function):
    return [r for r in rows if r.family == family and r.function == function and r.id in res]


def main() -> None:
    rows, res = _table(), _results()
    out: dict[str, object] = {"scipy": scipy.__version__}

    # f.isf: rv_continuous' default _isf is _ppf(1 - q), and f has no _isf of its own.
    f_isf = [
        r for r in scored_rows(rows, res, "f", "isf") if res[r.id]["verdict"] != "out-of-range"
    ]
    same = fixed = still = nan = 0
    wrong = [r for r in f_isf if res[r.id]["verdict"] == "wrong"]
    regress = 0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for r in f_isf:
            p = parse_id(r.id).p
            q, d1, d2 = float(p["p"]), float(p["dfn"]), float(p["dfd"])
            got = float(stats.f.isf(q, d1, d2))
            twin = float(stats.f.ppf(1.0 - q, d1, d2))
            same += got == twin or (math.isnan(got) and math.isnan(twin))
            cand = f_isf_candidate(q, d1, d2)
            v = verdict(cand, r.lo_q, r.hi_q, REL_DOUBLE, 0.0)
            if res[r.id]["verdict"] == "wrong":
                fixed += v == "correct"
                still += v != "correct"
                nan += math.isnan(cand)
            else:
                regress += v != "correct"
    out["f_isf"] = {
        "scored": len(f_isf),
        "wrong": len(wrong),
        "equal_to_ppf_of_1_minus_q": same,
        "largest_wrong_q": max(float(parse_id(r.id).p["p"]) for r in wrong) if wrong else None,
        "values": dict(Counter(res[r.id]["status"] for r in wrong)),
        "candidate_fixes": fixed,
        "candidate_still_wrong": still,
        "candidate_nan": nan,
        "candidate_breaks_correct_rows": regress,
    }

    # t.ppf / t.isf: stdtrit, the Boost Student t quantile.
    kinds: Counter[str] = Counter()
    onset: dict[str, float] = {}
    ratios: list[float] = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for fn in ("ppf", "isf"):
            for r in scored_rows(rows, res, "t", fn):
                rec = res[r.id]
                if rec["verdict"] != "wrong":
                    continue
                v, c = float(rec["value"]), float(r.lo_q)
                if math.isinf(v) and math.copysign(1, v) != math.copysign(1, c):
                    kind = "infinite, wrong sign"
                elif math.isinf(v) or math.isnan(v):
                    kind = "other non-finite"
                elif abs(c) > 1e154:
                    kind = "finite, stuck near sqrt(df / DBL_MIN)"
                else:
                    kind = "finite, off by a factor below 3"
                    ratios.append(abs(c) / abs(v))
                kinds[kind] += 1
                onset[kind] = max(onset.get(kind, 0.0), float(parse_id(r.id).p["p"]))
    out["t_quantile"] = {
        "kinds": dict(kinds),
        "largest_p_by_kind": onset,
        "factor_range": [min(ratios), max(ratios)] if ratios else None,
    }

    # Remaining wrong rows, for the triage table.
    other: Counter[str] = Counter()
    for pid, rec in res.items():
        if rec["verdict"] == "wrong" and not pid.startswith(("f.isf", "t.ppf", "t.isf")):
            other[pid.split("(")[0] + "/" + rec["status"]] += 1
    out["other_wrong"] = dict(sorted(other.items()))
    path = ROOT / "results" / "upstream-checks.json"
    path.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
