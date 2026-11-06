"""Scoring targets against the table: LRE bounds, verdicts and the census
(docs/DESIGN.md section 4).

Because the reference is an interval ``[lo, hi]``, the LRE of a target value ``v`` is an
interval too: ``LRE_lo`` is the LRE against the worse endpoint, ``LRE_hi`` against the best
point of the enclosure (``cap`` when ``v`` lies inside it). Both use strdcheck's
conventions (:mod:`certtables.lre`: clipped to ``[0, cap]``, below 1 reported as 0, ``nan``
for a non-finite value, log absolute error when the reference can be 0).

Verdicts, with tolerance ``tol(c) = max(rel * |c|, abs)`` declared by the target:

- ``correct``: ``|v - c| <= tol(c)`` for every ``c`` in the enclosure; the row records
  ``digits = floor(LRE_lo)`` as certified correct digits;
- ``wrong``: ``|v - c| > tol(c)`` for every ``c`` (certified wrong), and every non-finite
  ``v`` against a finite reference;
  (both checked through sufficient conditions, see :func:`verdict`);
- ``undecided``: neither;
- ``error``: the target raised; ``out-of-range``: the reference quantile lies outside the
  normal binary64 range (not scored).

Rows of kind ``reference`` (uncertified mpmath values) get ``ref-agree``,
``ref-disagree`` or ``ref-undecided`` and never enter the certified census.
"""

from __future__ import annotations

import csv
import json
import math
import sys
import time
import traceback
from fractions import Fraction
from pathlib import Path

from certtables.grid import parse_id
from certtables.lre import EXACT_CAP, clip_lre
from certtables.table import Row, read_rows
from certtables.targets import get_target

DBL_MIN = Fraction(sys.float_info.min)
DBL_MAX = Fraction(sys.float_info.max)
RESULT_COLUMNS = ("id", "family", "function", "kind", "value", "status", "lre_lo", "lre_hi",
                  "verdict", "digits", "rel_tol", "abs_tol", "note")  # fmt: skip


def _log10(x: Fraction) -> float:
    """log10 of a positive rational without float underflow."""
    return math.log10(x.numerator) - math.log10(x.denominator)


def lre_bounds(v: float, lo: Fraction, hi: Fraction, cap: int = EXACT_CAP) -> tuple[float, float]:
    """(LRE_lo, LRE_hi) of ``v`` against every reference value in ``[lo, hi]``.

    Relative error when the enclosure excludes 0 (|v - c|/|c| is quasi-convex in c, so its
    extremes over the interval are at the endpoints or 0 when v is inside); absolute error
    when it contains 0, as strdcheck does for a zero certified value.
    """
    if lo > hi:
        raise ValueError("lo > hi")
    if not math.isfinite(v):
        return math.nan, math.nan
    vq = Fraction(v)
    inside = lo <= vq <= hi
    if lo <= 0 <= hi:
        worst = max(abs(vq - lo), abs(vq - hi))
        best = Fraction(0) if inside else min(abs(vq - lo), abs(vq - hi))
    else:
        rel = [abs(vq - c) / abs(c) for c in (lo, hi)]
        worst = max(rel)
        best = Fraction(0) if inside else min(rel)

    def raw(err: Fraction) -> float:
        return math.inf if err == 0 else -_log10(err)

    return clip_lre(raw(worst), cap), clip_lre(raw(best), cap)


def verdict(v: float, lo: Fraction, hi: Fraction, rel: float, abs_: float) -> str:
    """``correct`` / ``wrong`` / ``undecided`` for ``v`` against the enclosure ``[lo, hi]``.

    Sufficient conditions, so both labels are proofs: ``correct`` when the farthest
    endpoint is within the smallest tolerance over the enclosure, ``wrong`` when the
    distance to the enclosure exceeds the largest tolerance over it.
    """
    if not math.isfinite(v):
        return "wrong"
    vq, r, a = Fraction(v), Fraction(rel), Fraction(abs_)
    min_abs = Fraction(0) if lo <= 0 <= hi else min(abs(lo), abs(hi))
    max_abs = max(abs(lo), abs(hi))
    far = max(abs(vq - lo), abs(vq - hi))
    dist = Fraction(0) if lo <= vq <= hi else min(abs(vq - lo), abs(vq - hi))
    if far <= max(r * min_abs, a):
        return "correct"
    if dist > max(r * max_abs, a):
        return "wrong"
    return "undecided"


def out_of_range(row: Row) -> bool:
    """A quantile-type reference outside the normal binary64 range (not scored)."""
    if row.function in ("cdf", "sf", "pvalue"):
        return False
    lo, hi = abs(row.lo_q), abs(row.hi_q)
    mx, mn = max(lo, hi), min(lo, hi)
    if mx == 0:
        return False
    return mx > DBL_MAX or mn < DBL_MIN


def score(target_name: str, rows: list[Row]) -> list[dict[str, object]]:
    """One result dict per row the target supports."""
    target = get_target(target_name)
    out: list[dict[str, object]] = []
    for row in rows:
        pt = parse_id(row.id)
        if not target.supports(pt):
            continue
        rel, abs_ = target.tolerance(pt)
        rec: dict[str, object] = {
            "id": row.id,
            "family": row.family,
            "function": row.function,
            "kind": row.kind,
            "value": "",
            "status": "ok",
            "lre_lo": "",
            "lre_hi": "",
            "verdict": "",
            "digits": "",
            "rel_tol": rel,
            "abs_tol": abs_,
            "note": "",
        }
        if out_of_range(row):
            rec.update(status="skipped", verdict="out-of-range")
            out.append(rec)
            continue
        try:
            v = target.evaluate(pt)
        except Exception as exc:  # the target refused: recorded, not dropped
            msg = traceback.format_exception_only(type(exc), exc)[-1].strip()
            rec.update(status="error", verdict="error", note=msg[:200])
            out.append(rec)
            continue
        lo, hi = row.lo_q, row.hi_q
        rec["value"] = repr(v)
        if not math.isfinite(v):
            rec["status"] = "nan" if math.isnan(v) else "inf"
        lre_lo, lre_hi = lre_bounds(v, lo, hi)
        rec["lre_lo"], rec["lre_hi"] = _fmt(lre_lo), _fmt(lre_hi)
        vd = verdict(v, lo, hi, rel, abs_)
        if vd == "correct":
            rec["digits"] = int(lre_lo) if math.isfinite(lre_lo) else ""
        if row.kind == "reference":
            vd = {"correct": "ref-agree", "wrong": "ref-disagree"}.get(vd, "ref-undecided")
        if row.function in ("cdf", "sf", "pvalue") and hi < DBL_MIN:
            rec["note"] = "reference below DBL_MIN: scored by verdict only"
        rec["verdict"] = vd
        out.append(rec)
    return out


def _fmt(x: float) -> str:
    return "nan" if math.isnan(x) else f"{x:.2f}"


def write_results(path: Path, recs: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=RESULT_COLUMNS, lineterminator="\n")
        w.writeheader()
        w.writerows(recs)
    tmp.replace(path)


def check_target(name: str, grid_csv: Path, random_csv: Path, out_dir: Path) -> dict:
    """Score ``name`` on the grid and the random draws; write ``<name>-grid.csv``,
    ``<name>-random.csv`` and ``<name>.json`` (versions, counts, seconds)."""
    target = get_target(name)
    try:
        version = target.version()
    except ImportError as exc:
        raise SystemExit(f"target {name!r} is not installed ({exc})") from exc
    summary: dict[str, object] = {
        "target": name,
        "version": version,
        "description": target.description,
    }
    for label, path in (("grid", grid_csv), ("random", random_csv)):
        if not path.exists():
            continue
        t0 = time.monotonic()
        recs = score(name, read_rows(path))
        write_results(out_dir / f"{name}-{label}.csv", recs)
        counts: dict[str, int] = {}
        for r in recs:
            counts[str(r["verdict"])] = counts.get(str(r["verdict"]), 0) + 1
        summary[label] = {
            "rows": len(recs),
            "verdicts": dict(sorted(counts.items())),
            "seconds": round(time.monotonic() - t0, 1),
        }
    (out_dir / f"{name}.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary
