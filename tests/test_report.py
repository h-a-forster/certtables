"""The published numbers: the census path from scored rows to results/."""

from __future__ import annotations

import csv
import math
from fractions import Fraction
from pathlib import Path

from certtables import study
from certtables.enclose.continuous import Enclosure
from certtables.grid import Point
from certtables.report import write_report
from certtables.table import make_row, write_rows
from certtables.targets.base import Target

ROOT = Path(__file__).resolve().parents[1]
HALF = Fraction(1, 2)
TINY = Fraction(1, 10**320)
HUGE = Fraction(10**400)


def _norm(fn, name, value):
    return Point("norm", fn, ((name, value),))


def _gst(look):
    return Point(
        "gst", "bound", (("design", "pocock"), ("alpha", 0.05), ("t", "1/2,1"), ("look", look))
    )


# (point, enclosure value, kind, what the stub target returns)
CASES = [
    (_norm("cdf", "x", 0.0), HALF, "exact", 0.5),  # correct
    (_norm("cdf", "x", 1.0), HALF, "exact", 0.75),  # wrong, finite
    (_norm("sf", "x", 2.0), HALF, "exact", math.nan),  # wrong, no LRE
    (_norm("sf", "x", 3.0), HALF, "exact", RuntimeError("refused")),
    (_norm("isf", "p", 1e-300), HUGE, "exact", None),  # never evaluated
    (_norm("sf", "x", 39.0), TINY, "exact", 0.0),  # correct through the DBL_MIN floor
    (_gst(1), Fraction(2), "reference", 2.0),
    (_gst(2), Fraction(2), "reference", 3.0),
]


class Stub(Target):
    name = "stub"
    description = "stub"

    def supports(self, pt: Point) -> bool:
        return True

    def evaluate(self, pt: Point) -> float:
        v = next(c[3] for c in CASES if c[0].id == pt.id)
        if isinstance(v, Exception):
            raise v
        assert v is not None, "an out-of-range reference must not be evaluated"
        return v


def _census(path: Path) -> dict[str, dict[str, str]]:
    with open(path, newline="", encoding="utf-8") as fh:
        return {r["family"]: r for r in csv.DictReader(fh)}


def test_scored_rows_reach_the_census(tmp_path, monkeypatch):
    monkeypatch.setattr(study, "get_target", lambda name: Stub())
    rows = [make_row(p, Enclosure(c, c, "test", 0, k)) for p, c, k, _ in CASES]
    data, targets = tmp_path / "data", tmp_path / "results" / "targets"
    write_rows(data / "certtables-v1.csv", rows)
    write_rows(data / "certtables-v1-random.csv", rows)

    study.check_target(
        "stub", data / "certtables-v1.csv", data / "certtables-v1-random.csv", targets
    )
    write_report(targets, tmp_path / "results", data / "MANIFEST.json")

    grid = _census(tmp_path / "results" / "census-grid.csv")
    norm, gst = grid["norm"], grid["gst"]
    assert (norm["points"], norm["scored"], norm["correct"], norm["wrong"]) == ("6", "4", "2", "2")
    assert (norm["error"], norm["out_of_range"], norm["frac_wrong"]) == ("1", "1", "0.5000")
    # The DBL_MIN row has no LRE; the nan LRE counts as below every threshold.
    assert (norm["lre_points"], norm["lre_lt_6"], norm["min_lre_lo"]) == ("3", "2", "nan")
    assert (gst["scored"], gst["ref_agree"], gst["ref_disagree"]) == ("0", "1", "1")

    rand = _census(tmp_path / "results" / "census-random.csv")["norm"]
    # Clopper-Pearson 2 of 4: (0.06759, 0.93241), rounded outward.
    assert (rand["scored"], rand["wrong"], rand["rate"]) == ("4", "2", "0.5000")
    assert (rand["ci_lo"], rand["ci_hi"]) == ("0.0675", "0.9325")

    with open(tmp_path / "results" / "wrong.csv", newline="", encoding="utf-8") as fh:
        wrong = list(csv.DictReader(fh))
    assert {r["id"] for r in wrong if r["set"] == "grid"} == {
        CASES[1][0].id,
        CASES[2][0].id,
        CASES[7][0].id,
    }
    assert all(r["lo"] for r in wrong)


def test_committed_census_is_what_the_committed_scores_give(tmp_path, monkeypatch):
    """REPORT.md's tables come from these files; any change to the census shows up here."""
    monkeypatch.chdir(tmp_path)  # paths must follow the manifest, not the working directory
    out = tmp_path / "out"
    paths = write_report(ROOT / "results" / "targets", out, ROOT / "data" / "MANIFEST.json")
    assert len(paths) == 4
    for p in paths:
        assert p.read_text(encoding="utf-8") == (ROOT / "results" / p.name).read_text(
            encoding="utf-8"
        ), p.name
