"""Slow checks (run with --runslow): rebuild the grid and compare with the committed table
row for row. Three tests, so each stays under the 15-minute limit for one job."""

from __future__ import annotations

from pathlib import Path

import pytest

from certtables import grid
from certtables.table import build, read_rows

ROOT = Path(__file__).resolve().parents[1]


def _part(pt: grid.Point) -> str:
    if pt.family != "gst":
        return "not-gst"
    return "gst-certified" if str(pt.p["t"]).count(",") < 3 else "gst-reference"


@pytest.mark.slow
@pytest.mark.parametrize("part", ["not-gst", "gst-certified", "gst-reference"])
def test_full_rebuild_reproduces_committed_table(part, tmp_path):
    committed = ROOT / "data" / "certtables-v1.csv"
    if not committed.exists():
        pytest.skip("table not built")
    points = [p for p in grid.deterministic_grid() if _part(p) == part]
    counts = build(points, tmp_path / "t.csv", tmp_path / "w", log=lambda s: None)
    assert counts["stopped"] == 0
    ids = {p.id for p in points}
    assert read_rows(tmp_path / "t.csv") == [r for r in read_rows(committed) if r.id in ids]
