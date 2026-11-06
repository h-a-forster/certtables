from __future__ import annotations

import json
from fractions import Fraction

import pytest

from certtables import cli, grid
from certtables.enclose.continuous import Enclosure
from certtables.table import PartFile, build, check_row, file_sha256, make_row, read_rows


def test_grid_ids_roundtrip_and_hash_is_deterministic():
    pts = grid.deterministic_grid()
    assert len({p.id for p in pts}) == len(pts)
    for p in pts[::37]:
        assert grid.parse_id(p.id) == p
    assert grid.grid_hash(pts) == grid.grid_hash(grid.deterministic_grid())


def test_random_draws_are_seeded():
    a = grid.random_points("t", n=50)
    b = grid.random_points("t", n=50)
    c = grid.random_points("t", n=50, seed=1)
    assert [p.id for p in a] == [p.id for p in b] != [p.id for p in c]


def test_row_hash_and_rounding():
    pt = grid.Point("norm", "sf", (("x", 1.0),))
    enc = Enclosure(Fraction(1, 3), Fraction(1, 3), "exact:test", 0, "exact")
    row = make_row(pt, enc)
    check_row(row)
    assert Fraction(row.lo) <= Fraction(1, 3) <= Fraction(row.hi)
    assert row.f64 == repr(1 / 3) and row.exact_sha
    bad = row.__class__(**{**row.as_dict(), "prec_bits": 0, "hi": "0.4"})
    with pytest.raises(ValueError):
        check_row(bad)


def test_build_checkpoints_and_resumes(tmp_path):
    pts = [grid.Point("binom", "cdf", (("n", 10), ("p", 0.5), ("k", k))) for k in range(4)]
    pts.append(grid.Point("ks", "cdf", (("n", 3), ("d", 0.5))))
    out = tmp_path / "t.csv"
    c1 = build(pts[:3], out, tmp_path / "w", log=lambda s: None)
    assert c1["computed"] == 3
    # a torn last line (killed mid-write) is dropped and recomputed
    part = tmp_path / "w" / "binom.csv"
    part.write_text(part.read_text() + "binom.cdf(n=10;p=0.5;k=3),bin", encoding="utf-8")
    c2 = build(pts, out, tmp_path / "w", log=lambda s: None)
    assert c2["skipped"] == 3 and c2["computed"] == 2 and c2["errors"] == 0
    rows = read_rows(out)
    assert [r.id for r in rows] == [p.id for p in pts]
    assert rows[0].lo == "9.765625e-4"  # 1/1024 exactly
    assert len(PartFile(part).done) == 4


def test_build_records_errors(tmp_path):
    pts = [grid.Point("fisher", "pvalue", (("table", "1,2,3,4"), ("alternative", "both")))]
    c = build(pts, tmp_path / "t.csv", tmp_path / "w", log=lambda s: None)
    assert c["errors"] == 1 and c["rows"] == 0
    assert "unknown alternative" in (tmp_path / "w" / "errors.jsonl").read_text()


def test_verify_notices_a_changed_grid(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "data").mkdir()
    pts = [grid.Point("binom", "cdf", (("n", 10), ("p", 0.5), ("k", 1)))]
    build(pts, tmp_path / "data" / "t.csv", tmp_path / "w", log=lambda s: None)
    entry = {"file": "t.csv", "sha256": file_sha256(tmp_path / "data" / "t.csv")}
    entry["grid_hash"] = grid.grid_hash(grid.deterministic_grid())
    manifest = tmp_path / "data" / "MANIFEST.json"
    manifest.write_text(json.dumps({"grid": entry}), encoding="utf-8")
    assert cli.main(["verify"]) == 0
    entry["grid_hash"] = "0" * 64
    manifest.write_text(json.dumps({"grid": entry}), encoding="utf-8")
    assert cli.main(["verify"]) == 1
    assert "declared points differ" in capsys.readouterr().out


def test_committed_rows_are_reproducible_and_hashes_check(tmp_path):
    """Recompute a deterministic sample of committed rows: same digits, same hashes."""
    from pathlib import Path

    committed = Path(__file__).resolve().parents[1] / "data" / "certtables-v1.csv"
    if not committed.exists():
        pytest.skip("table not built")
    rows = read_rows(committed)
    for r in rows:
        check_row(r)
    sample = [r for r in rows if r.family != "gst"][::400]
    pts = [grid.parse_id(r.id) for r in sample]
    build(pts, tmp_path / "t.csv", tmp_path / "w", log=lambda s: None)
    assert read_rows(tmp_path / "t.csv") == sample
