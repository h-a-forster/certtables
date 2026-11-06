from __future__ import annotations

import json
import time
from fractions import Fraction

import pytest

from certtables import cli, grid, table
from certtables.enclose.continuous import Enclosure
from certtables.table import PartFile, build, check_row, file_sha256, make_row, read_rows
from certtables.targets import TARGETS


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


def test_a_failed_design_reports_every_look(tmp_path, monkeypatch):
    def fail(design):
        raise RuntimeError("no convergence")

    monkeypatch.setattr(table, "enclose_gst", fail)
    pts = [
        grid.Point(
            "gst", "bound", (("design", "pocock"), ("alpha", 0.05), ("t", "1/2,1"), ("look", k))
        )
        for k in (1, 2)
    ]
    c = build(pts, tmp_path / "t.csv", tmp_path / "w", log=lambda s: None)
    assert c["rows"] == 0 and c["errors"] == 2
    errors = (tmp_path / "w" / "errors.jsonl").read_text(encoding="utf-8").splitlines()
    assert [json.loads(line)["id"] for line in errors] == [p.id for p in pts]


def test_max_minutes_is_one_budget_across_sets(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "data").mkdir()
    pts = {
        name: [grid.Point("norm", "cdf", (("x", x),)) for x in (i, i + 0.5)]
        for i, name in enumerate(("grid", "random"))
    }
    monkeypatch.setattr(cli, "_points", pts.get)

    def slow(pt):
        time.sleep(0.2)
        return Enclosure(Fraction(1), Fraction(1), "test", 0, "exact")

    monkeypatch.setattr(table, "enclose_point", slow)
    assert cli.main(["build", "--max-minutes", "0.001"]) == 3
    assert len(PartFile(tmp_path / "runs" / "build" / "grid" / "norm.csv").done) == 1
    assert not PartFile(tmp_path / "runs" / "build" / "random" / "norm.csv").done


def test_check_refuses_a_bad_target_before_scoring_any(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "MANIFEST.json").write_text("{}", encoding="utf-8")
    (tmp_path / "results" / "targets").mkdir(parents=True)
    with pytest.raises(SystemExit, match="unknown target"):
        cli.main(["check", "--target", "printed,nope"])
    assert not list((tmp_path / "results" / "targets").iterdir())
    with pytest.raises(SystemExit):
        cli.main(["check", "--help"])
    out = capsys.readouterr().out
    assert all(name in out for name in TARGETS)


def test_commands_refuse_to_run_outside_the_repository(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    with pytest.raises(SystemExit, match="repository root"):
        cli.main(["report"])
    assert not list(tmp_path.iterdir())


def test_verify_notices_a_changed_grid_or_table(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "data").mkdir()
    pts = [grid.Point("binom", "cdf", (("n", 10), ("p", 0.5), ("k", k))) for k in (1, 2)]
    monkeypatch.setattr(cli, "_points", lambda name: pts)
    build(pts[:1], tmp_path / "data" / "t.csv", tmp_path / "w", log=lambda s: None)
    entry = {
        "file": "t.csv",
        "sha256": file_sha256(tmp_path / "data" / "t.csv"),
        "grid_hash": grid.grid_hash(pts),
        "not_enclosed": {pts[1].id: "RootOutOfRange"},
    }
    manifest = tmp_path / "data" / "MANIFEST.json"

    def verify(**changes):
        doc = {"grid": {**entry, **changes}, "random": entry}
        manifest.write_text(json.dumps(doc), encoding="utf-8")
        status = cli.main(["verify"])
        return status, capsys.readouterr().out

    assert verify()[0] == 0
    status, out = verify(grid_hash="0" * 64)
    assert status == 1 and "declared points differ" in out
    # a point with neither a row nor a recorded failure
    status, out = verify(not_enclosed={})
    assert status == 1 and "do not partition" in out
    # a manifest missing an entry must not pass by checking nothing
    manifest.write_text("{}", encoding="utf-8")
    assert cli.main(["verify"]) == 1


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
    sample += [r for r in rows if "design=pocock;alpha=0.05;t=1/2,1;" in r.id]  # K = 2 is quick
    pts = [grid.parse_id(r.id) for r in sample]
    build(pts, tmp_path / "t.csv", tmp_path / "w", log=lambda s: None)
    assert read_rows(tmp_path / "t.csv") == sample
