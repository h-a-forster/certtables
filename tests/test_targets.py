"""Adapters: recorded outputs replay, and the constructed test inputs carry the intended
statistic (so a disagreement is the target's, not the adapter's)."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from certtables import cli
from certtables.enclose import enclose_point
from certtables.enclose import exact as E
from certtables.grid import Point, parse_id
from certtables.table import make_row, write_rows
from certtables.targets import TARGETS, get_target
from certtables.targets.base import mannwhitney_samples, wilcoxon_differences

RECORDED = json.loads((Path(__file__).parent / "data" / "recorded.json").read_text())


@pytest.mark.parametrize("name", sorted(RECORDED))
def test_recorded_outputs_replay(name):
    target = get_target(name)
    try:
        version = target.version()
    except ImportError:
        pytest.skip(f"{name} not installed")
    rec = RECORDED[name]
    if version != rec["version"]:
        pytest.skip(f"{name} {version} differs from recorded {rec['version']}")
    for pid, value in rec["values"].items():
        pt = parse_id(pid)
        assert target.supports(pt)
        assert repr(target.evaluate(pt)) == value, pid


@pytest.mark.parametrize(("n", "w"), [(5, 0), (5, 7), (10, 13), (10, 55), (12, 40)])
def test_wilcoxon_inputs_and_scipy_definition(n, w):
    stats = pytest.importorskip("scipy.stats")
    d = wilcoxon_differences(n, w)
    assert sorted(abs(x) for x in d) == list(range(1, n + 1))
    assert sum(abs(x) for x in d if x > 0) == w
    for alt in ("greater", "less", "two-sided"):
        got = stats.wilcoxon(d, alternative=alt, method="exact").pvalue
        assert math.isclose(got, float(E.wilcoxon_pvalue(n, w, alt)), rel_tol=1e-13)


@pytest.mark.parametrize(("m", "n", "u"), [(3, 4, 0), (3, 4, 5), (5, 10, 12), (10, 10, 100)])
def test_mannwhitney_inputs_and_scipy_definition(m, n, u):
    stats = pytest.importorskip("scipy.stats")
    x, y = mannwhitney_samples(m, n, u)
    assert len(set(x) | set(y)) == m + n
    assert sum(1 for a in x for b in y if a > b) == u
    for alt in ("greater", "less", "two-sided"):
        res = stats.mannwhitneyu(x, y, alternative=alt, method="exact")
        assert res.statistic == u
        assert math.isclose(res.pvalue, float(E.mannwhitney_pvalue(m, n, u, alt)), rel_tol=1e-13)


def test_every_target_declares_tolerance_and_support():
    pt = Point("norm", "cdf", (("x", 1.0),))
    for name, cls in TARGETS.items():
        t = cls()
        assert t.name == name and t.description
        rel, abs_ = t.tolerance(pt)
        assert rel >= 0 and abs_ >= 0
    assert get_target("scipy").supports(pt)
    assert not get_target("spotify").supports(pt)
    with pytest.raises(SystemExit):
        get_target("nope")


def test_mpmath_time_limit_stops_a_long_call_and_the_target_recovers(monkeypatch):
    pytest.importorskip("mpmath")
    monkeypatch.setattr("certtables.targets.mpmath_target.TIME_LIMIT", 4.0)
    target = get_target("mpmath")
    # Without the limit this point runs for minutes.
    with pytest.raises(TimeoutError):
        target.evaluate(parse_id("f.cdf(dfn=100000.0;dfd=100000.0;x=0.5)"))
    assert 0.1 < target.evaluate(parse_id("norm.cdf(x=-1.0)")) < 0.2


def test_mpmath_adapter_keeps_the_small_tail():
    pytest.importorskip("mpmath")
    target = get_target("mpmath")
    cdf = target.evaluate(parse_id("t.cdf(df=5.0;x=-100000.0)"))
    sf = target.evaluate(parse_id("t.sf(df=5.0;x=100000.0)"))
    assert 0 < sf == cdf


def test_repro_prints_the_row_and_the_verdict(tmp_path, monkeypatch, capsys):
    pytest.importorskip("scipy")
    monkeypatch.chdir(tmp_path)
    (tmp_path / "data").mkdir()
    pt = parse_id("norm.sf(x=1.0)")
    write_rows(tmp_path / "data" / "certtables-v1.csv", [make_row(pt, enclose_point(pt))])
    (tmp_path / "data" / "certtables-v1-random.csv").write_text("id\n", encoding="utf-8")
    assert cli.main(["repro", pt.id]) == 0
    assert json.loads(capsys.readouterr().out)["verdict"] == "correct"
    with pytest.raises(SystemExit):
        cli.main(["repro", "norm.sf(x=2.0)"])
