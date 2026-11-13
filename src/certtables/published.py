"""The tables README.md and REPORT.md publish, rendered from committed results.

`certtables report` writes them to `results/published.md`. A published row may have fewer
columns than the copy in a document: the copy must start with the row's cells, and any further
cells (a status, a defect described in words) are prose. `tests/test_published.py` holds
README.md and REPORT.md to this file, and CI rebuilds it and fails on a diff.

Sources: `results/census-*.csv` (the census `certtables report` writes),
`results/upstream-checks.json` (`scripts/upstream_checks.py`), the table CSVs under `data/` and
`studies/libs-2026-11-09/data/census.md` (that study's own scoring, which needs the four
libraries and is not rerun here).
"""

from __future__ import annotations

import csv
import re
from collections import Counter
from pathlib import Path

TARGETS = {
    "scipy": "SciPy",
    "statsmodels": "statsmodels",
    "jax": "JAX",
    "mpmath": "mpmath (15 digits)",
    "spotify": "spotify-confidence",
    "printed": "printed",
}
FAMILIES = {
    "norm": "normal",
    "t": "t",
    "chi2": "chi-square",
    "f": "F",
    "cp": "Clopper-Pearson",
    "binom": "binomial",
    "fisher": "Fisher",
    "wilcoxon": "Wilcoxon",
    "mannwhitney": "Mann-Whitney",
    "ks": "Kolmogorov-Smirnov",
    "gst": "boundaries, certified",
}
# README: one row per target and group of families.
README_ROWS = [
    ("scipy", ["norm"]),
    ("scipy", ["t"]),
    ("scipy", ["chi2"]),
    ("scipy", ["f"]),
    ("scipy", ["cp"]),
    ("scipy", ["binom", "fisher", "wilcoxon", "mannwhitney"]),
    ("scipy", ["ks"]),
    ("statsmodels", ["cp", "binom"]),
    ("jax", ["norm"]),
    ("jax", ["chi2"]),
    ("spotify", ["gst"]),
]
README_TARGETS = {**TARGETS, "spotify": "spotify-confidence"}
README_FAMILY = {"gst": "certified boundaries"}
LIBRARIES = [
    ("torch", "PyTorch"),
    ("jax", "JAX"),
    ("tfp", "TFP"),
    ("distributionsjl", "Distributions.jl"),
]


def _read(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _row(cells: list[str]) -> str:
    return "| " + " | ".join(cells) + " |"


def _table(header: list[str], rows: list[list[str]]) -> list[str]:
    return [_row(header), "|" + "---|" * len(header), *(_row(r) for r in rows), ""]


def _order(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    targets, families = list(TARGETS), list(FAMILIES)
    return sorted(rows, key=lambda r: (targets.index(r["target"]), families.index(r["family"])))


def _join(names: list[str]) -> str:
    return names[0] if len(names) == 1 else ", ".join(names[:-1]) + " and " + names[-1]


def _pct(x: str) -> str:
    return f"{100 * float(x):.2f}"


def census_tables(results: Path) -> list[str]:
    grid = {(r["target"], r["family"]): r for r in _read(results / "census-grid.csv")}
    rand = {(r["target"], r["family"]): r for r in _read(results / "census-random.csv")}
    out = ["## README: grid and random draws", ""]
    body = []
    for target, fams in README_ROWS:
        wrong = sum(int(grid[target, f]["wrong"]) for f in fams)
        scored = sum(int(grid[target, f]["scored"]) for f in fams)
        drawn = [rand[target, f] for f in fams if (target, f) in rand]
        if len(drawn) > 1:
            raise ValueError(f"{target}: a README row can show one random family, not {len(drawn)}")
        shown = (
            f"{_pct(drawn[0]['rate'])}% ({_pct(drawn[0]['ci_lo'])} to {_pct(drawn[0]['ci_hi'])})"
            if drawn
            else "not drawn"
        )
        names = [README_FAMILY.get(f, FAMILIES[f]) for f in fams]
        family = ", ".join(names) if len(fams) > 2 else _join(names).replace(" and ", ", ")
        body.append([README_TARGETS[target], family, f"{wrong} / {scored}", shown])
    out += _table(
        ["Target", "Family", "Grid: wrong / scored", "Random draws: failure rate (95% CI)"], body
    )

    out += ["## REPORT: deterministic grid", ""]
    body = [
        [TARGETS[r["target"]], FAMILIES[r["family"]], r["scored"], r["wrong"], r["undecided"],
         r["lre_points"], r["lre_lt_6"], r["lre_lt_10"], r["lre_lt_13"]]
        for r in _order(list(grid.values()))
    ]  # fmt: skip
    out += _table(
        ["Target", "Family", "Scored", "Wrong", "Undecided", "LRE pts", "LRE < 6", "LRE < 10",
         "LRE < 13"], body)  # fmt: skip

    out += ["## REPORT: random draws", ""]
    body = [
        [TARGETS[r["target"]], FAMILIES[r["family"]], r["scored"], r["wrong"], r["rate"],
         f"{r['ci_lo']} to {r['ci_hi']}"]
        for r in _order(list(rand.values()))
    ]  # fmt: skip
    out += _table(["Target", "Family", "Scored", "Wrong", "Rate", "95% CI"], body)
    return out


def kinds_table(data: Path) -> list[str]:
    kinds: Counter[str] = Counter()
    for name in ("certtables-v1.csv", "certtables-v1-random.csv"):
        kinds.update(r["kind"] for r in _read(data / name))
    out = ["## README: kinds of row", ""]
    body = [[f"`{k}`", f"{kinds[k]:,}"] for k in ("exact", "certified", "reference")]
    if sum(kinds.values()) != sum(int(n.replace(",", "")) for _, n in body):
        raise ValueError(f"rows of an unknown kind: {sorted(kinds)}")
    return out + _table(["Kind", "Rows"], body)


def causes_table(results: Path) -> list[str]:
    import json

    checks = json.loads((results / "upstream-checks.json").read_text(encoding="utf-8"))
    kinds = checks["t_quantile"]["kinds"]
    lo, hi = checks["t_quantile"]["factor_range"]
    other = checks["other_wrong"]
    ppf = sum(n for k, n in other.items() if k.startswith("f.ppf/"))
    chi2 = sum(n for k, n in other.items() if k.startswith("chi2."))
    ks = sum(n for k, n in other.items() if k.startswith("ks."))
    if ppf + chi2 + ks != sum(other.values()):
        raise ValueError(f"a cause the table does not name: {sorted(other)}")
    rows = [
        ("`f.isf` is `ppf(1 - q)`", checks["f_isf"]["wrong"]),
        ("`t` quantile: stuck near `sqrt(df / DBL_MIN)`",
         kinds["finite, stuck near sqrt(df / DBL_MIN)"]),
        ("`t` quantile: infinite with the wrong sign", kinds["infinite, wrong sign"]),
        (f"`t` quantile: finite, off by a factor of {lo:.1f} to {hi:.1f}",
         kinds["finite, off by a factor below 3"]),
        ("`f.ppf` `nan` or lost digits at tiny `p`", ppf),
        (f"Marginal or edge cases (chi-square {chi2}, Kolmogorov-Smirnov {ks})", chi2 + ks),
    ]  # fmt: skip
    total = sum(n for _, n in rows)
    out = ["## REPORT: root causes of SciPy's wrong points", ""]
    body = [[cause, str(n), f"{round(100 * n / total)}%"] for cause, n in rows]
    return out + _table(["Cause", "Points", f"Share of {total}"], body)


def _versions(claims: Path) -> dict[str, str]:
    """`torch 2.15.0+cpu` -> `2.15.0+cpu`; TFP also names its TensorFlow."""
    out = {}
    for r in _read(claims):
        numbers = re.findall(r"\d+(?:\.\d+)+(?:\+\w+)?", r["version"])
        tf = f", TF {numbers[1]}" if r["library"] == "tfp" else ""
        out[r["library"]] = numbers[0] + tf
    return out


def libraries_table(census: Path) -> list[str]:
    """Wrong / scored per library in double precision, summed over every scored API."""
    versions = _versions(census.parent / "claims.csv")
    sums: dict[tuple[str, str, str], list[int]] = {}
    section = None
    for line in census.read_text(encoding="utf-8").splitlines():
        if line.startswith("#"):
            heading = re.match(r"### (\S+) (f32|f64)$", line)
            section = heading.groups() if heading else None  # "ext" APIs are outside table v1
            continue
        cells = [c.strip() for c in line.strip("|").split("|")]
        if section and line.startswith("| ") and cells[2] in ("grid", "random"):
            total = sums.setdefault((*section, cells[2]), [0, 0])
            total[0] += int(cells[4])
            total[1] += int(cells[3])
    out = ["## REPORT: four more libraries", ""]
    body = []
    for key, name in LIBRARIES:
        grid, rand = sums[key, "f64", "grid"], sums[key, "f64", "random"]
        body.append([name, versions[key], f"{grid[0]} / {grid[1]:,}", f"{rand[0]} / {rand[1]:,}"])
    out += _table(["Library", "Version", "Double: grid wrong", "Double: random wrong"], body)
    single = []
    for key, name in LIBRARIES:
        if (key, "f32", "grid") in sums:
            g, r = sums[key, "f32", "grid"], sums[key, "f32", "random"]
            single.append(f"{name} {g[0]} / {g[1]:,} and {r[0]} / {r[1]:,}")
    out += ["Single precision (grid, random): " + ", ".join(single) + ".", ""]
    return out


def prose_figures(root: Path) -> list[str]:
    """Figures the README and REPORT summaries state in sentences, in their words."""
    import json

    results, data = root / "results", root / "data"
    grid = {(r["target"], r["family"]): r for r in _read(results / "census-grid.csv")}
    rand = {(r["target"], r["family"]): r for r in _read(results / "census-random.csv")}
    manifest = json.loads((data / "MANIFEST.json").read_text(encoding="utf-8"))
    rows = [r for n in ("certtables-v1.csv", "certtables-v1-random.csv") for r in _read(data / n)]
    kinds = Counter(r["kind"] for r in rows)
    below = sum(r["method"].endswith(";below-2^-1100") for r in rows)
    left_out = {s: manifest[s]["grid_points"] - manifest[s]["rows"] for s in ("grid", "random")}
    scipy = [r for (t, _), r in grid.items() if t == "scipy"]

    def total(field: str, rs=scipy) -> int:
        return sum(int(r[field]) for r in rs)

    clean = [r for r in scipy if r["wrong"] == "0"]
    wrong = total("wrong")
    if any(r["undecided"] != "0" for r in grid.values()):
        raise ValueError("a scored point came out undecided")

    def rate(key: tuple[str, str], digits: int) -> str:
        r = rand[key]
        lo, hi = 100 * float(r["ci_lo"]), 100 * float(r["ci_hi"])
        return f"{100 * float(r['rate']):.{digits}f}%", f"({lo:.{digits}f} to {hi:.{digits}f})"

    f_rate, f_ci = rate(("scipy", "f"), 1)
    t_rate, t_ci = rate(("scipy", "t"), 1)
    chi_rate, chi_ci = rate(("scipy", "chi2"), 2)
    jax_rate, jax_ci = rate(("jax", "chi2"), 1)
    zero = [rand["scipy", f] for f in ("norm", "cp")]
    if {r["wrong"] for r in zero} != {"0"} or len({r["scored"] for r in zero}) != 1:
        raise ValueError("SciPy normal and Clopper-Pearson draws are no longer 0 of the same n")
    isf = json.loads((results / "upstream-checks.json").read_text(encoding="utf-8"))["f_isf"]
    if isf["candidate_breaks_correct_rows"]:
        raise ValueError("the f.isf candidate fix now breaks correct rows")
    sm = [grid["statsmodels", f] for f in ("cp", "binom")]
    sm_rand = rand["statsmodels", "cp"]
    if total("wrong", sm) or sm_rand["wrong"] != "0":
        raise ValueError("statsmodels is no longer clean")
    return [
        f"{len(rows):,} rows: {kinds['certified']:,} certified Arb enclosures, "
        f"{kinds['exact']:,} exact rationals and {kinds['reference']}",
        f"({below} rows are instead the bound `[0, 2^-1100]`)",
        f"hold {len(rows):,} rows",
        f"{sum(left_out.values())} quantiles ({left_out['grid']} grid, "
        f"{left_out['random']} random)",
        f"{sum(left_out.values())} quantiles lie outside",
        f"On the deterministic grid of {total('scored'):,} scored points",
        f"no certified-wrong point ({total('scored', clean):,} points). {wrong} points were wrong "
        f"({100 * wrong / total('scored'):.1f}%), and of the {total('lre_points'):,} points with a "
        f"normal-range reference, {total('lre_lt_6')} had fewer than 6 correct digits, "
        f"{total('lre_lt_10')} fewer than 10 and {total('lre_lt_13')} fewer than 13.",
        f"`F` failed at **{f_rate}** {f_ci[:-1]}, 95% Clopper-Pearson) and `t` at **{t_rate}** "
        f"{t_ci}; chi-square at {chi_rate} {chi_ci}, normal and Clopper-Pearson at 0 of "
        f"{int(zero[0]['scored']):,} each.",
        f"{isf['wrong']} of {isf['scored']} points are wrong, {isf['values']['inf']} of them "
        f"`inf`. A one-function change fixes {isf['candidate_fixes']} of those and breaks none of "
        f"the {isf['scored'] - isf['wrong']} correct ones.",
        f"0 wrong of {total('scored', sm):,} grid and {int(sm_rand['scored']):,} random points",
        f"{jax_rate} {jax_ci} on chi-square, all with at least "
        f"{float(grid['jax', 'chi2']['min_lre_lo']):.2f} digits right",
    ]


def render(root: Path) -> str:
    results = root / "results"
    out = [
        "# Published tables",
        "",
        "Generated by `certtables report`; README.md and REPORT.md copy these rows (a copy may add",
        "cells of prose after them). Do not edit by hand.",
        "",
    ]
    out += census_tables(results)
    out += kinds_table(root / "data")
    out += causes_table(results)
    out += libraries_table(root / "studies" / "libs-2026-11-09" / "data" / "census.md")
    out += ["## Figures quoted in prose", "", *(f"- {f}" for f in prose_figures(root)), ""]
    return "\n".join(out).rstrip("\n") + "\n"
