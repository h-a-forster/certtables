"""Census tables from the per-target results (docs/DESIGN.md section 4.3).

Writes under ``results/``:

- ``census-grid.csv``: exact census per target x family on the declared grid (no
  intervals: the grid is the population);
- ``census-random.csv``: failure rate per target x family on the random draws, with
  Clopper-Pearson 95% intervals (from this package's own certified beta quantiles);
- ``wrong.csv``: every certified-wrong point with the target value and the enclosure;
- ``census.md``: the same tables in Markdown (the layout follows strdcheck's REPORT tables).
"""

from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path

from certtables.grid import FAMILIES
from certtables.table import read_rows

THRESHOLDS = (6, 10, 13)
GRID_FIELDS = ("target", "family", "points", "scored", "correct", "wrong", "undecided",
               "error", "out_of_range", "frac_wrong", "lre_points", *[f"lre_lt_{k}" for k in
               THRESHOLDS], "lre_ambiguous", "min_lre_lo", "ref_agree", "ref_disagree",
               "ref_undecided")  # fmt: skip
RANDOM_FIELDS = ("target", "family", "scored", "wrong", "rate", "ci_lo", "ci_hi", "error",
                 "out_of_range")  # fmt: skip


def _read(path: Path) -> list[dict[str, str]]:
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _f(s: str) -> float:
    return float(s) if s not in ("", "nan") else math.nan


def grid_census(target: str, recs: list[dict[str, str]]) -> list[dict[str, object]]:
    by_fam: dict[str, list[dict[str, str]]] = defaultdict(list)
    for r in recs:
        by_fam[r["family"]].append(r)
    out = []
    for fam in [f for f in FAMILIES if f in by_fam]:
        rs = by_fam[fam]
        cert = [r for r in rs if r["kind"] != "reference"]
        v = defaultdict(int)
        for r in rs:
            v[r["verdict"]] += 1
        scored = [r for r in cert if r["verdict"] in ("correct", "wrong", "undecided")]
        # LRE census: certified/exact rows with a reference in the normal range.
        lre_rows = [r for r in scored if not r["note"]]
        row: dict[str, object] = {
            "target": target,
            "family": fam,
            "points": len(rs),
            "scored": len(scored),
            "correct": v["correct"],
            "wrong": v["wrong"],
            "undecided": v["undecided"],
            "error": sum(1 for r in cert if r["verdict"] == "error"),
            "out_of_range": v["out-of-range"],
            "frac_wrong": f"{v['wrong'] / len(scored):.4f}" if scored else "",
            "lre_points": len(lre_rows),
        }
        amb = 0
        for k in THRESHOLDS:
            lo_below = [r for r in lre_rows if not _f(r["lre_lo"]) >= k]  # nan counts
            hi_below = [r for r in lre_rows if not _f(r["lre_hi"]) >= k]
            row[f"lre_lt_{k}"] = len(lo_below)
            amb += len(lo_below) - len(hi_below)
        row["lre_ambiguous"] = amb
        los = [_f(r["lre_lo"]) for r in lre_rows]
        row["min_lre_lo"] = (
            "nan" if any(math.isnan(x) for x in los) else (f"{min(los):.2f}" if los else "")
        )
        row["ref_agree"] = v["ref-agree"]
        row["ref_disagree"] = v["ref-disagree"]
        row["ref_undecided"] = v["ref-undecided"]
        out.append(row)
    return out


def random_census(target: str, recs: list[dict[str, str]]) -> list[dict[str, object]]:
    from certtables.enclose.continuous import clopper_pearson

    by_fam: dict[str, list[dict[str, str]]] = defaultdict(list)
    for r in recs:
        by_fam[r["family"]].append(r)
    out = []
    for fam in [f for f in FAMILIES if f in by_fam]:
        rs = by_fam[fam]
        scored = [r for r in rs if r["verdict"] in ("correct", "wrong", "undecided")]
        n, k = len(scored), sum(1 for r in scored if r["verdict"] == "wrong")
        row: dict[str, object] = {
            "target": target,
            "family": fam,
            "scored": n,
            "wrong": k,
            "error": sum(1 for r in rs if r["verdict"] == "error"),
            "out_of_range": sum(1 for r in rs if r["verdict"] == "out-of-range"),
        }
        if n:
            lo = clopper_pearson(k, n, 0.05, "lower")
            hi = clopper_pearson(k, n, 0.05, "upper")
            # Outward to 4 decimals: floor the lower bound, ceil the upper.
            row.update(
                rate=f"{k / n:.4f}",
                ci_lo=f"{math.floor(float(lo.lo) * 1e4) / 1e4:.4f}",
                ci_hi=f"{math.ceil(float(hi.hi) * 1e4) / 1e4:.4f}",
            )
        out.append(row)
    return out


def _write_csv(path: Path, rows: list[dict[str, object]], fields) -> Path:
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, lineterminator="\n", extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    return path


def _md_table(rows: list[dict[str, object]], cols: list[tuple[str, str]]) -> str:
    head = "| " + " | ".join(h for _, h in cols) + " |"
    sep = "|" + "|".join("---" for _ in cols) + "|"
    body = ["| " + " | ".join(str(r.get(k, "")) for k, _ in cols) + " |" for r in rows]
    return "\n".join([head, sep, *body])


def write_report(targets_dir: Path, out_dir: Path, manifest_path: Path) -> list[Path]:
    targets = sorted(p.stem for p in targets_dir.glob("*.json"))
    grid_rows: list[dict[str, object]] = []
    rand_rows: list[dict[str, object]] = []
    wrong: list[dict[str, object]] = []
    versions = {}
    tables = {}
    for name in ("grid", "random"):
        path = Path("data") / f"certtables-v1{'' if name == 'grid' else '-random'}.csv"
        if path.exists():
            tables[name] = {r.id: r for r in read_rows(path)}
    for t in targets:
        meta = json.loads((targets_dir / f"{t}.json").read_text(encoding="utf-8"))
        versions[t] = meta.get("version", "")
        for label in ("grid", "random"):
            p = targets_dir / f"{t}-{label}.csv"
            if not p.exists():
                continue
            recs = _read(p)
            if label == "grid":
                grid_rows += grid_census(t, recs)
            else:
                rand_rows += random_census(t, recs)
            for r in recs:
                if r["verdict"] in ("wrong", "ref-disagree"):
                    ref = tables.get(label, {}).get(r["id"])
                    wrong.append(
                        {
                            "target": t,
                            "set": label,
                            "id": r["id"],
                            "verdict": r["verdict"],
                            "value": r["value"],
                            "lo": ref.lo if ref else "",
                            "hi": ref.hi if ref else "",
                            "lre_lo": r["lre_lo"],
                            "lre_hi": r["lre_hi"],
                            "rel_tol": r["rel_tol"],
                            "abs_tol": r["abs_tol"],
                        }
                    )
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = [
        _write_csv(out_dir / "census-grid.csv", grid_rows, GRID_FIELDS),
        _write_csv(out_dir / "census-random.csv", rand_rows, RANDOM_FIELDS),
        _write_csv(
            out_dir / "wrong.csv",
            wrong,
            (
                "target",
                "set",
                "id",
                "verdict",
                "value",
                "lo",
                "hi",
                "lre_lo",
                "lre_hi",
                "rel_tol",
                "abs_tol",
            ),
        ),
    ]
    manifest = (
        json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    )
    md = [
        "# certtables census",
        "",
        "Generated by `certtables report` from `results/targets/`. Do not edit by hand.",
        "",
        "## Versions",
        "",
        _md_table(
            [{"t": t, "v": v} for t, v in versions.items()], [("t", "target"), ("v", "version")]
        ),
        "",
    ]
    if "grid" in manifest:
        md += [
            f"Grid: {manifest['grid']['rows']} rows, grid hash "
            f"`{manifest['grid']['grid_hash'][:16]}`, file sha256 "
            f"`{manifest['grid']['sha256'][:16]}`.",
            "",
        ]
    md += [
        "## Deterministic grid (exact census)",
        "",
        _md_table(
            grid_rows,
            [
                ("target", "target"),
                ("family", "family"),
                ("scored", "scored"),
                ("correct", "correct"),
                ("wrong", "wrong"),
                ("undecided", "undecided"),
                ("error", "error"),
                ("out_of_range", "out of range"),
                ("lre_points", "LRE pts"),
                ("lre_lt_6", "LRE<6"),
                ("lre_lt_10", "LRE<10"),
                ("lre_lt_13", "LRE<13"),
                ("min_lre_lo", "min LRE"),
                ("ref_agree", "ref agree"),
                ("ref_disagree", "ref disagree"),
            ],
        ),
        "",
        "## Random draws (failure rate, Clopper-Pearson 95%)",
        "",
        _md_table(
            rand_rows,
            [
                ("target", "target"),
                ("family", "family"),
                ("scored", "scored"),
                ("wrong", "wrong"),
                ("rate", "rate"),
                ("ci_lo", "CI low"),
                ("ci_hi", "CI high"),
                ("error", "error"),
            ],
        ),
        "",
        f"Certified-wrong and reference-disagreeing points: {len(wrong)} (`results/wrong.csv`).",
        "",
    ]
    md_path = out_dir / "census.md"
    md_path.write_text("\n".join(md), encoding="utf-8")
    paths.append(md_path)
    return paths
