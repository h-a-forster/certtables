"""Command line: ``certtables build | check | report | verify | repro | env``."""

from __future__ import annotations

import argparse
import contextlib
import json
import platform
import sys
import time
from pathlib import Path

from certtables import __version__

DATA = Path("data")
GRID_CSV = DATA / "certtables-v1.csv"
RANDOM_CSV = DATA / "certtables-v1-random.csv"
MANIFEST = DATA / "MANIFEST.json"
RESULTS = Path("results")
WORK = Path("runs") / "build"


def _versions() -> dict[str, str]:
    from importlib.metadata import PackageNotFoundError, version

    out = {"python": platform.python_version(), "certtables": __version__}
    for pkg in (
        "python-flint",
        "mpmath",
        "scipy",
        "numpy",
        "statsmodels",
        "spotify-confidence",
        "jax",
    ):
        with contextlib.suppress(PackageNotFoundError):
            out[pkg] = version(pkg)
    return out


def cmd_build(args: argparse.Namespace) -> int:
    from certtables import grid
    from certtables.table import build, file_sha256

    sets = {"grid": GRID_CSV, "random": RANDOM_CSV}
    which = ["grid", "random"] if args.set == "all" else [args.set]
    updates: dict[str, dict] = {}
    status = 0
    start = time.monotonic()
    for name in which:
        points = _points(name)
        if args.families:
            keep = set(args.families.split(","))
            unknown = keep - set(grid.FAMILIES)
            if unknown:
                raise SystemExit(f"unknown families: {sorted(unknown)}")
            points = [p for p in points if p.family in keep]
        out = sets[name] if not args.families else Path(args.out_dir) / f"{name}.csv"
        left = args.max_minutes  # one budget for the whole call, not one per set
        if left is not None:
            left -= (time.monotonic() - start) / 60
        counts = build(points, out, WORK / name, max_minutes=left)
        print(json.dumps({name: counts}))
        if counts["stopped"]:
            status = 3
            continue
        if not args.families:
            errors = [
                json.loads(line)
                for line in (WORK / name / "errors.jsonl").read_text(encoding="utf-8").splitlines()
            ]
            updates[name] = {
                "file": out.name,
                "sha256": file_sha256(out),
                "rows": counts["rows"],
                "grid_points": counts["points"],
                "grid_hash": grid.grid_hash(points),
                "not_enclosed": {e["id"]: e["error"] for e in errors},
            }
            if name == "random":
                updates[name]["seed"] = grid.RANDOM_SEED
                updates[name]["per_family"] = grid.RANDOM_PER_FAMILY
    if updates:
        # Re-read just before writing so separate `--set grid` / `--set random` runs merge.
        manifest = json.loads(MANIFEST.read_text(encoding="utf-8")) if MANIFEST.exists() else {}
        manifest.update(updates)
        manifest["built_with"] = {
            k: v
            for k, v in _versions().items()
            if k in ("python", "certtables", "python-flint", "mpmath")
        }
        MANIFEST.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return status


def _points(name: str) -> list:
    from certtables import grid

    if name == "grid":
        return grid.deterministic_grid()
    return [p for fam in grid.RANDOM_FAMILIES for p in grid.random_points(fam)]


def cmd_verify(args: argparse.Namespace) -> int:
    """Check row hashes, ordering, the manifest's file and grid hashes, and that every declared
    point has a row or a not_enclosed entry."""
    from certtables.grid import grid_hash
    from certtables.table import check_row, file_sha256, read_rows

    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    bad = 0
    for name in ("grid", "random"):
        if name not in manifest:
            print(f"MANIFEST.json has no {name!r} entry")
            bad += 1
            continue
        entry = manifest[name]
        path = DATA / entry["file"]
        if file_sha256(path) != entry["sha256"]:
            print(f"{path.name}: file sha256 differs from MANIFEST.json")
            bad += 1
        points = _points(name)
        if grid_hash(points) != entry["grid_hash"]:
            print(f"{path.name}: the declared points differ from MANIFEST.json")
            bad += 1
        rows = read_rows(path)
        covered = [r.id for r in rows] + list(entry["not_enclosed"])
        if sorted(covered) != sorted(p.id for p in points):
            print(f"{path.name}: rows and not_enclosed do not partition the declared points")
            bad += 1
        for r in rows:
            try:
                check_row(r)
            except ValueError as exc:
                print(exc)
                bad += 1
        print(f"{path.name}: {len(rows)} rows checked")
    return 1 if bad else 0


def cmd_check(args: argparse.Namespace) -> int:
    from certtables.study import check_target
    from certtables.targets import get_target

    names = args.target.split(",")
    for target in names:
        get_target(target)  # refuse a bad name before any target has been scored
    for target in names:
        summary = check_target(target, GRID_CSV, RANDOM_CSV, RESULTS / "targets")
        print(json.dumps(summary))
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    from certtables.report import write_report

    paths = write_report(RESULTS / "targets", RESULTS, MANIFEST)
    for p in paths:
        print(p)
    return 0


def cmd_repro(args: argparse.Namespace) -> int:
    """Evaluate one table row on one target and print the comparison."""
    from certtables.grid import parse_id
    from certtables.study import lre_bounds, verdict
    from certtables.table import read_rows
    from certtables.targets import get_target

    row = next(
        (r for path in (GRID_CSV, RANDOM_CSV) for r in read_rows(path) if r.id == args.id), None
    )
    if row is None:
        raise SystemExit(f"no row {args.id!r} in {GRID_CSV} or {RANDOM_CSV}")
    target = get_target(args.target)
    pt = parse_id(row.id)
    if not target.supports(pt):
        raise SystemExit(f"target {args.target!r} does not support {row.id!r}")
    value = target.evaluate(pt)
    rel, abs_ = target.tolerance(pt)
    out = {
        "id": row.id,
        "target": args.target,
        "version": target.version(),
        "value": repr(value),
        "lo": row.lo,
        "hi": row.hi,
        "kind": row.kind,
        "rel_tol": rel,
        "abs_tol": abs_,
        "verdict": verdict(value, row.lo_q, row.hi_q, rel, abs_),
        "lre": lre_bounds(value, row.lo_q, row.hi_q),
    }
    print(json.dumps(out, indent=2))
    return 0


def cmd_env(args: argparse.Namespace) -> int:
    print(json.dumps(_versions(), indent=2))
    return 0


def main(argv: list[str] | None = None) -> int:
    from certtables.targets import TARGETS

    ap = argparse.ArgumentParser(prog="certtables", description=__doc__)
    ap.add_argument("--version", action="version", version=f"certtables {__version__}")
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build", help="enclose the grid and/or the random draws (resumable)")
    b.add_argument("--set", choices=("grid", "random", "all"), default="all")
    b.add_argument("--families", help="comma-separated subset (writes to --out-dir instead)")
    b.add_argument("--out-dir", default="tmp/build")
    b.add_argument(
        "--max-minutes", type=float, help="stop after this long (exit 3); rerun to resume"
    )
    b.set_defaults(func=cmd_build)
    c = sub.add_parser("check", help="score targets against the table")
    c.add_argument(
        "--target",
        required=True,
        help=f"comma-separated, from: {', '.join(sorted(TARGETS))}",
    )
    c.set_defaults(func=cmd_check)
    r = sub.add_parser("report", help="write the census tables under results/")
    r.set_defaults(func=cmd_report)
    v = sub.add_parser("verify", help="check row hashes and MANIFEST.json")
    v.set_defaults(func=cmd_verify)
    p = sub.add_parser("repro", help="evaluate one row on one target (see results/wrong.csv)")
    p.add_argument("id", help="row id, for example 'f.isf(dfn=10.0;dfd=10.0;p=1e-20)'")
    p.add_argument("--target", default="scipy")
    p.set_defaults(func=cmd_repro)
    e = sub.add_parser("env", help="print package versions")
    e.set_defaults(func=cmd_env)
    args = ap.parse_args(argv)
    if args.cmd in ("check", "report", "verify", "repro") and not DATA.is_dir():
        raise SystemExit(f"no {DATA}/ here: run certtables from the repository root")
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
