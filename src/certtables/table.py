"""The reference table: CSV rows, hashing, and the checkpointed build
(docs/DESIGN.md section 3.3).

Columns of ``data/certtables-v1.csv``:

- ``id``: point id (:mod:`certtables.grid`); ``family``, ``function``;
- ``kind``: ``certified`` (Arb enclosure), ``exact`` (rational, no rounding) or
  ``reference`` (mpmath, NOT certified);
- ``method``, ``prec_bits``: how it was computed (Arb working precision, 0 for exact);
- ``lo``, ``hi``: decimal strings, rounded outward to :data:`DIGITS` significant digits, so
  the true value lies in ``[lo, hi]``;
- ``f64``: the correctly rounded binary64 value when both ends of the enclosure, before the
  decimal rounding of ``lo`` and ``hi``, round to the same double (else empty); ``inf`` or
  ``-inf`` beyond the double range;
- ``exact_sha``: for ``exact`` rows, SHA-256 (first 16 hex) of ``numerator/denominator``;
- ``sha``: SHA-256 (first 16 hex) of ``id|kind|method|prec_bits|lo|hi``.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import os
import time
import traceback
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

from certtables.enclose import Enclosure, design_of, enclose_gst, enclose_point
from certtables.grid import Point, parse_id

DIGITS = 25
COLUMNS = ("id", "family", "function", "kind", "method", "prec_bits", "lo", "hi", "f64",
           "exact_sha", "sha")  # fmt: skip


@dataclass(frozen=True)
class Row:
    id: str
    family: str
    function: str
    kind: str
    method: str
    prec_bits: int
    lo: str
    hi: str
    f64: str
    exact_sha: str
    sha: str

    @property
    def lo_q(self) -> Fraction:
        return Fraction(self.lo)

    @property
    def hi_q(self) -> Fraction:
        return Fraction(self.hi)

    def as_dict(self) -> dict[str, str]:
        return {c: str(getattr(self, c)) for c in COLUMNS}


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def nearest_double(x: Fraction) -> float:
    """Correctly rounded binary64 value of a rational (Python's Fraction.__float__ rounds
    correctly: it divides integers with round-half-even); overflow gives +-inf."""
    try:
        return float(x)
    except OverflowError:
        return math.inf if x > 0 else -math.inf


def make_row(pt: Point, enc: Enclosure) -> Row:
    from certtables.enclose.arbutil import to_decimal

    if enc.lo > enc.hi:
        raise ValueError(f"{pt.id}: lo > hi")
    lo = to_decimal(enc.lo, DIGITS, "down")
    hi = to_decimal(enc.hi, DIGITS, "up")
    f_lo, f_hi = nearest_double(enc.lo), nearest_double(enc.hi)
    f64 = repr(f_lo) if f_lo == f_hi else ""
    exact_sha = ""
    if enc.kind == "exact":
        if enc.lo != enc.hi:
            raise ValueError(f"{pt.id}: exact enclosure with lo != hi")
        exact_sha = _sha(f"{enc.lo.numerator:x}/{enc.lo.denominator:x}")
    sha = _sha(f"{pt.id}|{enc.kind}|{enc.method}|{enc.prec}|{lo}|{hi}")
    return Row(
        pt.id, pt.family, pt.function, enc.kind, enc.method, enc.prec, lo, hi, f64, exact_sha, sha
    )


def check_row(row: Row) -> None:
    """Raise if a row is internally inconsistent (hash, ordering, id)."""
    parse_id(row.id)
    want = _sha(f"{row.id}|{row.kind}|{row.method}|{row.prec_bits}|{row.lo}|{row.hi}")
    if row.sha != want:
        raise ValueError(f"{row.id}: sha mismatch")
    if row.lo_q > row.hi_q:
        raise ValueError(f"{row.id}: lo > hi")


def read_rows(path: str | Path) -> list[Row]:
    with open(path, newline="", encoding="utf-8") as fh:
        rows = []
        for d in csv.DictReader(fh):
            d["prec_bits"] = int(d["prec_bits"])
            rows.append(Row(**d))
    return rows


def _row_line(row: Row) -> str:
    buf = io.StringIO()
    csv.writer(buf, lineterminator="\n").writerow([getattr(row, c) for c in COLUMNS])
    return buf.getvalue()


def write_rows(path: str | Path, rows: Iterable[Row]) -> None:
    """Write a complete CSV atomically (temp file + replace)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", newline="", encoding="utf-8") as fh:
        fh.write(",".join(COLUMNS) + "\n")
        for r in rows:
            fh.write(_row_line(r))
    os.replace(tmp, path)


# ------------------------------------------------------------------------------ build


class PartFile:
    """Append-only checkpoint of finished rows for one family (one CSV per family)."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.done: dict[str, Row] = {}
        if path.exists():
            # A torn last line (process killed mid-write) is dropped and recomputed.
            text = path.read_text(encoding="utf-8")
            lines = text.splitlines(keepends=True)
            if lines and not lines[-1].endswith("\n"):
                lines = lines[:-1]
                path.write_text("".join(lines), encoding="utf-8")
            for d in csv.DictReader(io.StringIO("".join(lines))):
                d["prec_bits"] = int(d["prec_bits"])
                row = Row(**d)
                check_row(row)
                self.done[row.id] = row
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(",".join(COLUMNS) + "\n", encoding="utf-8")

    def append(self, row: Row) -> None:
        with open(self.path, "a", newline="", encoding="utf-8") as fh:
            fh.write(_row_line(row))
            fh.flush()
            os.fsync(fh.fileno())
        self.done[row.id] = row


def build(
    points: list[Point],
    out_csv: str | Path,
    work_dir: str | Path,
    *,
    log: Callable[[str], None] = print,
    max_minutes: float | None = None,
) -> dict[str, int]:
    """Enclose every point, checkpointing per family in ``work_dir`` (rows appended as they
    finish, finished ids skipped on rerun, ``progress.log``), then write ``out_csv`` in grid
    order. Points whose enclosure fails are logged to ``errors.jsonl`` and left out. Returns
    counts. Stops early (counts["stopped"] = 1) after ``max_minutes``; rerun to resume."""
    work = Path(work_dir)
    work.mkdir(parents=True, exist_ok=True)
    progress = work / "progress.log"
    errors_path = work / "errors.jsonl"
    errors_path.write_text("", encoding="utf-8")  # failed points are retried every run
    seen = {p.id for p in points}
    if len(seen) != len(points):
        raise ValueError("duplicate point ids in the grid")
    parts: dict[str, PartFile] = {}
    t_start = time.monotonic()
    counts = {"points": len(points), "computed": 0, "skipped": 0, "errors": 0, "stopped": 0}

    def plog(msg: str) -> None:
        line = f"{time.strftime('%Y-%m-%dT%H:%M:%S')} {msg}"
        with open(progress, "a", encoding="utf-8") as fh:
            fh.write(line + "\n")
        log(line)

    def error(pid: str, exc: BaseException) -> None:
        counts["errors"] += 1
        rec = {
            "id": pid,
            "error": f"{type(exc).__name__}: {exc}",
            "where": traceback.format_exception_only(type(exc), exc)[-1].strip(),
        }
        with open(errors_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec) + "\n")

    def part(family: str) -> PartFile:
        if family not in parts:
            parts[family] = PartFile(work / f"{family}.csv")
        return parts[family]

    gst_done: set[tuple] = set()
    gst_failed: dict[tuple, Exception] = {}  # every look of a failed design is an error
    last_report = time.monotonic()
    for i, pt in enumerate(points):
        pf = part(pt.family)
        if pt.id in pf.done:
            counts["skipped"] += 1
            continue
        if max_minutes is not None and time.monotonic() - t_start > 60 * max_minutes:
            counts["stopped"] = 1
            plog(f"stopping at the time limit after {counts['computed']} new rows")
            break
        try:
            if pt.family == "gst":
                d = design_of(pt)
                key = (d.kind, d.alpha, d.t)
                if key in gst_failed:
                    raise gst_failed[key]
                if key in gst_done:
                    continue
                gst_done.add(key)
                t0 = time.monotonic()
                sched = pt.p["t"]
                design = (("design", d.kind), ("alpha", d.alpha), ("t", sched))
                try:  # all rows before any append, so a design is never left half written
                    rows = [
                        make_row(Point("gst", "bound", (*design, ("look", look))), enc)
                        for look, enc in enumerate(enclose_gst(d), start=1)
                    ]
                except Exception as exc:
                    gst_failed[key] = exc
                    raise
                for row in rows:
                    if row.id in seen and row.id not in pf.done:
                        pf.append(row)
                        counts["computed"] += 1
                plog(f"gst {d.kind} alpha={d.alpha} t={sched}: {time.monotonic() - t0:.1f}s")
            else:
                pf.append(make_row(pt, enclose_point(pt)))
                counts["computed"] += 1
        except Exception as exc:  # recorded and counted, never silently dropped
            error(pt.id, exc)
        if time.monotonic() - last_report > 30:
            last_report = time.monotonic()
            plog(
                f"{i + 1}/{len(points)} points visited, {counts['computed']} new rows, "
                f"{counts['errors']} errors"
            )
    rows_by_id: dict[str, Row] = {}
    for pf in parts.values():
        rows_by_id.update(pf.done)
    for pt in points:  # families never visited in a stopped run
        if pt.family not in parts and (work / f"{pt.family}.csv").exists():
            rows_by_id.update(part(pt.family).done)
    ordered = [rows_by_id[p.id] for p in points if p.id in rows_by_id]
    counts["rows"] = len(ordered)
    if not counts["stopped"]:
        write_rows(out_csv, ordered)
        plog(f"wrote {len(ordered)} rows to {Path(out_csv).name} ({counts['errors']} errors)")
    return counts


def file_sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()
