"""README.md and REPORT.md publish copies of rows rendered from committed results.

`certtables report` writes results/published.md; CI reruns it and fails on a diff. A row in a
document must start with the cells of a published row; cells after them are prose.
"""

from pathlib import Path

from certtables import published

ROOT = Path(__file__).resolve().parents[1]
PUBLISHED = ROOT / "results" / "published.md"
SETTINGS = ("| Target | Relative tolerance |",)  # declared tolerances, not results


def _cells(line: str) -> list[str]:
    return [c.strip() for c in line.strip().strip("|").split("|")]


def _rows(path: Path) -> list[str]:
    rows, skipping = [], False
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.startswith("|"):
            skipping = False
        elif line.startswith(SETTINGS):
            skipping = True
        elif line.startswith("| ") and not skipping:
            rows.append(line)
    return rows


def test_published_tables_render_from_the_committed_results():
    assert PUBLISHED.read_text(encoding="utf-8") == published.render(ROOT)


def test_every_table_row_starts_with_a_published_row():
    source = [_cells(line) for line in _rows(PUBLISHED)]
    for doc in ("README.md", "REPORT.md"):
        rows = _rows(ROOT / doc)
        assert rows, doc
        unmatched = [row for row in rows if not any(_cells(row)[: len(s)] == s for s in source)]
        assert unmatched == [], doc


def test_figures_quoted_in_prose_are_the_rendered_ones():
    text = " ".join(
        " ".join((ROOT / doc).read_text(encoding="utf-8").split())
        for doc in ("README.md", "REPORT.md")
    )
    figures = published.prose_figures(ROOT)
    assert [f for f in figures if f not in text] == []
    single = PUBLISHED.read_text(encoding="utf-8").split("Single precision (grid, random): ")[1]
    assert "Single precision (grid, random): " + single.splitlines()[0] in text
