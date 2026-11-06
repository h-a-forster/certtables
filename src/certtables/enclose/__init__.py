"""Enclosures: certified (Arb), exact (rational) and reference (mpmath, not certified)."""

from __future__ import annotations

from fractions import Fraction

from certtables.enclose import continuous, exact, gst
from certtables.enclose.continuous import Enclosure
from certtables.grid import Point

__all__ = ["Enclosure", "enclose_point", "enclose_gst"]


def _exact(value: Fraction, method: str) -> Enclosure:
    return Enclosure(value, value, f"exact:{method}", 0, "exact")


def enclose_point(pt: Point) -> Enclosure:
    """Enclosure for one non-GST point (GST points go through :func:`enclose_gst`)."""
    p = pt.p
    fam, fn = pt.family, pt.function
    if fam in continuous.FAMILIES and fam != "beta":
        params = {k: float(v) for k, v in p.items() if k not in ("x", "p")}
        if fn in ("cdf", "sf"):
            return continuous.value(fam, fn, float(p["x"]), params)
        return continuous.quantile(fam, fn, float(p["p"]), params)
    if fam == "cp":
        return continuous.clopper_pearson(int(p["k"]), int(p["n"]), float(p["alpha"]), fn)
    if fam == "binom":
        f = exact.binom_cdf if fn == "cdf" else exact.binom_sf
        return _exact(f(int(p["k"]), int(p["n"]), float(p["p"])), "binomial-sum")
    if fam == "fisher":
        a, b, c, d = (int(v) for v in str(p["table"]).split(","))
        return _exact(exact.fisher_pvalue(a, b, c, d, str(p["alternative"])), "hypergeometric")
    if fam == "wilcoxon":
        return _exact(
            exact.wilcoxon_pvalue(int(p["n"]), int(p["w"]), str(p["alternative"])),
            "subset-sum-counts",
        )
    if fam == "mannwhitney":
        v = exact.mannwhitney_pvalue(int(p["m"]), int(p["n"]), int(p["u"]), str(p["alternative"]))
        return _exact(v, "gaussian-binomial")
    if fam == "ks":
        f = exact.ks_cdf if fn == "cdf" else exact.ks_sf
        method = "durbin-matrix" if float(p["d"]) < 0.5 else "birnbaum-tingey"
        return _exact(f(int(p["n"]), float(p["d"])), method)
    raise ValueError(f"no enclosure for {pt.id}")


def design_of(pt: Point) -> gst.Design:
    p = pt.p
    return gst.Design(str(p["design"]), float(p["alpha"]), gst.parse_fractions(str(p["t"])))


def enclose_gst(design: gst.Design) -> list[Enclosure]:
    """All K boundaries of a design: certified for K <= 3, mpmath reference above."""
    if design.K <= 3:
        guesses = gst.mp_boundaries(design, degree=5, dps=20)
        return gst.certified_boundaries(design, guesses)
    return gst.reference_boundaries(design)
