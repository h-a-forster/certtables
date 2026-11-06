"""Exact conversions between Arb balls, dyadic rationals and decimal strings, and the
certified root finder used for quantiles and boundary constants (docs/DESIGN.md section 2).

Every enclosure is a pair of exact rationals ``lo <= hi``. Endpoints read from an Arb ball are
``mid - rad`` and ``mid + rad`` computed exactly in :class:`fractions.Fraction`; decimal
strings are rounded outward, so ``[Fraction(lo_str), Fraction(hi_str)]`` contains
``[lo, hi]``.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass
from fractions import Fraction

from flint import arb, ctx, fmpz

BASE_PREC = 128
MAX_PREC = 1024
#: Bits kept in the mantissa of a bracket point (points are exact dyadic rationals).
POINT_BITS = 120


class EnclosureError(RuntimeError):
    """No finite enclosure could be produced within the precision limit."""


class RootOutOfRange(EnclosureError):
    """A certified sign proves the root lies beyond the search limit ``x_max`` or ``x_min``."""


@contextmanager
def precision(bits: int):
    """Set Arb's working precision for the block (thread-unsafe, like ``ctx.prec``)."""
    with ctx.workprec(bits):
        yield


def arb_to_fraction(x: arb) -> tuple[Fraction, Fraction]:
    """Exact rational endpoints ``(mid - rad, mid + rad)`` of a finite ball."""
    if not x.is_finite():
        raise EnclosureError(f"ball is not finite: {x}")
    m, e = x.mid().man_exp()
    r, f = x.rad().man_exp()
    mid = Fraction(int(m)) * (Fraction(2) ** int(e))
    rad = Fraction(int(r)) * (Fraction(2) ** int(f))
    return mid - rad, mid + rad


def dyadic(x: Fraction | float | int, bits: int = POINT_BITS) -> Fraction:
    """``x`` rounded to the nearest rational ``m * 2**e`` with ``|m| < 2**bits``."""
    x = Fraction(x)
    if x == 0:
        return Fraction(0)
    sign = -1 if x < 0 else 1
    a = abs(x)
    e = a.numerator.bit_length() - a.denominator.bit_length() - bits
    scaled = a / (Fraction(2) ** e)
    m = round(scaled)
    return sign * m * (Fraction(2) ** e)


def fraction_to_arb(x: Fraction) -> arb:
    """Exact Arb value of a dyadic rational (raises if ``x`` is not dyadic)."""
    x = Fraction(x)
    d = x.denominator
    if d & (d - 1):
        raise ValueError(f"not a dyadic rational: {x}")
    return arb((fmpz(x.numerator), -(d.bit_length() - 1)))


def ball(lo: Fraction, hi: Fraction) -> arb:
    """An Arb ball containing the rational interval ``[lo, hi]`` (``lo``, ``hi`` arbitrary)."""
    lo, hi = Fraction(lo), Fraction(hi)
    if lo > hi:
        raise ValueError("lo > hi")
    mid = dyadic((lo + hi) / 2, bits=max(ctx.prec, 64))
    rad = max(hi - mid, mid - lo)
    if rad == 0:
        return fraction_to_arb(mid)
    # Round the radius up to a dyadic number so the ball covers [lo, hi].
    r = dyadic(rad * (1 + Fraction(1, 2**25)), bits=30)
    while r < rad:
        r *= 2
    return arb(fraction_to_arb(mid), fraction_to_arb(r))


def rational_ball(x: Fraction) -> arb:
    """An Arb ball containing the rational ``x`` (exact when ``x`` is dyadic)."""
    x = Fraction(x)
    d = x.denominator
    if not d & (d - 1):
        return fraction_to_arb(x)
    return arb(x.numerator) / arb(d)


def to_decimal(x: Fraction, digits: int, direction: str) -> str:
    """``x`` as a decimal string with ``digits`` significant digits, rounded toward
    ``-inf`` (``direction='down'``) or ``+inf`` (``'up'``)."""
    if direction not in ("down", "up"):
        raise ValueError(f"direction must be 'down' or 'up', got {direction!r}")
    x = Fraction(x)
    if x == 0:
        return "0"
    neg = x < 0
    a = -x if neg else x
    # Decimal exponent e with 10**e <= a < 10**(e+1), from bit lengths, then fixed exactly.
    e = math.floor((a.numerator.bit_length() - a.denominator.bit_length()) * math.log10(2))
    while Fraction(10) ** e > a:
        e -= 1
    while Fraction(10) ** (e + 1) <= a:
        e += 1
    scaled = a * Fraction(10) ** (digits - 1 - e)
    toward_zero = (direction == "down") != neg
    n = scaled.numerator // scaled.denominator
    if not toward_zero and n != scaled:
        n += 1
    if n == 10**digits:  # rounding up carried into a new digit
        n //= 10
        e += 1
    s = str(n)
    mant = s[0] + ("." + s[1:].rstrip("0") if s[1:].rstrip("0") else "")
    return f"{'-' if neg else ''}{mant}e{e}"


@dataclass(frozen=True)
class Bracket:
    """A certified bracket ``lo < root < hi`` (or ``lo == hi == root`` when exact)."""

    lo: Fraction
    hi: Fraction
    prec: int
    evals: int


Evaluator = Callable[[arb], arb]


def _ilog2(x: Fraction) -> int:
    """floor(log2(x)) up to one, for x > 0 (used only to pick bisection points)."""
    return x.numerator.bit_length() - x.denominator.bit_length()


class _Signer:
    """Certified sign of ``s * G(x)`` with escalating precision; counts evaluations."""

    def __init__(self, G: Evaluator, s: int, base_prec: int, max_prec: int) -> None:
        self.G, self.s, self.base, self.max = G, s, base_prec, max_prec
        self.prec_used = base_prec
        self.evals = 0
        self.cache: dict[Fraction, tuple[int, arb]] = {}

    def __call__(self, x: Fraction) -> int:
        """+1, -1, 0 (undecided at the precision limit) or 2 (``G(x)`` is exactly zero).

        An undecided ball is re-evaluated at twice the precision, up to ``max_prec``, while
        that keeps shrinking it.
        """
        if x in self.cache:
            return self.cache[x][0]
        prec = self.base
        sign, val, last_rad = 0, None, None
        while prec <= self.max:
            with precision(prec):
                self.evals += 1
                val = self.G(fraction_to_arb(x)) * self.s
            self.prec_used = max(self.prec_used, prec)
            if val.is_finite():
                if val.is_exact() and val.is_zero():
                    sign = 2
                    break
                if val > 0:
                    sign = 1
                    break
                if val < 0:
                    sign = -1
                    break
                # Stop escalating when doubling the precision did not shrink the ball by
                # at least 2^(prec/4): its radius then comes from the inputs, not rounding.
                rad = val.rad()
                if last_rad is not None and not rad < last_rad * arb(2) ** (-(prec // 8)):
                    break
                last_rad = rad
            prec *= 2
        self.cache[x] = (sign, val if val is not None else arb(0))
        return sign

    def mid(self, x: Fraction) -> Fraction:
        """Midpoint of the cached ball at ``x`` as an exact rational (0 if unknown)."""
        val = self.cache.get(x, (0, None))[1]
        if val is None or not val.is_finite():
            return Fraction(0)
        m, e = val.mid().man_exp()
        return Fraction(int(m)) * Fraction(2) ** int(e)


def certified_root(
    G: Evaluator,
    guess: float | Fraction,
    *,
    increasing: bool,
    rel_width: float = 1e-20,
    lower: Fraction | None = None,
    base_prec: int = BASE_PREC,
    max_prec: int = MAX_PREC,
    max_steps: int = 400,
    x_max: Fraction | None = None,
    x_min: Fraction | None = None,
) -> Bracket:
    """Enclose the unique root of a strictly monotone ``G`` by a certified sign change.

    ``G`` maps an exact Arb point to a ball containing ``G(x)``. A point ``x`` is certified
    left of the root when the ball of ``s*G(x)`` is entirely negative and right of it when
    entirely positive (``s = +1`` if ``G`` increases, ``-1`` if it decreases); by strict
    monotonicity the root lies strictly between such points. ``lower`` is an exclusive lower
    bound of the domain (for example 0 for positive quantiles). The guess only affects speed.
    ``x_max`` stops the search for a sign change at that magnitude (for example just
    outside the binary64 range); ``x_min`` stops the geometric search towards ``lower`` once
    ``x - lower`` falls below it.

    Returns the narrowest bracket found: relative width at most ``rel_width`` unless the
    precision limit is reached first (the caller records the achieved width).
    """
    sg = _Signer(G, 1 if increasing else -1, base_prec, max_prec)
    lb = None if lower is None else Fraction(lower)
    g = dyadic(Fraction(guess))
    if lb is not None and g <= lb:
        g = dyadic(lb + 1) if lb < 0 else dyadic(max(lb * 2, Fraction(1)))

    def done(lo: Fraction, hi: Fraction) -> bool:
        scale = max(abs(lo), abs(hi))
        return hi - lo <= Fraction(rel_width) * scale

    def beyond(limit: Fraction, wanted: int, what: str) -> EnclosureError:
        """The error for a search stopped at ``limit``: ``RootOutOfRange`` when the certified
        sign there still puts the root on the far side (``wanted`` is that sign)."""
        if sg(limit) == wanted:
            return RootOutOfRange(f"root certified {what}")
        return EnclosureError(f"no certified sign change {what}")

    sgn = sg(g)
    if sgn == 2:
        return Bracket(g, g, sg.prec_used, sg.evals)
    lo = g if sgn == -1 else None
    hi = g if sgn == 1 else None
    unit = abs(g) if g != 0 else Fraction(1)
    step = unit * Fraction(2) ** -48
    # Expand outward until each side is certified: additive steps growing 4x, and below
    # the domain's lower bound geometric steps towards it.
    for direction in (-1, 1):
        k = j = 0
        while (lo if direction < 0 else hi) is None:
            if k + j >= max_steps:
                raise EnclosureError("no certified sign change found")
            # Steps grow 4x, then 2^16x after 30 steps so any magnitude is reached quickly.
            growth = Fraction(4) ** min(k, 30) * Fraction(2**16) ** max(k - 30, 0)
            x = g + direction * step * growth
            if x_max is not None and abs(x) > x_max:
                raise beyond(direction * x_max, -direction, "beyond |x| = x_max")
            if direction < 0 and lb is not None and x <= lb:
                x = lb + (g - lb) * Fraction(2) ** -(2 ** min(j, 40))
                j += 1
                if x_min is not None and x - lb < x_min:
                    x = lb + x_min  # the last point: the root may lie just above the limit
                    if sg(x) != -1:
                        raise beyond(x, 1, "below lower + x_min")
            else:
                k += 1
            x = dyadic(x)
            s = sg(x)
            if s == 2:
                return Bracket(x, x, sg.prec_used, sg.evals)
            if s == -1 and (lo is None or x > lo):
                lo = x
            elif s == 1 and (hi is None or x < hi):
                hi = x
    if lo is None or hi is None:
        raise EnclosureError("no certified sign change found")

    # Narrow: geometric bisection while the bracket spans orders of magnitude, then
    # Illinois steps (secant on ball midpoints) with a bisection fallback.
    side = same = 0
    flo, fhi = sg.mid(lo), sg.mid(hi)
    for _ in range(max_steps):
        if done(lo, hi):
            break
        w = hi - lo
        if lo > 0 and hi > 4 * lo:
            c = dyadic(Fraction(2) ** ((_ilog2(lo) + _ilog2(hi)) // 2))
        elif hi < 0 and lo < 4 * hi:
            c = -dyadic(Fraction(2) ** ((_ilog2(-lo) + _ilog2(-hi)) // 2))
        elif flo < 0 < fhi and same < 3:
            c = dyadic(lo - flo * w / (fhi - flo))
            if not lo + w / 1024 < c < hi - w / 1024:
                c = dyadic((lo + hi) / 2)
        else:
            c = dyadic((lo + hi) / 2)
        if not lo < c < hi:
            c = dyadic((lo + hi) / 2)
        if not lo < c < hi:  # the dyadic grid is exhausted at this resolution
            break
        s = sg(c)
        if s == 2:
            return Bracket(c, c, sg.prec_used, sg.evals)
        if s in (-1, 1):
            same = same + 1 if s == side else 0
            if s == -1:
                lo, flo = c, sg.mid(c)
                if side == -1:
                    fhi /= 2  # Illinois: damp the stale end
            else:
                hi, fhi = c, sg.mid(c)
                if side == 1:
                    flo /= 2
            side = s
        else:
            # c is within the resolution of G at the root: probe windows around c growing
            # 4x from a quarter of the target width; keep the closest certified points.
            tw = Fraction(rel_width) * max(abs(lo), abs(hi)) / 4
            improved = False
            for direction in (-1, 1):
                j = 0
                while True:
                    x = dyadic(c + direction * tw * Fraction(4) ** j)
                    if not lo < x < hi:
                        break
                    sx = sg(x)
                    if sx == 2:
                        return Bracket(x, x, sg.prec_used, sg.evals)
                    if sx == direction:
                        if direction < 0:
                            lo, flo = x, sg.mid(x)
                        else:
                            hi, fhi = x, sg.mid(x)
                        improved = True
                        break
                    j += 1
            if not improved:
                break
    return Bracket(lo, hi, sg.prec_used, sg.evals)
