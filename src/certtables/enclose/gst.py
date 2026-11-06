"""Two-sided group-sequential boundaries (docs/DESIGN.md section 2.6).

Under H0 the standardised statistics Z_k = W(t_k)/sqrt(t_k) at information fractions
t_1 < ... < t_K = 1 come from a standard Brownian motion W. A two-sided design rejects at the
first look with |Z_k| >= c_k. The probability of never rejecting is computed with the
Armitage-McPherson-Rowe recursion: the sub-density of W(t_k) on the continuation region is
carried from look to look by Gaussian convolution.

Certified (K <= 3): the continuation probability is an Arb enclosure from
``acb.integral`` (rigorous Gauss-Legendre with error bounds from the integrand's bound on
complex ellipses; the integrands are entire). K = 3 nests one ``acb.integral`` inside the
integrand of another; ball arithmetic makes the inner result enclose the inner integral for
every point of the (complex) ball it is evaluated on, which is what the outer error bound
needs. Constants are then enclosed by a certified sign change (monotonicity in the constant).

Reference (any K, NOT certified): the same recursion with fixed Gauss-Legendre rules in
mpmath; the error is estimated by comparing two rule sizes.

Designs (alpha is the two-sided level; spending functions are per side with a = alpha/2, so
the two-sided spend by look k is 2 a(t_k), the gsDesign/ldbounds convention):

- ``pocock``: c_k = C for all k, equally spaced looks (Pocock 1977).
- ``obf``: c_k = C sqrt(K/k), equally spaced looks (O'Brien and Fleming 1979).
- ``ld-obf``: Lan-DeMets O'Brien-Fleming-type spending a(t) = 2 - 2 Phi(z_{1-a/2}/sqrt t).
- ``ld-pocock``: Lan-DeMets Pocock-type spending a(t) = a log(1 + (e - 1) t).
- ``ld-power``: Kim-DeMets power family a(t) = a t^rho (rho = 2 in spotify-confidence).

Information fractions are exact rationals (``k/K`` or a schedule such as ``0.3,1`` read as
decimals).
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction

from flint import acb, arb

from certtables.enclose.arbutil import (
    EnclosureError,
    arb_to_fraction,
    ball,
    certified_root,
    precision,
    rational_ball,
)
from certtables.enclose.continuous import Enclosure, norm_cdf, norm_sf

DESIGNS = ("pocock", "obf", "ld-obf", "ld-pocock", "ld-power")
#: Working precision of the nested K = 3 integrals (their radius is about 1e-17).
PREC_K3 = 64
PREC_K2 = 128
#: mpmath working precision of the uncertified K >= 4 references.
REF_DPS = 20
#: Relative width at which a certified constant counts as tight enough.
GST_REL_WIDTH = {1: 1e-20, 2: 1e-20, 3: 2e-12}


@dataclass(frozen=True)
class Design:
    """A two-sided design: ``kind`` in :data:`DESIGNS`, level ``alpha``, fractions ``t``."""

    kind: str
    alpha: float
    t: tuple[Fraction, ...]
    rho: float = 2.0

    def __post_init__(self) -> None:
        if self.kind not in DESIGNS:
            raise ValueError(f"unknown design {self.kind!r}")
        if not (0.0 < self.alpha < 1.0):
            raise ValueError(f"alpha must be in (0, 1), got {self.alpha!r}")
        t = self.t
        if not t or t[-1] != 1 or t[0] <= 0 or any(b <= a for a, b in zip(t, t[1:], strict=False)):
            raise ValueError(f"fractions must increase strictly from > 0 to 1, got {t}")
        if self.kind in ("pocock", "obf") and t != equal_fractions(len(t)):
            raise ValueError(f"{self.kind} is defined for equally spaced looks")
        if not self.rho > 0:
            raise ValueError(f"rho must be positive, got {self.rho!r}")

    @property
    def K(self) -> int:
        return len(self.t)


def equal_fractions(K: int) -> tuple[Fraction, ...]:
    return tuple(Fraction(k, K) for k in range(1, K + 1))


def parse_fractions(s: str) -> tuple[Fraction, ...]:
    return tuple(Fraction(v) for v in s.split(","))


# ------------------------------------------------------------------------- certified, K <= 3


def _Phi(y):
    return (-y / arb(2).sqrt()).erfc() / 2


def _phi(y):
    return (-(y * y) / 2).exp() / (2 * arb.pi()).sqrt()


def cont_prob(c: list[arb], t: list[arb]) -> arb:
    """Enclosure of P(|Z_k| < c_k for every k) for K = len(c) <= 3 (balls in, ball out)."""
    K = len(c)
    if K == 1:
        return (c[0] / arb(2).sqrt()).erf()
    if K == 2:
        c1, c2 = c
        t1, t2 = t
        s2, b2, g = (t2 - t1).sqrt(), c2 * t2.sqrt(), c1 * t1.sqrt()

        def f(x, _analytic):
            return _phi(c1 * x) * (_Phi((b2 - g * x) / s2) - _Phi((-b2 - g * x) / s2))

        return (2 * c1 * acb.integral(f, 0, 1)).real
    if K == 3:
        c1, c2, c3 = c
        t1, t2, t3 = t
        s2, s3 = (t2 - t1).sqrt(), (t3 - t2).sqrt()
        b1, b2, b3 = c1 * t1.sqrt(), c2 * t2.sqrt(), c3 * t3.sqrt()

        def inner(u):
            def g(y, _analytic):
                v = b2 * y
                return _phi((v - u) / s2) * (_Phi((b3 - v) / s3) - _Phi((-b3 - v) / s3))

            return acb.integral(g, -1, 1) * b2 / s2

        def f(x, _analytic):
            return _phi(c1 * x) * inner(b1 * x)

        return (2 * c1 * acb.integral(f, 0, 1)).real
    raise ValueError(f"certified recursion is implemented for K <= 3, got K={K}")


def _spend(design: Design, t: arb, a: arb, z_obf: arb | None) -> arb:
    """Per-side cumulative spend a(t) as a ball."""
    if design.kind == "ld-obf":
        return 2 - 2 * norm_cdf(z_obf / t.sqrt())
    if design.kind == "ld-pocock":
        return a * (1 + (arb.const_e() - 1) * t).log()
    return a * t ** arb(design.rho)


def _enc_ball(e: Enclosure) -> arb:
    return ball(e.lo, e.hi)


def certified_boundaries(design: Design, guesses: list[float]) -> list[Enclosure]:
    """Certified enclosures of c_1..c_K (K <= 3). ``guesses`` only affect speed."""
    K = design.K
    if K > 3:
        raise ValueError("certified boundaries are available for K <= 3")
    prec = PREC_K3 if K == 3 else PREC_K2
    tf = design.t
    alpha = Fraction(design.alpha)
    method = f"arb-root:acb.integral(K={K})" if K > 1 else "arb-root:erf"

    if design.kind in ("pocock", "obf"):

        def G(C: arb) -> arb:
            t = [rational_ball(v) for v in tf]
            if design.kind == "pocock":
                cs = [C] * K
            else:
                cs = [C * (arb(K) / k).sqrt() for k in range(1, K + 1)]
            return cont_prob(cs, t) - (1 - rational_ball(alpha))

        g0 = guesses[0] if design.kind == "pocock" else guesses[-1]
        br = certified_root(
            G, g0, increasing=True, rel_width=GST_REL_WIDTH[K], lower=Fraction(0), base_prec=prec
        )
        out = []
        for k in range(1, K + 1):
            if design.kind == "pocock":
                out.append(Enclosure(br.lo, br.hi, method, br.prec))
            else:
                with precision(br.prec * 2):
                    f = (arb(K) / k).sqrt()
                    lo, _ = arb_to_fraction(ball(br.lo, br.lo) * f)
                    _, hi = arb_to_fraction(ball(br.hi, br.hi) * f)
                out.append(Enclosure(lo, hi, method, br.prec))
        return out

    # Spending designs: solve look by look; earlier boundaries enter as balls.
    a = alpha / 2
    z_enc = None
    if design.kind == "ld-obf":
        from certtables.enclose.continuous import quantile

        z_enc = quantile("norm", "isf", float(a / 2), {})
        if Fraction(float(a / 2)) != a / 2:  # pragma: no cover - halving a double is exact
            raise EnclosureError("alpha/4 is not a double")
    out: list[Enclosure] = []
    for k in range(1, K + 1):
        prev = list(out)

        def G(c: arb, k=k, prev=prev) -> arb:
            t = [rational_ball(v) for v in tf[:k]]
            z = _enc_ball(z_enc) if z_enc is not None else None
            spent = _spend(design, t[-1], rational_ball(a), z)
            if k == 1:
                return norm_sf(c) - spent  # P(Z_1 >= c_1) = a(t_1), decreasing in c
            cs = [_enc_ball(e) for e in prev] + [c]
            return cont_prob(cs, t) - (1 - 2 * spent)

        br = certified_root(
            G,
            guesses[k - 1],
            increasing=k > 1,
            rel_width=GST_REL_WIDTH[k],
            lower=Fraction(0),
            base_prec=PREC_K3 if k == 3 else PREC_K2,
        )
        out.append(
            Enclosure(
                br.lo, br.hi, f"arb-root:acb.integral(K={k})" if k > 1 else "arb-root:erfc", br.prec
            )
        )
    return out


# ------------------------------------------------------------------- reference (mpmath)


def mp_cont_prob(c: list, t: list, nodes: list, mp) -> object:
    """P(|Z_k| < c_k for all k) by the recursion with a fixed Gauss-Legendre rule ``nodes``
    (pairs (x, w) on [-1, 1]); NOT certified."""
    K = len(c)
    sq = [mp.sqrt(v) for v in t]
    b = [c[k] * sq[k] for k in range(K)]
    if K == 1:
        return mp.erf(c[0] / mp.sqrt(2))
    # look 1: points u_i on [-b1, b1], weights w_i * density
    pts = [b[0] * x for x, _ in nodes]
    wts = [b[0] * w * mp.npdf(u, 0, sq[0]) for (_, w), u in zip(nodes, pts, strict=True)]
    for k in range(1, K):
        s = mp.sqrt(t[k] - t[k - 1])
        if k == K - 1:
            return mp.fsum(
                w * (mp.ncdf((b[k] - u) / s) - mp.ncdf((-b[k] - u) / s))
                for u, w in zip(pts, wts, strict=True)
            )
        new_pts = [b[k] * x for x, _ in nodes]
        new_wts = [
            b[k] * wv * mp.fsum(w * mp.npdf(v - u, 0, s) for u, w in zip(pts, wts, strict=True))
            for (_, wv), v in zip(nodes, new_pts, strict=True)
        ]
        pts, wts = new_pts, new_wts
    raise AssertionError("unreachable")  # pragma: no cover


def mp_boundaries(design: Design, degree: int = 6, dps: int = 30) -> list[float]:
    """Boundaries from the mpmath recursion (3 * 2**(degree-1) nodes per look); NOT
    certified. Returns floats."""
    import mpmath
    from mpmath.calculus.quadrature import GaussLegendre

    mp = mpmath.mp
    with mpmath.workdps(dps):
        nodes = GaussLegendre(mp).calc_nodes(degree, mp.prec)
        K = design.K
        t = [mp.mpf(v.numerator) / v.denominator for v in design.t]
        alpha = mp.mpf(design.alpha)

        def solve(fun, x0, x1):
            # Illinois on a bracket: the continuation probability minus its target is
            # negative at x0 and positive at x1 (DESIGN 2.6), so it cannot wander off.
            return mp.findroot(fun, (mp.mpf(x0), mp.mpf(x1)), solver="illinois")

        if design.kind in ("pocock", "obf"):
            if design.kind == "pocock":
                cs = lambda C: [C] * K  # noqa: E731
            else:
                cs = lambda C: [C * mp.sqrt(mp.mpf(K) / k) for k in range(1, K + 1)]  # noqa: E731
            C = solve(lambda C: mp_cont_prob(cs(C), t, nodes, mp) - (1 - alpha), 0.5, 15.0)
            return [float(v) for v in cs(C)]
        a = alpha / 2
        # z_{1-a/2} = sqrt(2) erfinv(1 - a)
        z = mp.sqrt(2) * mp.erfinv(1 - a) if design.kind == "ld-obf" else None

        def spend(tk):
            if design.kind == "ld-obf":
                return 2 - 2 * mp.ncdf(z / mp.sqrt(tk))
            if design.kind == "ld-pocock":
                return a * mp.log(1 + (mp.e - 1) * tk)
            return a * tk ** mp.mpf(design.rho)

        out: list = []
        for k in range(1, K + 1):
            target = spend(t[k - 1])
            if k == 1:
                out.append(-mp.sqrt(2) * mp.erfinv(2 * target - 1))
                continue
            prev = list(out)
            ck = solve(
                lambda c, prev=prev, k=k, target=target: (
                    mp_cont_prob(prev + [c], t[:k], nodes, mp) - (1 - 2 * target)
                ),
                0.5,
                15.0,
            )
            out.append(ck)
        return [float(v) for v in out]


def reference_boundaries(design: Design) -> list[Enclosure]:
    """mpmath boundaries with an error estimate from two rule sizes (96 vs 48 nodes),
    widened tenfold; kind ``reference`` (NOT certified)."""
    hi_res = mp_boundaries(design, degree=6, dps=REF_DPS)
    lo_res = mp_boundaries(design, degree=5, dps=REF_DPS)
    out = []
    for v6, v5 in zip(hi_res, lo_res, strict=True):
        err = max(10 * abs(Fraction(v6) - Fraction(v5)), abs(Fraction(v6)) * Fraction(1, 10**15))
        out.append(
            Enclosure(
                Fraction(v6) - err,
                Fraction(v6) + err,
                "mpmath-gl:96v48(not certified)",
                100,
                "reference",
            )
        )
    return out
