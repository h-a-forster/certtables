"""mpmath at its default precision (15 significant digits): a non-certified comparator."""

from __future__ import annotations

import multiprocessing as mp
import queue

from certtables.grid import Point
from certtables.targets.base import Target

#: Seconds per point: mpmath's incomplete beta can run for minutes with both shapes large
#: (for example f.cdf at dfn = dfd = 1e5, x = 0.5); such a point counts as an error.
TIME_LIMIT = 2.0


def _compute(p: dict, family: str, sf: bool) -> float:
    import mpmath

    with mpmath.workdps(15):
        mp = mpmath.mp
        x = mp.mpf(float(p["x"]))
        if family == "norm":
            return float(mp.ncdf(-x) if sf else mp.ncdf(x))
        if family == "chi2":
            k = mp.mpf(float(p["df"]))
            if x <= 0:
                return 1.0 if sf else 0.0
            if sf:
                return float(mp.gammainc(k / 2, x / 2, mp.inf, regularized=True))
            return float(mp.gammainc(k / 2, 0, x / 2, regularized=True))
        if family == "t":
            df = mp.mpf(float(p["df"]))
            if x == 0:
                return 0.5
            ax = abs(x)
            tail = mp.betainc(df / 2, mp.mpf(1) / 2, 0, df / (df + ax * ax), regularized=True) / 2
            return float(tail if (x > 0) == sf else 1 - tail)  # no 1 - (1 - tail) in the tail
        d1, d2 = mp.mpf(float(p["dfn"])), mp.mpf(float(p["dfd"]))
        if x <= 0:
            return 1.0 if sf else 0.0
        if sf:
            return float(mp.betainc(d2 / 2, d1 / 2, 0, d2 / (d2 + d1 * x), regularized=True))
        return float(mp.betainc(d1 / 2, d2 / 2, 0, d1 * x / (d1 * x + d2), regularized=True))


def _serve(inbox, outbox) -> None:
    import mpmath  # noqa: F401 - loaded before the "ready" signal, so it is not timed

    outbox.put("ready")
    while True:
        args = inbox.get()
        try:
            outbox.put((True, _compute(*args)))
        except Exception as exc:
            outbox.put((False, f"{type(exc).__name__}: {str(exc).strip()[:120]}"))


class MpmathTarget(Target):
    name = "mpmath"
    description = "mpmath at mp.dps = 15: ncdf, gammainc and betainc (regularized)"

    def __init__(self) -> None:
        self._worker = None

    def version(self) -> str:
        import mpmath

        return mpmath.__version__

    def supports(self, pt: Point) -> bool:
        return pt.family in ("norm", "t", "chi2", "f") and pt.function in ("cdf", "sf")

    def _start(self):
        # A process, not a thread: only a process can be stopped inside a long mpmath call.
        ctx = mp.get_context("spawn")
        inbox, outbox = ctx.Queue(), ctx.Queue()
        proc = ctx.Process(target=_serve, args=(inbox, outbox), daemon=True)
        proc.start()
        outbox.get(timeout=120)  # the "ready" signal
        return proc, inbox, outbox

    def evaluate(self, pt: Point) -> float:
        if self._worker is None:
            self._worker = self._start()
        proc, inbox, outbox = self._worker
        inbox.put((pt.p, pt.family, pt.function == "sf"))
        try:
            ok, result = outbox.get(timeout=TIME_LIMIT)
        except queue.Empty:
            proc.terminate()
            proc.join()
            self._worker = None
            raise TimeoutError(f"no result within {TIME_LIMIT:g} s") from None
        if not ok:
            raise RuntimeError(result)
        return result
