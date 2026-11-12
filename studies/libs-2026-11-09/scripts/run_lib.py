"""Evaluate one installed library at the shared points, one point at a time.

usage: run_lib.py <torch|jax|tfp> <f64|f32> points.tsv out.tsv

Appends `id <TAB> api <TAB> value` (or `ERR:<type>` / `HANG`). Finished (id, api) pairs are
skipped on rerun. The (id, api) about to be evaluated is written to <out>.pending first, so a
rerun after a kill or crash records that pair as HANG instead of repeating it. Progress goes to
stdout. Standard library plus the target package only; no certtables import.
"""

import os
import sys
import time

LIB, MODE, POINTS, OUT = sys.argv[1:5]
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "1")
os.environ.setdefault("TF_NUM_INTEROP_THREADS", "1")
os.environ.setdefault("XLA_FLAGS", "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1")


def parse(pid):
    head, _, rest = pid.partition("(")
    fam, _, fn = head.partition(".")
    p = {}
    for item in rest[:-1].split(";"):
        k, _, v = item.partition("=")
        try:
            p[k] = int(v)
        except ValueError:
            try:
                p[k] = float(v)
            except ValueError:
                p[k] = v
    return fam, fn, p


def cp_shape(fn, p):
    """Clopper-Pearson as a beta quantile (DESIGN 2.4); None at the exact 0/1 boundary."""
    n, k, alpha = p["n"], p["k"], p["alpha"]
    if fn == "lower":
        return None if k == 0 else ("ppf", float(k), float(n - k + 1), alpha / 2)
    return None if k == n else ("isf", float(k + 1), float(n - k), alpha / 2)


def torch_apis(mode):
    import torch
    import torch.distributions as D

    torch.set_num_threads(1)
    dt = torch.float64 if mode == "f64" else torch.float32
    T = lambda v: torch.tensor(v, dtype=dt)  # noqa: E731
    N = D.Normal(T(0.0), T(1.0))

    def apis(fam, fn, p):
        if fam == "norm" and fn == "cdf":
            yield "distributions.Normal.cdf", lambda: N.cdf(T(p["x"]))
            yield "special.ndtr", lambda: torch.special.ndtr(T(p["x"]))
        elif fam == "norm" and fn == "ppf":
            yield "distributions.Normal.icdf", lambda: N.icdf(T(p["p"]))
            yield "special.ndtri", lambda: torch.special.ndtri(T(p["p"]))
        elif fam == "chi2" and fn == "cdf":
            yield "distributions.Chi2.cdf", lambda: D.Chi2(T(p["df"])).cdf(T(p["x"]))
        elif fam == "chi2" and fn == "sf":
            # No survival method on Chi2: the special function Gamma.cdf is built from.
            yield "special.gammaincc", lambda: torch.special.gammaincc(T(p["df"]) / 2, T(p["x"]) / 2)

    return apis, lambda r: float(r.item()), torch.__version__


def jax_apis(mode):
    import jax

    jax.config.update("jax_enable_x64", mode == "f64")
    jax.config.update("jax_platform_name", "cpu")
    import jax.numpy as jnp
    from jax.scipy import stats

    dt = jnp.float64 if mode == "f64" else jnp.float32
    T = lambda v: jnp.asarray(v, dtype=dt)  # noqa: E731

    def apis(fam, fn, p):
        if fam == "norm":
            arg = p["x"] if fn in ("cdf", "sf") else p["p"]
            yield f"scipy.stats.norm.{fn}", lambda: getattr(stats.norm, fn)(T(arg))
        elif fam == "chi2" and fn in ("cdf", "sf"):
            yield f"scipy.stats.chi2.{fn}", lambda: getattr(stats.chi2, fn)(T(p["x"]), T(p["df"]))
        elif fam == "beta" and fn in ("cdf", "sf"):
            yield f"scipy.stats.beta.{fn}", lambda: getattr(stats.beta, fn)(T(p["x"]), T(p["a"]), T(p["b"]))

    def check():
        got = str(T(1.0).dtype)
        want = "float64" if mode == "f64" else "float32"
        if got != want:
            raise SystemExit(f"dtype {got}, wanted {want}")

    check()
    return apis, lambda r: float(r), jax.__version__


def tfp_apis(mode):
    import tensorflow as tf
    import tensorflow_probability as tfp

    tfd = tfp.distributions
    dt = tf.float64 if mode == "f64" else tf.float32
    T = lambda v: tf.constant(v, dtype=dt)  # noqa: E731
    METHOD = {"cdf": "cdf", "sf": "survival_function", "ppf": "quantile"}

    def apis(fam, fn, p):
        if fam == "cp":
            s = cp_shape(fn, p)
            if s is None:
                return
            sub, a, b, q = s
            if sub == "ppf":
                yield "Beta.quantile", lambda: tfd.Beta(T(a), T(b)).quantile(T(q))
            return
        if fam == "binom" and fn in ("cdf", "sf"):
            yield f"Binomial.{METHOD[fn]}", lambda: getattr(
                tfd.Binomial(total_count=T(float(p["n"])), probs=T(p["p"])), METHOD[fn]
            )(T(float(p["k"])))
            return
        if fam not in ("norm", "chi2", "t", "beta") or fn not in METHOD:
            return
        arg = p["x"] if fn in ("cdf", "sf") else p.get("p")
        if fam == "norm":
            yield f"Normal.{METHOD[fn]}", lambda: getattr(tfd.Normal(T(0.0), T(1.0)), METHOD[fn])(T(arg))
        elif fam == "chi2":
            yield f"Chi2.{METHOD[fn]}", lambda: getattr(tfd.Chi2(T(p["df"])), METHOD[fn])(T(arg))
        elif fam == "t":
            yield f"StudentT.{METHOD[fn]}", lambda: getattr(
                tfd.StudentT(T(p["df"]), T(0.0), T(1.0)), METHOD[fn]
            )(T(arg))
        elif fam == "beta":
            yield f"Beta.{METHOD[fn]}", lambda: getattr(tfd.Beta(T(p["a"]), T(p["b"])), METHOD[fn])(T(arg))

    return apis, lambda r: float(r.numpy()), f"tfp {tfp.__version__} tf {tf.__version__}"


def main():
    t00 = time.time()
    apis, to_float, version = {"torch": torch_apis, "jax": jax_apis, "tfp": tfp_apis}[LIB](MODE)
    print(f"{LIB} {version} mode={MODE} import {time.time() - t00:.1f}s", flush=True)
    done = set()
    if os.path.exists(OUT):
        with open(OUT, encoding="utf-8") as fh:
            for ln in fh:
                r = ln.rstrip("\n").split("\t")
                if len(r) == 3:
                    done.add((r[0], r[1]))
    pending = OUT + ".pending"
    fh = open(OUT, "a", encoding="utf-8", newline="\n")
    if os.path.exists(pending):
        txt = open(pending, encoding="utf-8").read().strip()
        if txt and tuple(txt.split("\t")) not in done:
            fh.write(f"{txt}\tHANG\n")
            fh.flush()
            done.add(tuple(txt.split("\t")))
    ids = [ln.rstrip("\n").split("\t")[1] for ln in open(POINTS, encoding="utf-8")]
    t0, n = time.time(), 0
    for c, pid in enumerate(ids, 1):
        fam, fn, p = parse(pid)
        for api, call in apis(fam, fn, p):
            if (pid, api) in done:
                continue
            with open(pending, "w", encoding="utf-8") as ph:
                ph.write(f"{pid}\t{api}")
            try:
                v = repr(to_float(call()))
            except Exception as exc:
                v = f"ERR:{type(exc).__name__}:{str(exc)[:80]}".replace("\t", " ").replace("\n", " ")
            fh.write(f"{pid}\t{api}\t{v}\n")
            fh.flush()
            n += 1
        if c % 500 == 0:
            print(f"{c}/{len(ids)} evals={n} {time.time() - t0:.0f}s", flush=True)
    if os.path.exists(pending):
        os.remove(pending)
    print(f"finished points={len(ids)} evals={n} {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
