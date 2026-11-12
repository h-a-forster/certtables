import sys, jax
x64 = sys.argv[1] == "x64"
jax.config.update("jax_enable_x64", x64)
import jax.numpy as jnp
from jax.scipy import stats, special
print(jax.__version__, "x64" if x64 else "default", jnp.asarray(1.0).dtype)
for df, x in [(1e6, 1e6), (1e5, 1e5), (1e4, 9e3), (1e4, 1.1e4)]:
    print(f"df={df} x={x}", "cdf", repr(float(stats.chi2.cdf(x, df))), "sf", repr(float(stats.chi2.sf(x, df))),
          "jit cdf", repr(float(jax.jit(stats.chi2.cdf)(x, df))), "jit sf", repr(float(jax.jit(stats.chi2.sf)(x, df))))
xs = jnp.asarray([1e6, 1e5]); print("batch sf", stats.chi2.sf(xs, xs))
