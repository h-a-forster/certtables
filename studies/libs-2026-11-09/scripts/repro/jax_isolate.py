# Is jax#42458 the same bug as #41201? Split the error into prefactor (ax) vs series/CF.
import jax, mpmath as mp
mp.mp.dps = 50
from jax.scipy.special import gammainc, gammaincc
import jax.numpy as jnp
def run(dt, a):
    A = jnp.asarray(a, dt); X = A
    ax = A*jnp.log(X) - X - jax.lax.lgamma(A)          # the prefactor JAX computes
    axe = a*mp.log(a) - a - mp.loggamma(a)              # exact
    fac = float(mp.e**(mp.mpf(float(ax)) - axe))        # error factor from ax alone
    P, Q = float(gammainc(A, X)), float(gammaincc(A, X))
    Pt = mp.gammainc(a, 0, a, regularized=True) if a < 1e7 else mp.mpf(0.5) + 1/(3*mp.sqrt(2*mp.pi*a))  # Temme leading terms, error ~a^-1.5, ample at a>=1e7
    Qt = 1 - Pt
    print(f"{dt.__name__:8} a={a:8.0e}  ax err factor {fac:.6f} | P/Ptrue {P/float(Pt):.6f}  Q/Qtrue {Q/float(Qt):.6f} | after removing ax error: P {abs(P/fac/float(Pt)-1):.1e} Q {abs(Q/fac/float(Qt)-1):.1e}")
for a in (5e4, 5e5): run(jnp.float32, a)
jax.config.update("jax_enable_x64", True)
for a in (5e4, 5e5, 1e8, 1e10, 1e12): run(jnp.float64, a)
