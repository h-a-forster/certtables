"""Monte Carlo sanity check, independent of both Julia and Arb: P(D_n <= d)."""
import random, sys
rng = random.Random(20261109)
for n, d, jl, ref in [(3, 0.4, 0.4195555555555556, 0.40533333333333338), (5, 0.25, 0.16755000000000006, 0.1554)]:
    N, hit = 2_000_000, 0
    for _ in range(N):
        u = sorted(rng.random() for _ in range(n))
        D = max(max((i + 1) / n - x, x - i / n) for i, x in enumerate(u))
        hit += D <= d
    p = hit / N; se = (p * (1 - p) / N) ** 0.5
    print(f"n={n} d={d} MC={p:.5f} se={se:.5f} KSDist={jl:.5f} (z={(jl - p) / se:.0f}) exact={ref:.5f} (z={(ref - p) / se:.1f})")
