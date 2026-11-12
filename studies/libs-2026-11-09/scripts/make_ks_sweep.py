"""KS points over the Durbin branch of KSDist, on both sides of frac(n d) = 1/2, with exact rationals."""
import random
from certtables.grid import Point
rng = random.Random("20261109:ks")
out = {}
for n in (2, 3, 4, 5, 6, 8, 10, 15, 20, 30, 50, 100):
    for _ in range(30):
        k = rng.randint(1, max(1, min(n - 2, 20 if n <= 100 else 20)))
        f = rng.uniform(0.02, 0.98)
        d = float(f"{(k + f) / n:.8g}")
        if d >= 0.5 and n > 2:
            pass
        if d < 1.0 and n * d * d <= 4 and n * d > 1 and n * d < n - 1:
            for fn in ("cdf", "sf"):
                pt = Point("ks", fn, (("n", n), ("d", d)))
                out[pt.id] = pt
open("data/points-ks.tsv", "w").write("".join(f"sweep\t{i}\n" for i in out))
print(len(out))
