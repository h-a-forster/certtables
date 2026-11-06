# stats.f.ppf returns nan or loses digits for very small p

Draft, not filed. SciPy 1.18.1, Windows x86-64.

```python
from scipy import stats

print(stats.f.ppf(1e-200, 10, 100))  # nan; exact 5.01179869275565e-41
print(stats.f.ppf(9.28072e-296, 11348.1, 41.5357))  # 0.02600075065654653; exact 0.0252608616...
```

Of 22 wrong points, 18 are `nan` at `p` from 1e-300 to 4e-181 with `dfn` from about 4 to 16,
and 4 are finite, with LRE from 1.53 to 11.86 (`results/wrong.csv`).

`f.ppf` goes through `fdtri`, the Boost inverse incomplete beta. scipy#26531 reports that
inverse failing for small `q` in `stats.beta.ppf`, and a maintainer found it fixed on Boost's
develop branch. This draft adds `f.ppf` as a second entry point to check after the Boost
upgrade; a separate issue should not be needed. Duplicates checked: scipy#20835 (closed,
older path), scipy#22699 (closed, port to Boost).
