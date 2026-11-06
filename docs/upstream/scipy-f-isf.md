# BUG: stats.f.isf loses the upper tail: inf below q = 1.1e-16, digits lost well above it

Draft, not filed. SciPy 1.18.1 (current release on 2026-11-06), Windows x86-64.

## Reproducing code

```python
from scipy import stats

for q in (1e-7, 1e-10, 1e-15, 1e-20):
    print(q, stats.f.isf(q, 10, 10), stats.f.sf(stats.f.isf(q, 10, 10), 10, 10))
```

| `q` | `stats.f.isf(q, 10, 10)` | exact value (interval-certified) | correct digits |
|---|---|---|---|
| 1e-7 | 64.40659479777638 | 64.4065947908191929... | 9.97 |
| 1e-10 | 261.40319596323957 | 261.403200316599380... | 7.78 |
| 1e-15 | 2629.470754045719 | 2629.05001748664891... | 3.80 |
| 1e-20 | inf | 26305.5019678183129... | none |

`stats.f.sf` is accurate at these points, so `isf` is the odd one out, and `q` is a valid
input: a p-value of 1e-20 is not unusual in genetics or large A/B data.

## Cause

`f_gen` defines `_ppf` with `fdtri` but no `_isf`, so `rv_continuous._isf` applies:
`self._ppf(1.0 - q)`. Below `q = 1.1e-16` the argument is exactly 1 and `ppf` returns `inf`.
Above that, `1 - q` carries a relative error of about 1e-16 / q in `q`. On 778 points,
`f.isf(q, ...)` equals `f.ppf(1 - q, ...)` on all 778.

## Extent

Against an interval-certified table (778 scored points on a grid and on log-uniform draws of
`dfn`, `dfd` in [0.1, 1e6] and `q` in [1e-300, 0.5]): 551 points wrong by more than a
relative 1e-12, 490 of them `inf`. The largest wrong `q` is 1e-5.

## Suggested fix

`SF(x) = I_z(dfd/2, dfn/2)` with `z = dfd / (dfd + dfn x)`, so for small `q`:

```python
def _isf(self, q, dfn, dfd):
    z = sc.betaincinv(dfd / 2, dfn / 2, q)
    return np.where(q < 0.01, dfd / dfn * (1 / z - 1), sc.fdtri(dfn, dfd, 1 - q))
```

On the same 778 points this makes 533 of the 551 correct and breaks none of the 227 that
were correct. 16 of the other 18 are `nan` from `betaincinv` at very small `q`, probably the
Boost failure of scipy#26531, which the Boost upgrade discussed there may remove.
(Not a tested patch; numbers from `scripts/upstream_checks.py`.)

## Duplicates checked

scipy#17832 (tracking issue, lists `f` without `_isf`, no report of this failure); scipy#20835
(`fdtri` at extreme arguments through `f.ppf`, closed, a different path); scipy#9748 (`f.sf`,
closed). No issue found for `f.isf`.
