# stats.t.ppf / t.isf in the far tail: two silent failures besides the sign flip of #25968

Draft of a comment on scipy#25968, not filed. SciPy 1.18.1, Windows x86-64.

scipy#25968 reports `+inf` for very small `p`. The certified table shows two more modes, and
one line of that report needs correcting: `stdtrit(3, 1e-230) = -2.3976e+76` is called fine,
but the true value is -4.795e+76.

## Reproducing code

```python
from scipy import stats

print(stats.t.ppf(1e-200, 3))  # -2.397637878497004e+66; exact -4.79527572046922e+66
print(stats.t.cdf(stats.t.ppf(1e-200, 3), 3))  # 8.0e-200, not 1e-200
print(stats.t.ppf(1e-100, 0.5))  # -4.740375954054589e+153; exact -1.02849115631634e+199
print(stats.t.ppf(1e-300, 5))  # inf; exact -1.56839255909934e+60
```

## Three modes

Of 91 wrong points (grid and random draws, `t.ppf` and `t.isf`):

| Mode | Points | Largest `p` seen | Where |
|---|---|---|---|
| Finite, stuck near `sqrt(df / DBL_MIN)` (about 6.7e153 sqrt(df)) | 47 | 1e-20 | true `|x|` above about 1e154, small df |
| Finite, off by a factor of 1.6 to 2.8 | 15 | 1.1e-142 | `df` from about 2.05 to 4.5 |
| Infinite with the wrong sign (#25968) | 29 | 2.0e-230 | |

- **Stuck.** The value stops moving with `p`: for df = 0.1, `p = 1e-20` and `p = 1e-30` return
  the same -2.1199605744342963e+153, against exact values of -1.60e+196 and -1.60e+296. A
  reduced variable like `df / (df + x^2)` underflows there. These quantiles are finite in
  binary64.
- **Factor.** Silent, with no warning, and the result looks plausible: `t.cdf(t.ppf(p, df),
  df)` gives about `8 p`, not `p` (8.9 at df = 2.05, 8.0 at df = 3, 7.6 at df = 4.5). For
  df = 3 this starts between `p` = 1e-160 and 1e-165. Not at df = 2 or 4 (closed forms).

The factors above come from `stats.t.cdf(stats.t.ppf(p, df), df) / p` over `p = 10^-k`,
`k = 5, 10, ..., 295`.

## Status upstream

boostorg/math#1436 was closed by boostorg/math#1437 (merged 2026-08-26), which stops `p1`
underflowing in `fast_students_t_quantile_imp`. SciPy 1.18.1 was released on 2026-08-21, so
it does not have that change. Whether it also cures the two modes above has not been tested
here: rerun this table once SciPy ships a newer Boost.
