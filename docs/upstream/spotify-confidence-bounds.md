# sequential_bounds: boundaries about 1e-5 too low from the second look

Draft, not filed. spotify-confidence 4.1.0.

`sequential_bounds(t, alpha=0.05, sides=2)` (power family, rho = 2) against certified
boundaries (K <= 3) and mpmath references (K = 4, 5, not certified):

| `t` | look | spotify-confidence | reference | relative error |
|---|---|---|---|---|
| 1/2, 1 | 2 | 2.01826847 | 2.01831025 (certified) | -2.07e-05 |
| 1/3, 2/3, 1 | 3 | 2.06186385 | 2.06191365 (certified) | -2.41e-05 |
| 1/5, ..., 1 | 5 | 2.11396790 | 2.11402742 (reference) | -2.82e-05 |

Look 1 is correct to 15 digits. Every later look is low, by 6e-6 to 2.8e-5 relative on the 13
later looks in the grid (`results/wrong.csv`), so the boundary is slightly anti-conservative.
At `alpha = 0.05` the last boundary's error of 5e-5 moves the tail probability by about
`2 phi(c) dc = 5e-6`: harmless in practice.

`landem` integrates with the trapezoid rule at a fixed step `h = 0.05` and stops the bisection
at a probability tolerance of 1e-7; the size of the error fits that step, though this was not
tested by changing `h`. A note in the docs that bounds are good to about 4 decimals would do.
No issue found on spotify/confidence (searched `sequential bounds accuracy`, `precision`,
`ldbounds`).
