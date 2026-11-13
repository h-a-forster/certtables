# certtables design

This document fixes what is computed, why the enclosures are proofs, how targets are
scored, and what is out of scope. `AGENTS.md` points here before any change to a formula,
a grid or a scoring rule.

## 1. Goal and estimands

certtables provides reference values for statistical functions as intervals `[lo, hi]`
that contain the true value (certified or exact), and uses them to count how many digits
current Python implementations get right.

Estimands (section 4):

- **Grid census** (exact, no interval): on the declared deterministic grid, for each target
  and family, the number and fraction of points *certified wrong*, and the number of points
  with LRE below 6, 10 and 13.
- **Domain failure rate**: on seeded log-uniform draws over each family's parameter domain
  (about 2,000 points per family), the fraction of points certified wrong, with a
  Clopper-Pearson 95% interval.

## 2. Enclosures

### 2.1 Inputs are doubles

Every input parameter is the exact binary64 value an implementation receives: `p = 0.975`
means 0.97499999999999997780... The reference is the function at that exact input. This
matters at the last digit: the quantile of the double 0.975 rounds to 1.9599639845400538,
one unit in the last place below the familiar 1.959963984540054. Clopper-Pearson bounds use
`alpha / 2` as a double (halving is exact). Group-sequential information fractions are exact
rationals (`k/K`, or a schedule such as `3/10,1`); targets receive their nearest doubles,
which moves a boundary by about 1e-16 relative, far below the tolerances used there.

### 2.2 Ball arithmetic gives proofs

Arb (through python-flint 0.8.0, pinned; wheels for Windows, Linux and macOS, CPython
3.11-3.14) represents a real number as a ball `[m +/- r]` and every operation returns a ball
that contains the exact result for every input in the input balls. Its special functions
(`erfc`, `gamma_upper`, `beta_lower` with `regularized=1`, `erf`) carry rigorous error bounds
for their series, asymptotic expansions and transformations. So if the ball for `F(x) - p`
is entirely positive, `F(x) > p` is a theorem about the exact input, not an estimate.

Functions used (all inputs exact):

| family | CDF / SF |
|---|---|
| normal | `erfc(-x/sqrt 2)/2`, `erfc(x/sqrt 2)/2` |
| t | tail `I_{df/(df+x^2)}(df/2, 1/2)/2` for x > 0; the other side by `1 - tail`; exactly 1/2 at x = 0 |
| chi-square | SF `gamma_upper(k/2, x/2, regularized)`; CDF by the series below for x/2 <= k/2 + 1, else `1 - SF` |
| F | `I_{d1 x/(d1 x + d2)}(d1/2, d2/2)` and `I_{d2/(d2 + d1 x)}(d2/2, d1/2)` |
| beta | `I_x(a, b)` and `I_{1-x}(b, a)` |

One gap was found and closed: python-flint 0.8.0's `gamma_lower` returns non-finite or very
wide balls, and takes about a second, for large shape `s` near `z = s` (for example
`s = 5e5`, `z = 0.999 s`). The chi-square lower tail therefore uses the series
`P(s, z) = z^s e^-z / Gamma(s+1) * sum_k z^k / ((s+1)...(s+k))`, summed in ball arithmetic
until the remainder bound `t_N r / (1 - r)` with `r = z/(s+N+1) < 1` is below `2^-(prec+10)`
of the sum; the bound is added to the ball. A test checks `P + Q - 1` contains 0 against
Arb's `gamma_upper`.

A second gap: `beta_lower` can return nan up to the precision limit for an argument very
close to 1 (for example `f.cdf(4666.36; 1.81417, 772.678)`, whose SF is about 2.6e-417; the
ball is finite from 2048 bits). When no precision gives a finite ball, `continuous.value`
falls back to `1 - other tail` at 1024 bits, which is rigorous and accepted only if tight.

Precision: values are computed at 128 bits and recomputed at 256, 512 and 1024 bits until
the relative radius is at most 1e-22. A value certainly below `2^-1100` (far under the
smallest subnormal, 2^-1074) is stored as the enclosure `[0, 2^-1100]` with method suffix
`;below-2^-1100`: every binary64 implementation should return 0 or a subnormal there, and
exact digits would cost megabit rationals.

### 2.3 Quantiles: the monotonicity argument

For a strictly increasing `G(x) = CDF(x) - p` (or decreasing `SF(x) - q`), if the ball of
`G(a)` is entirely negative and that of `G(b)` entirely positive, then the unique root lies
in `(a, b)`. `certified_root` (in `enclose/arbutil.py`) works only with exact dyadic points:

1. From a starting point (1 for quantiles; never a SciPy value, so the digits of the
   reference do not depend on an implementation under test) it expands outwards, 4x per step
   and 2^16x after 30 steps, and geometrically towards a domain bound (0 for chi-square, F
   and beta), until both signs are certified.
2. It narrows the bracket by Illinois (secant on ball midpoints) steps, a bisection
   fallback after three same-side steps, and geometric bisection while the bracket spans
   more than a factor of 4. Every new point replaces an end only after its sign is
   certified, so the bracket stays certified at every step.
3. When a point is undecided (the ball contains 0), it is re-evaluated at twice the
   precision while that keeps shrinking the ball; if it stays undecided the point is within
   the resolution of `G`, and the search probes windows around it growing 4x from a quarter
   of the target width, keeping the closest certified points.

The starting point cannot affect soundness, only speed and the exact bracket endpoints.
Group-sequential searches start from an mpmath value, so their digits depend on the mpmath
version in `MANIFEST.json`; their validity does not.

Target relative width: 1e-20 (quantiles), so LRE bounds against a binary64 value are
essentially exact. Exact zeros (`G(x)` an exact zero ball, for example the normal median)
return the point itself. Quantiles outside `[2^-1100, 2^1100]` are not enclosed and are
listed in `data/MANIFEST.json` under `not_enclosed`; there `RootOutOfRange` means the sign of
`G` at the limit proves the quantile lies beyond it, so such a point has no finite binary64
answer (or only 0 or a subnormal) rather than an unknown one.

### 2.4 Clopper-Pearson

The lower bound for `k` successes in `n` trials solves `P(X >= k | p) = alpha/2`, the beta
quantile `ppf(alpha/2; k, n-k+1)`; the upper bound solves `P(X <= k | p) = alpha/2`, the beta
inverse survival function `isf(alpha/2; k+1, n-k)`. The bounds are exactly 0 for `k = 0`
and 1 for `k = n`. Closed forms checked by tests: for `k = n` the lower bound is
`(alpha/2)^(1/n)`.

### 2.5 Exact rational results

No rounding at all: integer arithmetic in Python and FLINT.

- Binomial tails at the exact double `p = m / 2^e`: `sum_i C(n,i) m^i (2^e - m)^(n-i) / 2^(e n)`.
- Fisher's exact test on a 2x2 table with both margins fixed: `less` = `P(X <= a)`,
  `greater` = `P(X >= a)`, `two-sided` = sum over tables whose probability is at most the
  observed one, compared exactly. (SciPy compares with a relative slack of 1e-7 to catch
  ties lost to rounding; exact comparison needs none, so the two agree unless two tables
  differ in probability by less than 1e-7 relative.)
- Wilcoxon signed-rank: counts of subsets of `{1..n}` by sum; p-values `P(W+ >= w)`,
  `P(W+ <= w)`, two-sided `min(1, 2 min(...))`.
- Mann-Whitney U: coefficients of the Gaussian binomial `[m+n choose m]_q`, built by
  multiplying by `(1 - q^(n+i))` and dividing by `(1 - q^i)` exactly; p-values as above.
- One-sample two-sided Kolmogorov-Smirnov `P(D_n <= d)` at the exact double `d`: Durbin's
  matrix formula (Marsaglia, Tsang and Wang 2003) with an exact `fmpq_mat` power for
  `d < 1/2`; for `d >= 1/2` the two one-sided events are disjoint and
  `P(D_n >= d) = 2 P(D_n^+ >= d)` with the Birnbaum-Tingey sum. Tests check that both formulas
  give the same rational where both apply. Durbin matrices above dimension 121 are refused;
  the grid keeps `n d <= 30` below `d = 1/2` (n up to 100).

Exact rows store outward-rounded 25-digit decimals and the SHA-256 of `numerator/denominator`
(hex) so a recomputation can be checked bit for bit.

### 2.6 Group-sequential boundaries

Two-sided designs with `|Z_k| >= c_k` at information fractions `t_1 < ... < t_K = 1`, `Z_k`
the standardised statistic of a Brownian motion `W`. The probability of never rejecting is
the Armitage-McPherson-Rowe recursion:

- K = 1: `erf(c_1 / sqrt 2)`.
- K = 2: `2 c_1 int_0^1 phi(c_1 x) [Phi((b_2 - b_1 x)/s_2) - Phi((-b_2 - b_1 x)/s_2)] dx` with
  `b_k = c_k sqrt(t_k)`, `s_k = sqrt(t_k - t_{k-1})` (the substitution makes the limits exact).
- K = 3: the same with an inner integral over look 2 inside the outer integrand.

`acb.integral` (Arb's rigorous Gauss-Legendre with adaptive bisection) bounds the error of
each integral from the integrand's magnitude on ellipses in the complex plane; it needs an
integrand that is analytic there, and these are entire (products of `exp` and `erfc`). For
K = 3 the outer integrand is evaluated on complex balls, and the inner `acb.integral` is run
with that ball as a parameter. Every operation inside it is ball arithmetic, so its result
contains the inner integral for every point of the ball; that is exactly the bound the outer
error estimate needs. Earlier boundaries of a spending design enter later looks as balls, so
uncertainty propagates rather than being dropped. Working precision: 128 bits for K = 2,
64 bits for K = 3 (each nested evaluation takes about 10 s; the ball radius is about 1e-17).

Constants are enclosed by the same certified sign change, using monotonicity: the
continuation probability increases in the constant (Pocock `c_k = C`, O'Brien-Fleming
`c_k = C sqrt(K/k)`) and, look by look, in `c_k` for spending designs (Lan-DeMets
O'Brien-Fleming-type `a(t) = 2 - 2 Phi(z_{1-a/2}/sqrt t)`, Pocock-type
`a(t) = a log(1 + (e-1) t)`, power family `a(t) = a t^rho`, all per side with
`a = alpha/2`; the two-sided spend is `2 a(t_k)`, the gsDesign/ldbounds convention). Target
widths: 1e-20 (K <= 2), 2e-12 (K = 3; the mpmath start is so close that the rows reach about
4e-15).

The time-box in the plan (ship K = 2 if certified K = 3 was not closed early) was not
needed: nested K = 3 worked on the first prototype, and the 7 K = 3 designs on the grid
take about 4 CPU minutes in all.

**K = 4, 5 are references, NOT certified**: the same recursion on fixed Gauss-Legendre rules
in mpmath (20 digits), solved with 96 and with 48 nodes per look; the stored interval is the
96-node value +/- max(10 x |difference|, 1e-15 relative). Rows have `kind = reference` and are
scored only as agree/disagree, never in the certified census. A slow test checks that the same
recursion lands on the certified K = 3 value to 1e-14.

## 3. Grid, draws and the table

### 3.1 Deterministic grid (`certtables.grid.deterministic_grid`)

Continuous families: CDF and SF on fixed x grids (deep tails included), PPF and ISF at 24
probabilities from 1e-300 to 0.999; t df 0.1 to 1e8 (20 values), chi-square df 0.1 to 1e6
(17), 15 F pairs. Clopper-Pearson: n = 1 to 10^4, k in {0, 1, 2, n/10, n/4, n/2, n-1, n},
alpha in {0.1, 0.05, 0.01}. Binomial, Fisher, Wilcoxon, Mann-Whitney, KS and the
group-sequential designs as listed in `grid.py`. The grid hash (SHA-256 of the newline-joined
point ids) is in `data/MANIFEST.json`.

### 3.2 Random draws (`grid.random_points`)

Seed `20261030`, 2,000 draws per family (duplicates dropped), parameters rounded to 6
significant digits (so ids stay short; the rounded double is the input):

- function uniform over cdf, sf, ppf, isf; df (t, chi-square) and dfn, dfd (F) log-uniform on
  [0.1, 1e6];
- x: normal sign x log-uniform [1e-3, 38]; t sign x log-uniform [1e-3, 1e3]; chi-square
  `df * 10^U(-3, 2)`; F log-uniform [1e-4, 1e4]; p (ppf, isf) log-uniform [1e-300, 0.5];
- Clopper-Pearson: n = round(log-uniform [1, 1e4]), k uniform on 0..n, alpha log-uniform
  [1e-4, 0.2], side uniform.

The failure rate is therefore a rate under this measure, not over "typical use".

### 3.3 Table and build

`certtables build` writes `data/certtables-v1.csv` and `data/certtables-v1-random.csv`
(columns in `table.py`), checkpointing per family under `runs/build/<set>/` (rows appended
and fsynced as they finish, finished ids skipped on rerun, a torn last line dropped,
`progress.log`, `errors.jsonl`), with `--max-minutes` to stop and resume. Points whose
enclosure fails are recorded in `MANIFEST.json` and left out; a group-sequential design that
fails records every one of its looks. `--max-minutes` is one budget for the whole call.
`certtables verify` checks every row hash, the file hashes, that the declared points still
hash to the manifest's grid hash, and that each declared point has exactly one row or
`not_enclosed` entry.

## 4. Scoring

### 4.1 LRE bounds

strdcheck's LRE (vendored verbatim in `certtables/lre.py`): `-log10(|x - c|/|c|)`, clipped to
`[0, 15]`, below 1 reported as 0, `nan` for non-finite values, absolute error when the
reference can be 0. Against an interval, `LRE_lo` uses the worst point of the enclosure and
`LRE_hi` the best (15 when the value lies inside). `|x/c - 1|` is quasi-convex in `c`, so the
extremes are at the endpoints. Logs are taken of exact rationals, so deep-tail references
(1e-300 and below) do not underflow.

### 4.2 Verdicts and tolerances

`tol(c) = max(rel |c|, abs)`. A value is **certified correct** when the farthest endpoint is
within the smallest tolerance over the enclosure, **certified wrong** when its distance to
the enclosure exceeds the largest tolerance over it, and **undecided** otherwise. Both
labels are sufficient conditions, hence proofs. Non-finite values against finite references
are wrong; exceptions are counted as `error`; quantile references outside the normal binary64
range are `out-of-range` and not scored.

| target | rel | abs |
|---|---|---|
| double-precision functions (SciPy, statsmodels, mpmath at 15 digits, JAX) | 1e-12 | `DBL_MIN` = 2.2e-308 for probability outputs (CDF, SF, p-values), else 0 |
| spotify-confidence boundaries (no declared accuracy; its solver stops at a probability tolerance of 1e-7) | 1e-6 | 0 |
| printed constants (three decimals) | 0 | 5e-4 |

mpmath runs each point in a worker process with a 2-second limit, because its incomplete beta
can run for minutes with both shapes large; a point past the limit is an `error`, so its
`error` count depends slightly on machine speed.

The absolute floor `DBL_MIN` reflects gradual underflow: below the smallest normal double a
result is subnormal and loses relative precision legitimately, so returning 0 for 1e-310 is
correct, while returning 0 for 1e-300 is wrong. Rows whose reference is below `DBL_MIN` are
left out of the LRE census (scored by verdict only).

### 4.3 Census

`certtables report` writes `results/census-grid.csv` (exact counts; `lre_ambiguous` counts
points where `LRE_lo` and `LRE_hi` fall on different sides of a threshold, expected 0),
`results/census-random.csv` (rates with Clopper-Pearson 95% intervals from this package's own
certified beta quantiles, rounded outward to 4 decimals), `results/wrong.csv` and
`results/census.md`.

### 4.4 Triage of wrong points

Every wrong point can be reproduced on its own with `certtables repro '<id>' --target <name>`,
which prints the target's value, the enclosure, the verdict and the LRE bounds. Root causes
are grouped in `docs/upstream/`, one draft per cause, with numbers from
`scripts/upstream_checks.py`, checked against the project's tracker before drafting. A draft is filed only with the
owner's go-ahead; [upstream.md](upstream.md) lists what was filed and its outcome.

`scripts/crosscheck_mpmath.py` tests the rows behind those verdicts, and a seeded sample of
the others, against mpmath at 80 digits. That is evidence, not a proof, and only about how Arb
is used: for t and F, mpmath evaluates the same beta identities. The identities themselves
rest on the closed-form tests and on agreement with targets.

## 5. Out of scope

- **Simulation-defined tables**, such as the Model Confidence Set critical values and the
  Acerbi-Szekely expected-shortfall backtest thresholds. Their "true" values are defined by a
  bootstrap or Monte Carlo procedure, not by a closed-form or integral expression, so there
  is nothing to enclose: an interval would only restate Monte Carlo error.
- Discrete quantiles (binomial ppf) and noncentral distributions: not yet; adding them is a
  new family (CONTRIBUTING.md).
- Certified K >= 4 group-sequential boundaries: each extra look nests another adaptive
  integral, multiplying the cost by the number of outer evaluations (K = 2 takes
  milliseconds, K = 3 about 10 s per evaluation); they are references here.
- Accuracy claims about R packages (gsDesign, rpact, ldbounds): the table is
  language-neutral and can test them, but this release runs Python targets only.

## 6. Prior work

The method is not new; a current, public table and a census of today's stack are.

- **Interval methods for distributions.** Wang and Kennedy (1994) gave self-validating
  computations for the normal, incomplete gamma, incomplete beta and noncentral chi-square
  functions in interval arithmetic. Baharev, Schichl and Rev (2017) compute the noncentral F
  with guaranteed accuracy (even denominator df). Neither publishes a reusable table or tests
  current software.
- **Reliability audits.** McCullough (1998) set the LRE method; Knusel (1998) tested Excel's
  distributions against his ELV program, which is not an interval method. Almiron, Almeida and
  Miranda (2009) is the last comparison found that included Python, on NIST datasets rather
  than on tail quantiles. `strdcheck` repeats the StRD part for today's Python stack.
- **Maintainers' own tests.** SciPy and Boost.Math test against high-precision mpmath or
  multiprecision values, function by function and without enclosures. R's `DPQ` and
  `DPQmpfr` packages diagnose R's own functions the same way.
- **libm audits.** Gladman, Innocente, Mather and Zimmermann track worst-case ulp errors of
  elementary and some special functions across libm implementations, by search; that is the
  closest modern analogue, one layer below these distributions.
- **Group-sequential software.** gsDesign, rpact and ldbounds compute boundaries to numerical
  tolerance and are validated against each other (differences below about 1e-6); no
  certified reference was found.

A web search on 2026-11-06 found no interval-certified public table of these quantities and
no census of current SciPy against one. The closest SciPy reports are listed with the
findings in `docs/upstream/`.

## 7. References

- Johansson, F. (2017). Arb: efficient arbitrary-precision midpoint-radius interval
  arithmetic. IEEE Transactions on Computers 66(8). arXiv:1611.02831.
- Johansson, F. (2018). Numerical integration in arbitrary-precision ball arithmetic.
  arXiv:1802.07942.
- Wang, C.-C. M. and Kennedy, W. J. (1994). Self-validating computations of probabilities
  for selected central and noncentral univariate probability functions. JASA 89(427),
  878-887.
- Baharev, A., Schichl, H. and Rev, E. (2017). Computing the noncentral-F distribution and
  the power of the F-test with guaranteed accuracy. Computational Statistics 32(2), 763-779.
- Knusel, L. (1998). On the accuracy of statistical distributions in Microsoft Excel 97.
  Computational Statistics & Data Analysis 26(3), 375-377.
- Almiron, M. G., Almeida, E. S. and Miranda, M. N. (2009). The reliability of statistical
  functions in four software packages freely used in numerical computation. Brazilian
  Journal of Probability and Statistics 23(2), 107-119.
- Gladman, B., Innocente, V., Mather, J. and Zimmermann, P. Accuracy of mathematical
  functions in single, double, double extended, and quadruple precision. HAL hal-03141101.
- McCullough, B. D. (1998). Assessing the reliability of statistical software: Part I. The
  American Statistician 52(4), 358-366.
- Marsaglia, G., Tsang, W. W. and Wang, J. (2003). Evaluating Kolmogorov's distribution.
  Journal of Statistical Software 8(18).
- Armitage, P., McPherson, C. K. and Rowe, B. C. (1969). Repeated significance tests on
  accumulating data. JRSS A 132(2), 235-244.
- Jennison, C. and Turnbull, B. W. (2000). Group Sequential Methods with Applications to
  Clinical Trials. Chapman and Hall/CRC.
- Lan, K. K. G. and DeMets, D. L. (1983). Discrete sequential boundaries for clinical trials.
  Biometrika 70(3), 659-663.
- Clopper, C. J. and Pearson, E. S. (1934). The use of confidence or fiducial limits
  illustrated in the case of the binomial. Biometrika 26(4), 404-413.
