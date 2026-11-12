
### torch f64

| function | api | set | scored | wrong | error | out of range | rate (95% CP) |
|---|---|---|---|---|---|---|---|
| chi2.cdf | distributions.Chi2.cdf | grid | 306 | 15 | 0 | 0 |  |
| chi2.cdf | distributions.Chi2.cdf | random | 527 | 12 | 0 | 0 | 2.28% (1.18% to 3.94%) |
| chi2.sf | special.gammaincc | grid | 306 | 16 | 0 | 0 |  |
| chi2.sf | special.gammaincc | random | 452 | 11 | 0 | 0 | 2.43% (1.22% to 4.31%) |
| norm.cdf | distributions.Normal.cdf | grid | 25 | 4 | 0 | 0 |  |
| norm.cdf | distributions.Normal.cdf | random | 483 | 49 | 0 | 0 | 10.14% (7.60% to 13.19%) |
| norm.cdf | special.ndtr | grid | 25 | 4 | 0 | 0 |  |
| norm.cdf | special.ndtr | random | 483 | 49 | 0 | 0 | 10.14% (7.60% to 13.19%) |
| norm.ppf | distributions.Normal.icdf | grid | 24 | 9 | 0 | 0 |  |
| norm.ppf | distributions.Normal.icdf | random | 479 | 468 | 0 | 0 | 97.70% (95.93% to 98.85%) |
| norm.ppf | special.ndtri | grid | 24 | 0 | 0 | 0 |  |
| norm.ppf | special.ndtri | random | 479 | 0 | 0 | 0 | 0.00% (0.00% to 0.77%) |

### torch f32

| function | api | set | scored | wrong | error | out of range | rate (95% CP) |
|---|---|---|---|---|---|---|---|
| chi2.cdf | distributions.Chi2.cdf | grid | 306 | 0 | 0 | 0 |  |
| chi2.cdf | distributions.Chi2.cdf | random | 527 | 0 | 0 | 0 | 0.00% (0.00% to 0.70%) |
| chi2.sf | special.gammaincc | grid | 306 | 0 | 0 | 0 |  |
| chi2.sf | special.gammaincc | random | 452 | 0 | 0 | 0 | 0.00% (0.00% to 0.81%) |
| norm.cdf | distributions.Normal.cdf | grid | 25 | 2 | 0 | 0 |  |
| norm.cdf | distributions.Normal.cdf | random | 483 | 26 | 0 | 0 | 5.38% (3.55% to 7.79%) |
| norm.cdf | special.ndtr | grid | 25 | 2 | 0 | 0 |  |
| norm.cdf | special.ndtr | random | 483 | 26 | 0 | 0 | 5.38% (3.55% to 7.79%) |
| norm.ppf | distributions.Normal.icdf | grid | 20 | 5 | 0 | 0 |  |
| norm.ppf | distributions.Normal.icdf | random | 62 | 52 | 0 | 0 | 83.87% (72.33% to 91.98%) |
| norm.ppf | special.ndtri | grid | 20 | 0 | 0 | 0 |  |
| norm.ppf | special.ndtri | random | 62 | 0 | 0 | 0 | 0.00% (0.00% to 5.78%) |

### jax f64

| function | api | set | scored | wrong | error | out of range | rate (95% CP) |
|---|---|---|---|---|---|---|---|
| chi2.cdf | scipy.stats.chi2.cdf | grid | 306 | 5 | 0 | 0 |  |
| chi2.cdf | scipy.stats.chi2.cdf | random | 527 | 4 | 0 | 0 | 0.76% (0.21% to 1.93%) |
| chi2.sf | scipy.stats.chi2.sf | grid | 306 | 6 | 0 | 0 |  |
| chi2.sf | scipy.stats.chi2.sf | random | 452 | 8 | 0 | 0 | 1.77% (0.77% to 3.46%) |
| norm.cdf | scipy.stats.norm.cdf | grid | 25 | 0 | 0 | 0 |  |
| norm.cdf | scipy.stats.norm.cdf | random | 483 | 0 | 0 | 0 | 0.00% (0.00% to 0.76%) |
| norm.isf | scipy.stats.norm.isf | grid | 24 | 0 | 0 | 0 |  |
| norm.isf | scipy.stats.norm.isf | random | 534 | 0 | 0 | 0 | 0.00% (0.00% to 0.69%) |
| norm.ppf | scipy.stats.norm.ppf | grid | 24 | 0 | 0 | 0 |  |
| norm.ppf | scipy.stats.norm.ppf | random | 479 | 0 | 0 | 0 | 0.00% (0.00% to 0.77%) |
| norm.sf | scipy.stats.norm.sf | grid | 25 | 0 | 0 | 0 |  |
| norm.sf | scipy.stats.norm.sf | random | 504 | 0 | 0 | 0 | 0.00% (0.00% to 0.73%) |

### jax f32

| function | api | set | scored | wrong | error | out of range | rate (95% CP) |
|---|---|---|---|---|---|---|---|
| chi2.cdf | scipy.stats.chi2.cdf | grid | 306 | 3 | 0 | 0 |  |
| chi2.cdf | scipy.stats.chi2.cdf | random | 527 | 2 | 0 | 0 | 0.38% (0.05% to 1.36%) |
| chi2.sf | scipy.stats.chi2.sf | grid | 306 | 2 | 0 | 0 |  |
| chi2.sf | scipy.stats.chi2.sf | random | 452 | 2 | 0 | 0 | 0.44% (0.05% to 1.59%) |
| norm.cdf | scipy.stats.norm.cdf | grid | 25 | 0 | 0 | 0 |  |
| norm.cdf | scipy.stats.norm.cdf | random | 483 | 0 | 0 | 0 | 0.00% (0.00% to 0.76%) |
| norm.isf | scipy.stats.norm.isf | grid | 20 | 0 | 0 | 0 |  |
| norm.isf | scipy.stats.norm.isf | random | 73 | 0 | 0 | 0 | 0.00% (0.00% to 4.93%) |
| norm.ppf | scipy.stats.norm.ppf | grid | 20 | 0 | 0 | 0 |  |
| norm.ppf | scipy.stats.norm.ppf | random | 62 | 0 | 0 | 0 | 0.00% (0.00% to 5.78%) |
| norm.sf | scipy.stats.norm.sf | grid | 25 | 0 | 0 | 0 |  |
| norm.sf | scipy.stats.norm.sf | random | 504 | 0 | 0 | 0 | 0.00% (0.00% to 0.73%) |

### tfp f64

| function | api | set | scored | wrong | error | out of range | rate (95% CP) |
|---|---|---|---|---|---|---|---|
| binom.cdf | Binomial.cdf | grid | 296 | 4 | 0 | 0 |  |
| binom.sf | Binomial.survival_function | grid | 296 | 72 | 0 | 0 |  |
| chi2.cdf | Chi2.cdf | grid | 306 | 4 | 0 | 0 |  |
| chi2.cdf | Chi2.cdf | random | 527 | 4 | 0 | 0 | 0.76% (0.21% to 1.93%) |
| chi2.ppf | Chi2.quantile | grid | 396 | 10 | 0 | 0 |  |
| chi2.ppf | Chi2.quantile | random | 462 | 0 | 0 | 4 | 0.00% (0.00% to 0.80%) |
| chi2.sf | Chi2.survival_function | grid | 306 | 53 | 0 | 0 |  |
| chi2.sf | Chi2.survival_function | random | 452 | 50 | 0 | 0 | 11.06% (8.32% to 14.32%) |
| cp.lower | Beta.quantile | grid | 222 | 9 | 0 | 0 |  |
| cp.lower | Beta.quantile | random | 963 | 1 | 0 | 0 | 0.10% (0.00% to 0.58%) |
| norm.cdf | Normal.cdf | grid | 25 | 0 | 0 | 0 |  |
| norm.cdf | Normal.cdf | random | 483 | 0 | 0 | 0 | 0.00% (0.00% to 0.76%) |
| norm.ppf | Normal.quantile | grid | 24 | 0 | 0 | 0 |  |
| norm.ppf | Normal.quantile | random | 479 | 0 | 0 | 0 | 0.00% (0.00% to 0.77%) |
| norm.sf | Normal.survival_function | grid | 25 | 0 | 0 | 0 |  |
| norm.sf | Normal.survival_function | random | 504 | 0 | 0 | 0 | 0.00% (0.00% to 0.73%) |
| t.cdf | StudentT.cdf | grid | 500 | 0 | 0 | 0 |  |
| t.cdf | StudentT.cdf | random | 498 | 0 | 0 | 0 | 0.00% (0.00% to 0.74%) |
| t.ppf | StudentT.quantile | grid | 474 | 14 | 0 | 0 |  |
| t.ppf | StudentT.quantile | random | 459 | 29 | 0 | 3 | 6.32% (4.27% to 8.95%) |
| t.sf | StudentT.survival_function | grid | 500 | 0 | 0 | 0 |  |
| t.sf | StudentT.survival_function | random | 502 | 0 | 0 | 0 | 0.00% (0.00% to 0.73%) |

### tfp f32

| function | api | set | scored | wrong | error | out of range | rate (95% CP) |
|---|---|---|---|---|---|---|---|
| chi2.cdf | Chi2.cdf | grid | 306 | 3 | 0 | 0 |  |
| chi2.cdf | Chi2.cdf | random | 527 | 3 | 0 | 0 | 0.57% (0.12% to 1.65%) |
| chi2.ppf | Chi2.quantile | grid | 325 | 0 | 0 | 13 |  |
| chi2.ppf | Chi2.quantile | random | 55 | 0 | 0 | 7 | 0.00% (0.00% to 6.49%) |
| chi2.sf | Chi2.survival_function | grid | 306 | 29 | 0 | 0 |  |
| chi2.sf | Chi2.survival_function | random | 452 | 29 | 0 | 0 | 6.42% (4.34% to 9.08%) |
| norm.cdf | Normal.cdf | grid | 25 | 0 | 0 | 0 |  |
| norm.cdf | Normal.cdf | random | 483 | 0 | 0 | 0 | 0.00% (0.00% to 0.76%) |
| norm.ppf | Normal.quantile | grid | 20 | 0 | 0 | 0 |  |
| norm.ppf | Normal.quantile | random | 62 | 0 | 0 | 0 | 0.00% (0.00% to 5.78%) |
| norm.sf | Normal.survival_function | grid | 25 | 0 | 0 | 0 |  |
| norm.sf | Normal.survival_function | random | 504 | 0 | 0 | 0 | 0.00% (0.00% to 0.73%) |
| t.cdf | StudentT.cdf | grid | 500 | 20 | 0 | 0 |  |
| t.cdf | StudentT.cdf | random | 498 | 7 | 0 | 0 | 1.41% (0.57% to 2.87%) |
| t.ppf | StudentT.quantile | grid | 392 | 7 | 0 | 8 |  |
| t.ppf | StudentT.quantile | random | 59 | 1 | 0 | 7 | 1.69% (0.04% to 9.09%) |
| t.sf | StudentT.survival_function | grid | 500 | 20 | 0 | 0 |  |
| t.sf | StudentT.survival_function | random | 502 | 3 | 0 | 0 | 0.60% (0.12% to 1.74%) |

### tfp ext

| function | api | set | scored | wrong | error | out of range | rate (95% CP) |
|---|---|---|---|---|---|---|---|
| beta.cdf | Beta.cdf | grid | 110 | 2 | 0 | 0 |  |
| beta.cdf | Beta.cdf | random | 264 | 56 | 0 | 0 | 21.21% (16.44% to 26.64%) |
| beta.ppf | Beta.quantile | grid | 86 | 17 | 0 | 0 |  |
| beta.ppf | Beta.quantile | random | 228 | 156 | 0 | 0 | 68.42% (61.96% to 74.40%) |
| beta.sf | Beta.survival_function | grid | 110 | 30 | 0 | 0 |  |
| beta.sf | Beta.survival_function | random | 268 | 34 | 0 | 0 | 12.69% (8.95% to 17.27%) |

### jax ext

| function | api | set | scored | wrong | error | out of range | rate (95% CP) |
|---|---|---|---|---|---|---|---|
| beta.cdf | scipy.stats.beta.cdf | grid | 110 | 2 | 0 | 0 |  |
| beta.cdf | scipy.stats.beta.cdf | random | 264 | 59 | 0 | 0 | 22.35% (17.47% to 27.86%) |
| beta.sf | scipy.stats.beta.sf | grid | 110 | 2 | 0 | 0 |  |
| beta.sf | scipy.stats.beta.sf | random | 268 | 31 | 0 | 0 | 11.57% (8.00% to 16.02%) |

### distributionsjl f64

| function | api | set | scored | wrong | error | out of range | rate (95% CP) |
|---|---|---|---|---|---|---|---|
| binom.cdf | cdf(Binomial) | grid | 296 | 0 | 0 | 0 |  |
| binom.sf | ccdf(Binomial) | grid | 296 | 0 | 0 | 0 |  |
| chi2.cdf | cdf(Chisq) | grid | 306 | 1 | 0 | 0 |  |
| chi2.cdf | cdf(Chisq) | random | 527 | 0 | 0 | 0 | 0.00% (0.00% to 0.70%) |
| chi2.isf | cquantile(Chisq) | grid | 404 | 2 | 4 | 0 |  |
| chi2.isf | cquantile(Chisq) | random | 471 | 10 | 17 | 0 | 2.12% (1.02% to 3.87%) |
| chi2.ppf | quantile(Chisq) | grid | 396 | 0 | 0 | 0 |  |
| chi2.ppf | quantile(Chisq) | random | 462 | 3 | 0 | 4 | 0.65% (0.13% to 1.89%) |
| chi2.sf | ccdf(Chisq) | grid | 306 | 0 | 0 | 0 |  |
| chi2.sf | ccdf(Chisq) | random | 452 | 0 | 0 | 0 | 0.00% (0.00% to 0.81%) |
| cp.lower | quantile(Beta) | grid | 222 | 0 | 0 | 0 |  |
| cp.lower | quantile(Beta) | random | 963 | 0 | 0 | 0 | 0.00% (0.00% to 0.38%) |
| cp.upper | cquantile(Beta) | grid | 222 | 0 | 0 | 0 |  |
| cp.upper | cquantile(Beta) | random | 896 | 0 | 0 | 0 | 0.00% (0.00% to 0.41%) |
| f.cdf | cdf(FDist) | grid | 240 | 1 | 0 | 0 |  |
| f.cdf | cdf(FDist) | random | 487 | 37 | 0 | 0 | 7.60% (5.41% to 10.32%) |
| f.isf | cquantile(FDist) | grid | 355 | 74 | 0 | 0 |  |
| f.isf | cquantile(FDist) | random | 422 | 149 | 1 | 0 | 35.31% (30.75% to 40.08%) |
| f.ppf | quantile(FDist) | grid | 351 | 13 | 0 | 0 |  |
| f.ppf | quantile(FDist) | random | 441 | 35 | 0 | 3 | 7.94% (5.59% to 10.87%) |
| f.sf | ccdf(FDist) | grid | 240 | 15 | 0 | 0 |  |
| f.sf | ccdf(FDist) | random | 522 | 83 | 0 | 0 | 15.90% (12.87% to 19.32%) |
| fisher.pvalue | ccdf(Hypergeometric) | grid | 15 | 0 | 0 | 0 |  |
| fisher.pvalue | cdf(Hypergeometric) | grid | 15 | 0 | 0 | 0 |  |
| ks.cdf | cdf(KSDist) | grid | 111 | 4 | 0 | 0 |  |
| ks.sf | ccdf(KSDist) | grid | 111 | 3 | 0 | 0 |  |
| norm.cdf | cdf(Normal) | grid | 25 | 0 | 0 | 0 |  |
| norm.cdf | cdf(Normal) | random | 483 | 0 | 0 | 0 | 0.00% (0.00% to 0.76%) |
| norm.isf | cquantile(Normal) | grid | 24 | 0 | 0 | 0 |  |
| norm.isf | cquantile(Normal) | random | 534 | 0 | 0 | 0 | 0.00% (0.00% to 0.69%) |
| norm.ppf | quantile(Normal) | grid | 24 | 0 | 0 | 0 |  |
| norm.ppf | quantile(Normal) | random | 479 | 0 | 0 | 0 | 0.00% (0.00% to 0.77%) |
| norm.sf | ccdf(Normal) | grid | 25 | 0 | 0 | 0 |  |
| norm.sf | ccdf(Normal) | random | 504 | 0 | 0 | 0 | 0.00% (0.00% to 0.73%) |
| t.cdf | cdf(TDist) | grid | 500 | 48 | 0 | 0 |  |
| t.cdf | cdf(TDist) | random | 498 | 16 | 0 | 0 | 3.21% (1.85% to 5.17%) |
| t.isf | cquantile(TDist) | grid | 474 | 146 | 0 | 0 |  |
| t.isf | cquantile(TDist) | random | 450 | 135 | 0 | 1 | 30.00% (25.80% to 34.47%) |
| t.ppf | quantile(TDist) | grid | 474 | 146 | 0 | 0 |  |
| t.ppf | quantile(TDist) | random | 459 | 150 | 0 | 3 | 32.68% (28.40% to 37.18%) |
| t.sf | ccdf(TDist) | grid | 500 | 48 | 0 | 0 |  |
| t.sf | ccdf(TDist) | random | 502 | 11 | 0 | 0 | 2.19% (1.10% to 3.89%) |

### distributionsjl ks

| function | api | set | scored | wrong | error | out of range | rate (95% CP) |
|---|---|---|---|---|---|---|---|
| ks.cdf | cdf(KSDist) | sweep | 254 | 74 | 0 | 0 |  |
| ks.sf | ccdf(KSDist) | sweep | 254 | 76 | 0 | 0 |  |

### distributionsjl ext

| function | api | set | scored | wrong | error | out of range | rate (95% CP) |
|---|---|---|---|---|---|---|---|
| beta.cdf | cdf(Beta) | grid | 110 | 0 | 0 | 0 |  |
| beta.cdf | cdf(Beta) | random | 264 | 0 | 0 | 0 | 0.00% (0.00% to 1.39%) |
| beta.isf | cquantile(Beta) | grid | 87 | 0 | 0 | 0 |  |
| beta.isf | cquantile(Beta) | random | 266 | 2 | 0 | 0 | 0.75% (0.09% to 2.69%) |
| beta.ppf | quantile(Beta) | grid | 86 | 2 | 0 | 0 |  |
| beta.ppf | quantile(Beta) | random | 228 | 11 | 0 | 0 | 4.82% (2.43% to 8.47%) |
| beta.sf | ccdf(Beta) | grid | 110 | 0 | 0 | 0 |  |
| beta.sf | ccdf(Beta) | random | 268 | 0 | 0 | 0 | 0.00% (0.00% to 1.37%) |

### Classes

| library | run | class | count | worst |
|---|---|---|---|---|
| distributionsjl | ext | D-beta-quantile: beta_inc_inv sticks near 1e-154 at tiny p | 15 | beta.ppf(a=0.129037;b=1.93474;p=1.44932e-36) quantile(Beta) -> 1.8595226279382035e-156 |
| distributionsjl | f64 | D-chi2: gamma_inc/gamma_inc_inv failure at extreme p (SpecialFunctions) | 16 | chi2.isf(df=0.5;p=1e-200) cquantile(Chisq) -> 133.088413502053 |
| distributionsjl | f64 | D-f-quantile: x/(1-x) after beta_inc_inv | 271 | f.isf(dfn=1.0;dfd=1.0;p=1e-100) cquantile(FDist) -> Inf |
| distributionsjl | f64 | D-f-tail: beta argument near 1 in fdist(c)cdf | 136 | f.sf(dfn=10000.0;dfd=10.0;x=100000000.0) ccdf(FDist) -> 2.6093796176776288e-39 |
| distributionsjl | f64 | D-ks-convention: d one rounding above 1/(2n), not a defect | 2 | ks.cdf(n=10;d=0.05) cdf(KSDist) -> 0.0 |
| distributionsjl | f64 | D-ks-durbin: (2h-1) without ^m in cdf_durbin | 4 | ks.cdf(n=5;d=0.25) cdf(KSDist) -> 0.16755000000000006 |
| distributionsjl | f64 | D-ks-marginal: LRE about 11, Miller or Durbin rounding | 1 | ks.sf(n=100;d=0.2) ccdf(KSDist) -> 0.0005551927328074682 |
| distributionsjl | f64 | D-t-cdf: inv(1+nu/x^2) collapses to 1 (noted in StatsFuns#229) | 123 | t.cdf(df=7.5;x=-10000000000.0) cdf(TDist) -> 0.0 |
| distributionsjl | f64 | D-t-quantile: t through fdistinvccdf (open PR StatsFuns#229) | 577 | t.ppf(df=0.1;p=1e-30) quantile(TDist) -> -Inf |
| distributionsjl | f64 | E D-chi2: gamma_inc/gamma_inc_inv failure at extreme p (SpecialFunctions) | 21 | chi2.isf(df=0.5;p=1e-100) cquantile(Chisq) -> ERR:DomainError with -186. |
| distributionsjl | f64 | E D-f-quantile: x/(1-x) after beta_inc_inv | 1 | f.isf(dfn=55.1294;dfd=8834.02;p=1.14424e-289) cquantile(FDist) -> HANG |
| distributionsjl | ks | D-ks-durbin: (2h-1) without ^m in cdf_durbin | 148 | ks.cdf(n=100;d=0.010234546) cdf(KSDist) -> 2.6540936472341694e-40 |
| distributionsjl | ks | D-ks-marginal: LRE about 11, Miller or Durbin rounding | 2 | ks.sf(n=100;d=0.19517432) ccdf(KSDist) -> 0.0008205634011794194 |
| jax | ext | J-betainc: stats.beta cdf/sf through betainc (known: jax#41678) | 94 | beta.sf(a=0.818217;b=46024.4;x=2.97479e-05) scipy.stats.beta.sf -> 0.19279370507607618 |
| jax | f32 | J-large-a: gammainc/gammaincc at large a (known in part: jax#41201, PR #41255) | 9 | chi2.sf(df=1000000.0;x=1000000.0) scipy.stats.chi2.sf -> 0.43354201316833496 |
| jax | f64 | J-large-a: gammainc/gammaincc at large a (known in part: jax#41201, PR #41255) | 23 | chi2.sf(df=1000000.0;x=1000000.0) scipy.stats.chi2.sf -> 0.4998119366737945 |
| tfp | ext | P-beta-binom: betainc-based, see claims | 3 | beta.cdf(a=0.158885;b=76799.2;x=1.85766e-06) Beta.cdf -> 0.7741410975517333 |
| tfp | ext | P-beta-quantile: Beta.quantile wrong at p=0.5 (known: tfp#2034) and at tiny p or NaN | 173 | beta.ppf(a=86572.4;b=3.57565;p=4.11404e-269) Beta.quantile -> nan |
| tfp | ext | P-marginal: LRE >= 10, betainc-based | 55 | beta.cdf(a=15.2866;b=60828.1;x=2.31582e-05) Beta.cdf -> 1.7540573369271445e-11 |
| tfp | ext | P-sf: survival_function falls back to 1 - cdf | 64 | beta.sf(a=50.0;b=200.0;x=0.99) Beta.survival_function -> 1.1102230246251565e-16 |
| tfp | f32 | P-igamma: tf igamma/quantile at large df | 6 | chi2.cdf(df=1000000.0;x=1000000.0) Chi2.cdf -> 0.4318324625492096 |
| tfp | f32 | P-sf: survival_function falls back to 1 - cdf | 58 | chi2.sf(df=1.8000600337982178;x=90.59819793701172) Chi2.survival_function -> 0.0 |
| tfp | f32 | P-t-quantile: far-tail StudentT.quantile, inf or digits lost | 8 | t.ppf(df=0.10000000149011612;p=0.0010000000474974513) StudentT.quantile -> -inf |
| tfp | f32 | P-t32: StudentT.cdf/sf NaN in binary32 at large df or |x| | 50 | t.cdf(df=10000.0;x=-10000000000.0) StudentT.cdf -> nan |
| tfp | f64 | P-igamma: tf igamma/quantile at large df | 18 | chi2.cdf(df=1000000.0;x=1000000.0) Chi2.cdf -> 0.4978423123230248 |
| tfp | f64 | P-marginal: LRE >= 10, betainc-based | 14 | cp.lower(n=10000;k=1;alpha=0.1) Beta.quantile -> 5.1293162838177655e-06 |
| tfp | f64 | P-sf: survival_function falls back to 1 - cdf | 175 | binom.sf(n=200;p=0.001;k=199) Binomial.survival_function -> 1.1102230246251565e-16 |
| tfp | f64 | P-t-quantile: far-tail StudentT.quantile, inf or digits lost | 43 | t.ppf(df=0.1;p=1e-30) StudentT.quantile -> -inf |
| torch | f32 | T-cancel: 1+erf cancels for x<0 (known: pytorch#192295, PR #189234) | 56 | norm.cdf(x=-7.609029769897461) distributions.Normal.cdf -> 0.0 |
| torch | f32 | T-icdf: erfinv(2p-1) loses small p | 57 | norm.ppf(p=1.0000000031710769e-30) distributions.Normal.icdf -> -inf |
| torch | f64 | T-cancel: 1+erf cancels for x<0 (known: pytorch#192295, PR #189234) | 106 | norm.cdf(x=-12.9514) distributions.Normal.cdf -> 0.0 |
| torch | f64 | T-icdf: erfinv(2p-1) loses small p | 477 | norm.ppf(p=1e-300) distributions.Normal.icdf -> -inf |
| torch | f64 | T-marginal: igamma outside that series, LRE >= 11 | 3 | chi2.sf(df=6192.11;x=9437.8) special.gammaincc -> 1.0473506360548089e-140 |
| torch | f64 | T-pi: single-precision pi in the double igamma series (Math.h) | 51 | chi2.cdf(df=534.671;x=400.711) distributions.Chi2.cdf -> 3.7566991336529045e-06 |
