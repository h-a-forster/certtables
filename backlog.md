# Backlog

## In progress

## Waiting on me

- Decide whether to file the drafts in docs/upstream/ (SciPy f.isf, the two new t-quantile modes on scipy#25968, f.ppf, spotify-confidence); nothing is filed.
- Decide whether the repository stays private or goes public.

## Decided

- 2026-11-06: Finish rather than drop: the method has prior art (Wang and Kennedy 1994; Baharev et al. 2017), but no public interval-certified table or census of the current stack exists.
- 2026-11-06: Quantile searches start at 1, not at a SciPy value, so the table does not depend on an implementation under test.
- 2026-11-06: Release notes live in docs/release-notes-<version>.md; there is no CHANGELOG.
- 2026-11-06: K = 4, 5 boundaries stay uncertified references; R packages are not targets in this release.

## Proposed

- Certified K = 4, 5 group-sequential boundaries by a different method, since nesting another adaptive integral is too slow.
- Beta quantiles, noncentral t and F, and Poisson or discrete quantiles as table families, published as a new table version.
- gsDesign, rpact and ldbounds scored against the boundaries through a results file, so R need not be a dependency.
- Census rerun on Linux and macOS SciPy wheels.
- Census rerun when SciPy ships a Boost.Math with the Student t and incomplete-beta fixes.
