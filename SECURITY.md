# Security policy

Security fixes are made for the latest release.

## Reporting

Report vulnerabilities privately through the repository's **Security** tab ("Report a
vulnerability"). Do not open a public issue. Include the certtables version, the operating
system, the impact, and steps that reproduce it.

## Scope

certtables computes enclosures locally and reads and writes CSV and JSON under `data/`,
`results/` and `runs/` in the working directory. It makes no network requests and calls no
models. Target adapters import and call the libraries under test (SciPy, statsmodels,
mpmath, spotify-confidence, JAX) in-process.

In scope:

- certtables reading or writing files outside those directories;
- code execution triggered by the contents of a table or results file (point ids are parsed
  as numbers and plain strings only);
- local paths or machine details written to committed outputs.

Out of scope:

- vulnerabilities in dependencies (report those upstream);
- a wrong reference value: that is a correctness bug, please open an ordinary issue with
  the row id (see CONTRIBUTING.md). It is the most important kind of report.
