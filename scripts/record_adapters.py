"""Record a few outputs per target adapter into tests/data/recorded.json.

Run after changing an adapter or upgrading a target; tests/test_targets.py replays the
recording when the installed target version matches.
"""

from __future__ import annotations

import json
from pathlib import Path

from certtables.grid import parse_id
from certtables.targets import get_target

POINTS = {
    "scipy": [
        "norm.isf(p=1e-300)",
        "t.cdf(df=2.5;x=-30.0)",
        "chi2.ppf(df=0.5;p=1e-10)",
        "f.sf(dfn=3.0;dfd=7.5;x=10.0)",
        "cp.lower(n=100;k=2;alpha=0.05)",
        "binom.sf(n=50;p=0.1;k=5)",
        "fisher.pvalue(table=8,2,1,5;alternative=two-sided)",
        "wilcoxon.pvalue(n=10;w=13;alternative=greater)",
        "mannwhitney.pvalue(m=5;n=10;u=12;alternative=less)",
        "ks.sf(n=20;d=0.3)",
    ],
    "statsmodels": [
        "cp.upper(n=100;k=2;alpha=0.05)",
        "binom.cdf(n=50;p=0.1;k=5)",
        "binom.sf(n=50;p=0.1;k=5)",
    ],
    "mpmath": [
        "norm.sf(x=10.0)",
        "t.sf(df=2.5;x=30.0)",
        "chi2.cdf(df=3.0;x=0.5)",
        "f.cdf(dfn=3.0;dfd=7.5;x=2.0)",
    ],
    "spotify": ["gst.bound(design=ld-power;alpha=0.05;t=1/3,2/3,1;look=2)"],
    "jax": ["norm.isf(p=1e-10)", "chi2.sf(df=3.0;x=5.0)"],
    "printed": ["gst.bound(design=obf;alpha=0.05;t=1/3,2/3,1;look=3)"],
}


def main() -> None:
    out = {}
    for name, ids in POINTS.items():
        t = get_target(name)
        out[name] = {
            "version": t.version(),
            "values": {pid: repr(t.evaluate(parse_id(pid))) for pid in ids},
        }
    path = Path(__file__).resolve().parents[1] / "tests" / "data" / "recorded.json"
    path.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(path.name)


if __name__ == "__main__":
    main()
