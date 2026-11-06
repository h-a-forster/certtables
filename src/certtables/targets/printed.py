"""Printed constants: Pocock C_P(K, alpha) and O'Brien-Fleming C_B(K, alpha) for two-sided
tests with K equally spaced looks, as tabulated in Jennison and Turnbull (2000), *Group
Sequential Methods with Applications to Clinical Trials*, Tables 2.1 (C_P) and 2.3 (C_B)
(alpha = 0.05 and 0.01, K = 1..5; three decimals). K = 1 has no design on the grid, so its
constants are transcribed but not scored. The values were transcribed for this release and not
re-read against a physical copy; a disagreement is reported as a transcription or table
discrepancy, not as an error in the book.

Scored against half a unit in the third decimal (absolute 0.0005), the precision printed.
"""

from __future__ import annotations

from certtables.grid import Point
from certtables.targets.base import Target

#: (design, alpha) -> constants for K = 1..5. Pocock: c_k = C_P; OBF: c_k = C_B sqrt(K/k).
TABLE_2_1 = {
    ("pocock", 0.05): (1.960, 2.178, 2.289, 2.361, 2.413),
    ("pocock", 0.01): (2.576, 2.772, 2.873, 2.939, 2.986),
    ("obf", 0.05): (1.960, 1.977, 2.004, 2.024, 2.040),
    ("obf", 0.01): (2.576, 2.580, 2.595, 2.609, 2.621),
}
ABS_PRINTED = 5e-4

_EQUAL = {"1/2,1": 2, "1/3,2/3,1": 3, "1/4,1/2,3/4,1": 4, "1/5,2/5,3/5,4/5,1": 5}


class PrintedTarget(Target):
    name = "printed"
    description = "Jennison & Turnbull (2000) Tables 2.1, 2.3: Pocock C_P, O'Brien-Fleming C_B"

    def supports(self, pt: Point) -> bool:
        if pt.family != "gst":
            return False
        p = pt.p
        key = (p["design"], float(p["alpha"]))
        if key not in TABLE_2_1 or p["t"] not in _EQUAL:
            return False
        # C_P is every look's boundary; C_B is the final look's.
        return p["design"] == "pocock" or int(p["look"]) == _EQUAL[str(p["t"])]

    def tolerance(self, pt: Point) -> tuple[float, float]:
        return 0.0, ABS_PRINTED

    def evaluate(self, pt: Point) -> float:
        p = pt.p
        return TABLE_2_1[(str(p["design"]), float(p["alpha"]))][_EQUAL[str(p["t"])] - 1]
