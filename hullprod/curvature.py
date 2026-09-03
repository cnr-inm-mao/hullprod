"""Validated mesh-curvature backend used by the paper release contract.

The implementation is the Rusinkiewicz (3DPVT 2004) curvature-tensor
recovery already validated by HullProd's analytical and CAD-reference
benchmarks.  This narrow module keeps the ordinary metric pipeline separate
from the other research estimators in ``experimental_standard_curvature``.
"""

from __future__ import annotations

import trimesh

from .experimental_standard_curvature import DifferentialEstimate, rusinkiewicz_curvature


def estimate_mesh_curvature(mesh: trimesh.Trimesh) -> DifferentialEstimate:
    """Return signed conventional H and intrinsic K from tensor recovery.

    Mean curvature uses ``H=(k1+k2)/2``.  Its sign follows the consistently
    wound mesh orientation; Gaussian curvature is orientation invariant.
    Curvature-derivative recovery is deliberately skipped because production
    fairness uses the separately validated P1 face-gradient integral.
    """
    return rusinkiewicz_curvature(mesh, derivatives=False)
