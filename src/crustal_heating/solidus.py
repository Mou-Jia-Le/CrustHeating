"""Eclogite solidus calculations."""

import numpy as np


def get_eclogite_solidus(depth_km, rho=2870):
    """Return dry and water-saturated eclogite solidus temperatures in deg C."""

    try:
        import pyMelt
    except ImportError as exc:
        raise ImportError(
            "pyMelt is required for eclogite solidus calculations. Install it with `pip install pyMelt`."
        ) from exc

    depth = np.asarray(depth_km, dtype=float)
    dry_eclogite = pyMelt.lithologies.matthews.eclogite()
    wet_eclogite = pyMelt.hydrousLithology(dry_eclogite, 10.0)
    pressure_gpa = rho * 9.81 * depth * 1000.0 / 1e9

    if pressure_gpa.ndim == 0:
        return dry_eclogite.TSolidus(float(pressure_gpa)), wet_eclogite.TSolidus(float(pressure_gpa))

    dry = np.array([dry_eclogite.TSolidus(float(p)) for p in pressure_gpa])
    wet = np.array([wet_eclogite.TSolidus(float(p)) for p in pressure_gpa])
    return dry, wet
