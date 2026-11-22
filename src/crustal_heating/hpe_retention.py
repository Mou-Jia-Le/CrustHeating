"""Mass-conserving, two-layer model with lower-crust HPE retention.

R is the fraction of the TOTAL HPE mass retained in the lower crust,
not its concentration relative to the upper crust. All HPEs share R,
and both layers retain the original model's common density and conductivity.
The observed upper-crust enrichment E(H) is held fixed.
"""

from __future__ import annotations

import numpy as np

from .thermal import DEFAULT_CONSTANTS, enrichment_factor


def layer_properties(thickness_m, retention, constants=DEFAULT_CONSTANTS):
    """Return upper-layer thickness (m) and upper/lower heat production (W/m³)."""
    H, R = np.broadcast_arrays(
        np.asarray(thickness_m, dtype=float), np.asarray(retention, dtype=float)
    )
    if np.any(~np.isfinite(H)) or np.any(H <= 0):
        raise ValueError("Crustal thickness must be finite and positive.")
    if np.any(~np.isfinite(R)) or np.any((R < 0) | (R > 1)):
        raise ValueError("R must be between 0 and 1.")
    E = enrichment_factor(H)
    if np.any(E < 1):
        raise ValueError("This enriched upper-crust model requires E(H) >= 1.")
    h = (1 - R) * H / E
    upper = constants.heat_present_w_m3 * E
    lower = np.divide(
        R * constants.heat_present_w_m3 * H, H - h,
        out=np.zeros_like(H), where=(H - h) > 0,
    )
    return h, upper, lower


def geotherm(depth_m, thickness_m, retention, constants=DEFAULT_CONSTANTS):
    """Exact conductive solution; avoids numerical smearing at the interface."""
    z = np.asarray(depth_m, dtype=float)
    H = np.asarray(thickness_m, dtype=float)
    if np.any(~np.isfinite(z)) or np.any((z < 0) | (z > H)):
        raise ValueError("Depth must lie between the surface and crustal base.")
    h, upper, lower = layer_properties(H, retention, constants)
    top_integral = np.where(z <= h, h * z - z**2 / 2, h**2 / 2)
    return constants.surface_temperature_c + (
        constants.basal_heat_flux_w_m2 * z
        + lower * (H * z - z**2 / 2)
        + (upper - lower) * top_integral
    ) / constants.conductivity_w_m_k


def basal_temperature(thickness_m, retention, constants=DEFAULT_CONSTANTS):
    """T_b = T_s + q_b H/k + A0 H² [R + (1-R)/E(H)]/(2k)."""
    H = np.asarray(thickness_m, dtype=float)
    h, _, _ = layer_properties(H, retention, constants)
    return constants.surface_temperature_c + (
        constants.basal_heat_flux_w_m2 * H
        + constants.heat_present_w_m3 * H * (h + np.asarray(retention) * H) / 2
    ) / constants.conductivity_w_m_k
