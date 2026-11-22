"""Core thermal and heat-production calculations."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import json
from .paths import CALIBRATION_PATH


if not CALIBRATION_PATH.exists():
    raise FileNotFoundError("Run crustal-heating calibrate to generate regression_calibration.json first.")
_CALIBRATION = json.loads(CALIBRATION_PATH.read_text(encoding="utf-8"))
K2O_THICKNESS_SLOPE_WT_PER_KM = _CALIBRATION["k2o_slope_wt_per_km"]
K2O_THICKNESS_INTERCEPT_WT = _CALIBRATION["k2o_intercept_wt"]
TH_PER_K2O_ZERO_INTERCEPT = _CALIBRATION["fits"]["Th"]["slope"]
U_PER_K2O_ZERO_INTERCEPT = _CALIBRATION["fits"]["U"]["slope"]
TEN_KM_CRUST_THICKNESS_KM = 10.0
DENSITY_KG_M3 = 2870.0


def k2o_from_thickness(crustal_thickness_km):
    """Return K2O concentration from crustal thickness in kilometers."""

    thickness_km = np.asarray(crustal_thickness_km, dtype=float)
    return K2O_THICKNESS_SLOPE_WT_PER_KM * thickness_km + K2O_THICKNESS_INTERCEPT_WT


TEN_KM_CRUST_K2O_WT = float(k2o_from_thickness(TEN_KM_CRUST_THICKNESS_KM))
TEN_KM_CRUST_TH_PPM = TH_PER_K2O_ZERO_INTERCEPT * TEN_KM_CRUST_K2O_WT
TEN_KM_CRUST_U_PPM = U_PER_K2O_ZERO_INTERCEPT * TEN_KM_CRUST_K2O_WT


def heat_generation_from_concentrations(k2o_wt, th_ppm, u_ppm, density_kg_m3=DENSITY_KG_M3):
    """Return radiogenic heat production in micro-W/m^3 from K2O, Th, and U."""

    return density_kg_m3 * (9.83 * u_ppm + 2.62 * th_ppm + 2.81 * k2o_wt) * 1e-5


def heat_generation_fractions(k2o_wt, th_ppm, u_ppm):
    """Return fractional U, Th, and K2O contributions to heat production."""

    u_term = 9.83 * u_ppm
    th_term = 2.62 * th_ppm
    k_term = 2.81 * k2o_wt
    total = u_term + th_term + k_term
    return u_term / total, th_term / total, k_term / total


TEN_KM_CRUST_HEAT_PRODUCTION_W_M3 = float(
    heat_generation_from_concentrations(
        TEN_KM_CRUST_K2O_WT,
        TEN_KM_CRUST_TH_PPM,
        TEN_KM_CRUST_U_PPM,
    )
    * 1e-6
)
TEN_KM_CRUST_U_HEAT_FRACTION, TEN_KM_CRUST_TH_HEAT_FRACTION, TEN_KM_CRUST_K_HEAT_FRACTION = (
    heat_generation_fractions(TEN_KM_CRUST_K2O_WT, TEN_KM_CRUST_TH_PPM, TEN_KM_CRUST_U_PPM)
)
U238_SHARE_OF_U_HEAT = 0.96
U235_SHARE_OF_U_HEAT = 0.04


@dataclass(frozen=True)
class ThermalConstants:
    """Default constants used by the original thermal scripts."""

    heat_present_w_m3: float = TEN_KM_CRUST_HEAT_PRODUCTION_W_M3
    conductivity_w_m_k: float = 2.1
    basal_heat_flux_w_m2: float = 0.015
    surface_temperature_c: float = 0.0
    u238_fraction: float = TEN_KM_CRUST_U_HEAT_FRACTION * U238_SHARE_OF_U_HEAT
    u235_fraction: float = TEN_KM_CRUST_U_HEAT_FRACTION * U235_SHARE_OF_U_HEAT
    th_fraction: float = TEN_KM_CRUST_TH_HEAT_FRACTION
    k_fraction: float = TEN_KM_CRUST_K_HEAT_FRACTION
    half_life_u238_years: float = 4.4683e9
    half_life_u235_years: float = 0.70348e9
    half_life_th_years: float = 14.1e9
    half_life_k_years: float = 1.262e9

    @property
    def u_total_fraction(self) -> float:
        return self.u238_fraction + self.u235_fraction


DEFAULT_CONSTANTS = ThermalConstants()


def enrichment_factor(crustal_thickness_m):
    """Return E = C/C_10km using 10 km crust concentration as the baseline."""

    thickness_km = np.asarray(crustal_thickness_m, dtype=float) / 1000.0
    return k2o_from_thickness(thickness_km) / TEN_KM_CRUST_K2O_WT


def concentration_series(crustal_thickness_km):
    """Calculate K2O, Th, U, enrichment factors, and heat production."""

    thickness_km = np.asarray(crustal_thickness_km, dtype=float)
    k2o_wt = k2o_from_thickness(thickness_km)
    th_ppm = TH_PER_K2O_ZERO_INTERCEPT * k2o_wt
    u_ppm = U_PER_K2O_ZERO_INTERCEPT * k2o_wt
    heat_generation = heat_generation_from_concentrations(k2o_wt, th_ppm, u_ppm)

    thickness_m = thickness_km * 1000.0
    e = enrichment_factor(thickness_m)
    return {
        "k2o_wt": k2o_wt,
        "th_ppm": th_ppm,
        "u_ppm": u_ppm,
        "heat_generation_uW_m3": heat_generation,
        "e": e,
    }


def radiogenic_heat_through_time(time_years, constants: ThermalConstants = DEFAULT_CONSTANTS):
    """Return U238, U235, Th, K, and total heat production before present."""

    time = np.asarray(time_years, dtype=float)
    heat = constants.heat_present_w_m3
    heat_u238 = heat * constants.u238_fraction * np.exp(np.log(2.0) * time / constants.half_life_u238_years)
    heat_u235 = heat * constants.u235_fraction * np.exp(np.log(2.0) * time / constants.half_life_u235_years)
    heat_th = heat * constants.th_fraction * np.exp(np.log(2.0) * time / constants.half_life_th_years)
    heat_k = heat * constants.k_fraction * np.exp(np.log(2.0) * time / constants.half_life_k_years)
    heat_total = heat_u238 + heat_u235 + heat_th + heat_k
    return heat_u238, heat_u235, heat_th, heat_k, heat_total


def basal_temperature_undifferentiated(
    crustal_thickness_m,
    heat_production_w_m3,
    constants: ThermalConstants = DEFAULT_CONSTANTS,
):
    """Calculate basal temperature for a uniform heat-production crust."""

    thickness = np.asarray(crustal_thickness_m, dtype=float)
    heat = np.asarray(heat_production_w_m3, dtype=float)
    return (
        heat / (2 * constants.conductivity_w_m_k) * thickness**2
        + constants.basal_heat_flux_w_m2 / constants.conductivity_w_m_k * thickness
        + constants.surface_temperature_c
    )


def basal_temperature_differentiated(
    crustal_thickness_m,
    heat_u238_w_m3,
    heat_u235_w_m3,
    heat_th_w_m3,
    heat_k_w_m3,
    constants: ThermalConstants = DEFAULT_CONSTANTS,
):
    """Calculate basal temperature using the enrichment factor E."""

    thickness = np.asarray(crustal_thickness_m, dtype=float)
    e = enrichment_factor(thickness)
    if np.any(e <= 0):
        raise ValueError("Enrichment factors must be positive. Check thickness units.")

    effective_heat = (
        np.asarray(heat_u238_w_m3)
        + np.asarray(heat_u235_w_m3)
        + np.asarray(heat_th_w_m3)
        + np.asarray(heat_k_w_m3)
    ) / e
    return (
        constants.basal_heat_flux_w_m2 / constants.conductivity_w_m_k * thickness
        + thickness**2 / (2 * constants.conductivity_w_m_k) * effective_heat
        + constants.surface_temperature_c
    )


def geotherm_differentiated(
    depth_m,
    crustal_thickness_m,
    bulk_heat_production_w_m3,
    constants: ThermalConstants = DEFAULT_CONSTANTS,
):
    """Return the steady geotherm for an enriched upper crust over a heat-free lower crust.

    Depth is measured from the surface to the Moho. The enrichment model
    requires E >= 1 so that the enriched layer of thickness H/E fits in the crust.
    """
    depth = np.asarray(depth_m, dtype=float)
    e = enrichment_factor(crustal_thickness_m)
    if np.any(e < 1):
        raise ValueError("The enriched layer must fit within the crust (E >= 1).")

    enriched_thickness = crustal_thickness_m / e
    depth_in_enriched_layer = np.minimum(depth, enriched_thickness)
    upper_heat = bulk_heat_production_w_m3 * e
    # The radiogenic temperature contribution is constant below the enriched layer.
    radiogenic_term = upper_heat * depth_in_enriched_layer * (
        enriched_thickness - depth_in_enriched_layer / 2
    )
    return (
        constants.surface_temperature_c
        + (constants.basal_heat_flux_w_m2 * depth + radiogenic_term)
        / constants.conductivity_w_m_k
    )


def cumulative_trapezoid_forward(x, y):
    """Integrate from x[0] to each x[i]."""

    x_values = np.asarray(x, dtype=float)
    y_values = np.asarray(y, dtype=float)
    output = np.zeros_like(y_values, dtype=float)
    output[1:] = np.cumsum(0.5 * (y_values[:-1] + y_values[1:]) * np.diff(x_values))
    return output


def cumulative_trapezoid_reverse(x, y):
    """Integrate from each x[i] to x[-1]."""

    x_values = np.asarray(x, dtype=float)
    y_values = np.asarray(y, dtype=float)
    output = np.zeros_like(y_values, dtype=float)
    output[:-1] = np.cumsum((0.5 * (y_values[:-1] + y_values[1:]) * np.diff(x_values))[::-1])[::-1]
    return output


def temperature_from_heat_profile(
    depth_m,
    heat_profile_w_m3,
    constants: ThermalConstants = DEFAULT_CONSTANTS,
):
    """Solve the 1D steady conductive profile for arbitrary heat production."""

    depth = np.asarray(depth_m, dtype=float)
    heat_profile = np.asarray(heat_profile_w_m3, dtype=float)
    heat_below_depth = cumulative_trapezoid_reverse(depth, heat_profile)
    gradient = (constants.basal_heat_flux_w_m2 + heat_below_depth) / constants.conductivity_w_m_k
    return constants.surface_temperature_c + cumulative_trapezoid_forward(depth, gradient)


def step_profile(depth_m, enriched_depth_m, upper_heat_w_m3):
    """Return a heat-production profile enriched from the surface to depth h."""

    depth = np.asarray(depth_m, dtype=float)
    profile = np.zeros_like(depth, dtype=float)
    profile[depth <= enriched_depth_m] = upper_heat_w_m3
    return profile


def build_heat_profiles(
    crustal_thickness_m: float = 60_000.0,
    erosion_depth_m: float = 40_000.0,
    constants: ThermalConstants = DEFAULT_CONSTANTS,
):
    """Build undifferentiated, enriched, and eroded heat-production profiles."""

    e = float(enrichment_factor(crustal_thickness_m))
    if e <= 0:
        raise ValueError("Enrichment factors must be positive. Check thickness units.")

    h_u = crustal_thickness_m / e
    h_th = crustal_thickness_m / e
    h_k = crustal_thickness_m / e
    depth = np.unique(np.r_[np.linspace(0, crustal_thickness_m, 300), h_u, h_th, h_k, erosion_depth_m])

    heat = constants.heat_present_w_m3
    u_profile = step_profile(depth, h_u, heat * constants.u_total_fraction * e)
    th_profile = step_profile(depth, h_th, heat * constants.th_fraction * e)
    k_profile = step_profile(depth, h_k, heat * constants.k_fraction * e)
    differentiated = u_profile + th_profile + k_profile
    undifferentiated = np.full_like(depth, heat)
    eroded = np.zeros_like(depth)
    eroded[depth > erosion_depth_m] = np.nan

    return {
        "depth_m": depth,
        "u": u_profile,
        "th": th_profile,
        "k": k_profile,
        "differentiated": differentiated,
        "undifferentiated": undifferentiated,
        "eroded": eroded,
        "h_u": h_u,
        "h_th": h_th,
        "h_k": h_k,
        "e": e,
    }


def first_crossing_time(time_years, delta_temperature):
    """Interpolate the first crossing time where delta temperature changes sign."""

    time = np.asarray(time_years, dtype=float)
    delta = np.asarray(delta_temperature, dtype=float)
    crossings = np.where(np.diff(np.sign(delta)) != 0)[0]
    if crossings.size == 0:
        return np.nan

    index = crossings[0]
    x0, x1 = time[index], time[index + 1]
    y0, y1 = delta[index], delta[index + 1]
    if y1 == y0:
        return x0
    return x0 - y0 * (x1 - x0) / (y1 - y0)
