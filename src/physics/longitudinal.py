from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd

@dataclass(frozen=True)
class VehicleParams:
    mass_kg: float
    Cd: float
    area_m2: float
    crr: float
    tyre_radius_m: float
    rho_air: float = 1.225 #can we change this after?
    g: float = 9.81

def add_longitudinal_forces(df: pd.DataFrame, params: VehicleParams, *,
                            v_col: str = "speed_ms", a_col: str = "accel_ms2",
                            grade_rad: float = 0.0) -> pd.DataFrame:
    v = df[v_col].astype(float).to_numpy()
    a = df[a_col].astype(float).to_numpy()

    df["F_aero_N"] = 0.5 * params.rho_air * params.Cd * params.area_m2 * v**2
    df["F_roll_N"] = params.crr * params.mass_kg * params.g * np.cos(grade_rad)
    df["F_slope_N"] = params.mass_kg * params.g * np.sin(grade_rad)
    df["F_inertia_N"] = params.mass_kg * a
    df["F_trac_N"] = + df["F_inertia_N"] + df["F_aero_N"] + df["F_roll_N"] + df["F_slope_N"] 
    return df
