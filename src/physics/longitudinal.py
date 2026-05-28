from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

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




COLOR_MAP = {
    "aero": "blue",
    "roll": "red",
    "inertia": "green",
    "drive": "purple",
    "brake": "black",
    "trac": "orange",
    "slope": "brown",
    
}


def get_force_color(col_name: str):
    for key in COLOR_MAP:
        if key in col_name:
            return COLOR_MAP[key]
    return "gray"




def plot_longitudinal_forces(
    df: pd.DataFrame,
    *,
    t_col: str = "elapsed_time_s",
    n_points: int | None = None,
    start_idx: int = 0,
    time_in_minutes: bool = False,
    force_cols: list[str] | None = None,
):
    """
    Plot longitudinal forces vs time using Plotly.
    """

    if force_cols is None:
        force_cols = [
            "F_trac_N",
            "F_inertia_N",
            "F_aero_N",
            "F_roll_N",
            "F_slope_N",
        ]

    # Slice dataframe
    if n_points is None:
        df_plot = df.iloc[start_idx:].copy()
    else:
        df_plot = df.iloc[start_idx:start_idx + n_points].copy()

    # Time axis
    t = df_plot[t_col].astype(float).to_numpy()

    if time_in_minutes:
        t = t / 60
        x_label = "Time (min)"
    else:
        x_label = "Time (s)"

    fig = go.Figure()

#    for col in force_cols:
#        if col in df_plot.columns:
#            fig.add_trace(
#                go.Scatter(
#                    x=t,
#                    y=df_plot[col],
#                    mode="lines",
#                    name=col,
#                )
#            )
            
    for col in force_cols:
        if col in df_plot.columns:
            fig.add_trace(
                go.Scatter(
                    x=t,
                    y=df_plot[col],
                    mode="lines",
                    name=col,
                    line=dict(color=get_force_color(col), width=2)
                )
            )
            
            
    fig.update_layout(
        title="Longitudinal Forces vs Time",
        xaxis_title=x_label,
        yaxis_title="Force (N)",
        hovermode="x unified",
        template="plotly_white",
        height=500,
        width=1000,
    )

    return fig















































'''
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
'''