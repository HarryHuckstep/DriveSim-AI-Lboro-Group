from __future__ import annotations
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def cumulative_trapz(power_W: np.ndarray, t_s: np.ndarray) -> np.ndarray:
    p = np.nan_to_num(power_W.astype(float), nan=0.0)
    t = t_s.astype(float)
    E = np.zeros_like(p)
    dt = np.diff(t)
    inc = 0.5 * (p[1:] + p[:-1]) * dt
    E[1:] = np.cumsum(inc)
    return E

def add_power_terms(df: pd.DataFrame, *, v_col: str = "speed_ms") -> pd.DataFrame:
    
    v = df[v_col].astype(float).to_numpy()

    # component powers (matching your notebook)
    df["P_aero_W"] = df["F_aero_N"].astype(float).to_numpy() * v
    df["P_roll_W"] = df["F_roll_N"].astype(float).to_numpy() * v
    df["P_inertia_W"] = df["F_inertia_N"].astype(float).to_numpy() * v

    df["P_wheel_W"] = df["F_trac_N"].astype(float).to_numpy() * v
    df["P_drive_W"] = np.clip(df["P_wheel_W"].to_numpy(), 0, None)
    df["P_brake_W"] = np.clip(df["P_wheel_W"].to_numpy(), None, 0)

    return df


def add_energy_terms(df: pd.DataFrame, *, t_col: str = "elapsed_time_s") -> pd.DataFrame:
    t = df[t_col].astype(float).to_numpy()

    df["E_aero_J"] = cumulative_trapz(df["P_aero_W"].to_numpy(), t)
    df["E_roll_J"] = cumulative_trapz(df["P_roll_W"].to_numpy(), t)
    df["E_inertia_J"] = cumulative_trapz(df["P_inertia_W"].to_numpy(), t)
    df["E_drive_J"] = cumulative_trapz(df["P_drive_W"].to_numpy(), t)
    df["E_brake_J"] = cumulative_trapz((-df["P_brake_W"]).to_numpy(), t)

    # convenience columns
    for name in ["E_aero", "E_roll", "E_inertia", "E_drive", "E_brake"]:
        df[f"{name}_kJ"] = df[f"{name}_J"] / 1e3

    return df



def get_force_color(col_name: str):
    for key in COLOR_MAP:
        if key in col_name:
            return COLOR_MAP[key]
    return "gray"



COLOR_MAP = {
    "aero": "blue",
    "roll": "red",
    "inertia": "green",
    "drive": "purple",
    "brake": "black",
    "trac": "orange",
    "slope": "brown",
}





def plot_power(
    df: pd.DataFrame,
    *,
    t_col: str = "elapsed_time_s",
    n_points: int | None = None,
    start_idx: int = 0,
    time_in_minutes: bool = False,
    power_cols: list[str] | None = None,
):
    """
    Plot power terms vs time using Plotly.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing power columns.
    t_col : str
        Time column name.
    n_points : int | None
        Number of datapoints to plot. If None, plots all from start_idx onward.
    start_idx : int
        Index to start plotting from.
    time_in_minutes : bool
        If True, convert time axis from seconds to minutes.
    power_cols : list[str] | None
        Power columns to plot.
    """

    if power_cols is None:
        power_cols = [
            "P_aero_W",
            "P_roll_W",
            "P_inertia_W",
            "P_drive_W",
            "P_brake_W",
        ]

    if n_points is None:
        df_plot = df.iloc[start_idx:].copy()
    else:
        df_plot = df.iloc[start_idx:start_idx + n_points].copy()

    t = df_plot[t_col].astype(float).to_numpy()
    if time_in_minutes:
        t = t / 60
        x_label = "Time (min)"
    else:
        x_label = "Time (s)"

    fig = go.Figure()

#    for col in power_cols:
#        if col in df_plot.columns:
#            fig.add_trace(
#                go.Scatter(
#                    x=t,
#                    y=df_plot[col],
#                    mode="lines",
#                    name=col,
#                )
#            )

    for col in power_cols:
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
        title="Power vs Time",
        xaxis_title=x_label,
        yaxis_title="Power (W)",
        hovermode="x unified",
        template="plotly_white",
        height=500,
        width=1000,
    )

    return fig



def plot_cumulative_energy(
    df: pd.DataFrame,
    *,
    t_col: str = "elapsed_time_s",
    n_points: int | None = None,
    start_idx: int = 0,
    time_in_minutes: bool = False,
    energy_cols: list[str] | None = None,
):
    """
    Plot cumulative energy terms vs time using Plotly.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing cumulative energy columns.
    t_col : str
        Time column name.
    n_points : int | None
        Number of datapoints to plot. If None, plots all from start_idx onward.
    start_idx : int
        Index to start plotting from.
    time_in_minutes : bool
        If True, convert time axis from seconds to minutes.
    energy_cols : list[str] | None
        Energy columns to plot.
    """

    if energy_cols is None:
        energy_cols = [
            "E_aero_kJ",
            "E_roll_kJ",
            "E_inertia_kJ",
            "E_drive_kJ",
            "E_brake_kJ",
        ]

    if n_points is None:
        df_plot = df.iloc[start_idx:].copy()
    else:
        df_plot = df.iloc[start_idx:start_idx + n_points].copy()

    t = df_plot[t_col].astype(float).to_numpy()
    if time_in_minutes:
        t = t / 60
        x_label = "Time (min)"
    else:
        x_label = "Time (s)"

    fig = go.Figure()

#    for col in energy_cols:
#        if col in df_plot.columns:
#            fig.add_trace(
#                go.Scatter(
#                    x=t,
#                    y=df_plot[col],
#                    mode="lines",
#                    name=col,
#                )
#            )
            
    for col in energy_cols:
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
        title="Cumulative Energy vs Time",
        xaxis_title=x_label,
        yaxis_title="Cumulative Energy (J)",
        hovermode="x unified",
        template="plotly_white",
        height=500,
        width=1000,
    )

    return fig

























'''
def cumulative_trapz(power_W: np.ndarray, t_s: np.ndarray) -> np.ndarray:
    p = np.nan_to_num(power_W.astype(float), nan=0.0)
    t = t_s.astype(float)
    E = np.zeros_like(p)
    dt = np.diff(t)
    inc = 0.5 * (p[1:] + p[:-1]) * dt
    E[1:] = np.cumsum(inc)
    return E

def add_power_terms(df: pd.DataFrame, *, v_col: str = "speed_ms") -> pd.DataFrame:
    v = df[v_col].astype(float).to_numpy()

    # component powers (matching your notebook)
    df["P_aero_W"] = df["F_aero_N"].astype(float).to_numpy() * v
    df["P_roll_W"] = df["F_roll_N"].astype(float).to_numpy() * v
    df["P_inertia_W"] = df["F_inertia_N"].astype(float).to_numpy() * v

    df["P_wheel_W"] = df["F_trac_N"].astype(float).to_numpy() * v
    df["P_drive_W"] = np.clip(df["P_wheel_W"].to_numpy(), 0, None)
    df["P_brake_W"] = np.clip(df["P_wheel_W"].to_numpy(), None, 0)

    return df

def add_energy_terms(df: pd.DataFrame, *, t_col: str = "elapsed_time_s") -> pd.DataFrame:
    t = df[t_col].astype(float).to_numpy()

    df["E_aero_J"] = cumulative_trapz(df["P_aero_W"].to_numpy(), t)
    df["E_roll_J"] = cumulative_trapz(df["P_roll_W"].to_numpy(), t)
    df["E_inertia_J"] = cumulative_trapz(df["P_inertia_W"].to_numpy(), t)
    df["E_drive_J"] = cumulative_trapz(df["P_drive_W"].to_numpy(), t)
    df["E_brake_J"] = cumulative_trapz((-df["P_brake_W"]).to_numpy(), t)

    # convenience columns
    for name in ["E_aero", "E_roll", "E_inertia", "E_drive", "E_brake"]:
        df[f"{name}_kJ"] = df[f"{name}_J"] / 1e3

    return df

'''
