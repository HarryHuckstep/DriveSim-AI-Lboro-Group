from __future__ import annotations
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def add_mdot_air_kgs(df):
    df["maf_kgps"] = (df["maf_gps"] / 1000)
    
    return df

def add_mdot_fuel(df, AFR = 14.7):
    df["mdot_fuel"] = df["maf_kgps"]  / AFR

    return df

def add_chemical_power(df, LHV = 43e6):
    df["Pfuel"]  = df["mdot_fuel"] * LHV 
    valid_fuel = df["Pfuel"] > 5000
    valid = (valid_fuel)
    
    return df.where(valid)


def add_chemical_efficiency(df):
    add_mdot_air_kgs(df)
    add_mdot_fuel(df)
    add_chemical_power(df)
    df["chemical_efficiency"] = df["P_drive_W"] / df["Pfuel"]

    return df





def plot_rolling_energy_efficiency(
    df: pd.DataFrame,
    *,
    t_col: str = "elapsed_time_s",
    eff_col: str = "eff_rolling_energy",
    n_points: int | None = None,
    start_idx: int = 0,
    time_in_minutes: bool = False,
    as_percent: bool = True,
):
    """
    Plot rolling energy efficiency vs time using Plotly.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing rolling efficiency column.
    t_col : str
        Time column name.
    eff_col : str
        Rolling efficiency column name.
    n_points : int | None
        Number of datapoints to plot. If None, plots all from start_idx onward.
    start_idx : int
        Index to start plotting from.
    time_in_minutes : bool
        If True, convert time axis from seconds to minutes.
    as_percent : bool
        If True, plot efficiency as percentage.
    """

    if eff_col not in df.columns:
        raise KeyError(f"Column '{eff_col}' not found in DataFrame.")

    if t_col not in df.columns:
        raise KeyError(f"Column '{t_col}' not found in DataFrame.")

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

    y = df_plot[eff_col].astype(float)

    if as_percent:
        y = y * 100
        y_label = "Rolling Energy Efficiency (%)"
    else:
        y_label = "Rolling Energy Efficiency"

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=t,
            y=y,
            mode="lines",
            name=eff_col,
            line=dict(width=2),
        )
    )

    # fig.update_layout(
    #     title="Rolling Energy Efficiency vs Time",
    #     xaxis_title=x_label,
    #     yaxis_title=y_label,
    #     hovermode="x unified",
    #     template="plotly_white",
    #     height=500,
    #     width=1000,
    # )

    fig.update_layout(
    xaxis_title=x_label,
    yaxis_title="Energy Efficiency %",
    height=500,)
    
    return fig