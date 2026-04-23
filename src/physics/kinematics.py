from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter #new line
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def add_speed_ms(df: pd.DataFrame, *, speed_kmh_col: str = "speed_kmh", out_col: str = "speed_ms") -> pd.DataFrame:
    v_kmh = df[speed_kmh_col].astype(float).to_numpy()
    #df[out_col] = v_kmh * (5.0 / 18.0)
    v_ms_raw = v_kmh * (5.0 / 18.0)
    v_smooth = savgol_filter(v_ms_raw, window_length=51, polyorder=3)#new line
    
    
    df[out_col] = v_smooth
    
    return df

def add_acceleration(df: pd.DataFrame, *, t_col: str = "elapsed_time_s", v_col: str = "speed_ms",
                     out_col: str = "accel_ms2") -> pd.DataFrame:
    t = df[t_col].astype(float).to_numpy()
    v = df[v_col].astype(float).to_numpy()
    a_raw = np.gradient(v, t)
    #df[out_col] = np.gradient(v, t)
    a_smooth = savgol_filter(a_raw, window_length=21, polyorder=3)#new line
    df[out_col] = a_smooth #new line
    return df



def plot_velocity(
    df: pd.DataFrame,
    *,
    t_col: str = "elapsed_time_s",
    v_col: str = "speed_ms",
    n_points: int | None = None,
    start_idx: int = 0,
    time_in_minutes: bool = False,
    speed_in_kmh: bool = False,
):
    """
    Plot velocity vs time using Plotly.
    """

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

    # Velocity axis
    v = df_plot[v_col].astype(float).to_numpy()
    if speed_in_kmh:
        v = v * 3.6
        y_label = "Speed (km/h)"
    else:
        y_label = "Speed (m/s)"

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=t,
            y=v,
            mode="lines",
            name=v_col,
            line=dict(width=2)
        )
    )

    fig.update_layout(
        title="Velocity vs Time",
        xaxis_title=x_label,
        yaxis_title=y_label,
        hovermode="x unified",
        template="plotly_white",
        height=500,
        width=1000,
    )

    return fig


















'''
def add_speed_ms(df: pd.DataFrame, *, speed_kmh_col: str = "speed_kmh", out_col: str = "speed_ms") -> pd.DataFrame:
    v_kmh = df[speed_kmh_col].astype(float).to_numpy()
    df[out_col] = v_kmh * (5.0 / 18.0)
    return df

def add_acceleration(df: pd.DataFrame, *, t_col: str = "elapsed_time_s", v_col: str = "speed_ms",
                     out_col: str = "accel_ms2") -> pd.DataFrame:
    t = df[t_col].astype(float).to_numpy()
    v = df[v_col].astype(float).to_numpy()
    df[out_col] = np.gradient(v, t)
    return df
'''