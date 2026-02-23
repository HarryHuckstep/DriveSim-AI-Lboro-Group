from __future__ import annotations
import numpy as np
import pandas as pd

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
