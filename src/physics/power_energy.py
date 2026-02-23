from __future__ import annotations
import numpy as np
import pandas as pd

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
