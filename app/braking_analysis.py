"""
braking_analysis.py

Reusable braking calculations converted from Braking_Force.ipynb.
Place this file in the same folder as dashboard_app_with_driver_classifier.py.

The main dashboard function is calculate_braking_analysis(df, ...), which accepts the
processed dashboard dataframe produced by prepare_dashboard_df(). It returns two
JSON-safe dictionaries:
    - "time_series": values to graph in Dash
    - "events": braking-event summary table data
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class BrakingParams:
    vehicle_mass_kg: float = 1241.0
    wheel_radius_m: float = 0.30
    front_bias: float = 0.65
    rho_air: float = 1.225
    Cd: float = 0.30
    frontal_area_m2: float = 2.2
    Crr: float = 0.012
    g: float = 9.81
    smooth_window: int = 9
    braking_threshold_mps2: float = -0.3
    merge_gap_samples: int = 10


def _first_existing_column(df: pd.DataFrame, candidates: list[str]) -> str:
    for col in candidates:
        if col in df.columns:
            return col
    raise KeyError(f"None of these columns were found in the dataframe: {candidates}")


def _to_numeric_array(series: pd.Series) -> np.ndarray:
    return pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)


def moving_average(values: np.ndarray, window: int = 9) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    window = int(window)

    if window < 2 or len(values) == 0:
        return values

    if window % 2 == 0:
        window += 1

    weights = np.ones(window) / window
    return np.convolve(values, weights, mode="same")


def compute_acceleration(time_s: np.ndarray, speed_kmh: np.ndarray, smooth_window: int = 9) -> tuple[np.ndarray, np.ndarray]:
    speed_mps = np.asarray(speed_kmh, dtype=float) / 3.6
    speed_mps_smooth = moving_average(speed_mps, window=smooth_window)

    if len(time_s) < 2:
        acceleration = np.zeros_like(speed_mps_smooth)
    else:
        acceleration = np.gradient(speed_mps_smooth, time_s)

    return speed_mps_smooth, acceleration


def compute_brake_force(
    vehicle_mass_kg: float,
    acceleration_mps2: np.ndarray,
    speed_mps: np.ndarray,
    rho_air: float = 1.225,
    Cd: float = 0.30,
    frontal_area_m2: float = 2.2,
    Crr: float = 0.012,
    g: float = 9.81,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    drag_force_n = 0.5 * rho_air * Cd * frontal_area_m2 * (speed_mps ** 2)
    rolling_force_n = np.full_like(speed_mps, Crr * vehicle_mass_kg * g, dtype=float)

    brake_force_n = -(vehicle_mass_kg * acceleration_mps2) - drag_force_n - rolling_force_n
    brake_force_n = np.maximum(brake_force_n, 0.0)

    return brake_force_n, drag_force_n, rolling_force_n


def detect_braking(acceleration_mps2: np.ndarray, threshold_mps2: float = -0.3) -> np.ndarray:
    return np.asarray(acceleration_mps2, dtype=float) < threshold_mps2


def compute_brake_torque(brake_force_n: np.ndarray, wheel_radius_m: float) -> np.ndarray:
    return np.asarray(brake_force_n, dtype=float) * wheel_radius_m


def split_brake_torque(brake_torque_nm: np.ndarray, front_bias: float = 0.65) -> tuple[np.ndarray, np.ndarray]:
    front_torque_nm = front_bias * brake_torque_nm
    rear_torque_nm = (1.0 - front_bias) * brake_torque_nm
    return front_torque_nm, rear_torque_nm


def compute_brake_power(brake_force_n: np.ndarray, speed_mps: np.ndarray) -> np.ndarray:
    return np.asarray(brake_force_n, dtype=float) * np.asarray(speed_mps, dtype=float)


def compute_brake_energy(brake_power_w: np.ndarray, time_s: np.ndarray) -> np.ndarray:
    if len(time_s) < 2:
        return np.zeros_like(brake_power_w, dtype=float)

    dt = np.gradient(time_s)
    return np.cumsum(np.asarray(brake_power_w, dtype=float) * dt)


def find_braking_events(braking_mask: np.ndarray) -> list[tuple[int, int]]:
    braking_mask = np.asarray(braking_mask, dtype=bool)

    if len(braking_mask) == 0:
        return []

    starts = np.where((~braking_mask[:-1]) & (braking_mask[1:]))[0] + 1
    ends = np.where((braking_mask[:-1]) & (~braking_mask[1:]))[0] + 1

    if braking_mask[0]:
        starts = np.r_[0, starts]
    if braking_mask[-1]:
        ends = np.r_[ends, len(braking_mask) - 1]

    return list(zip(starts, ends))


def merge_close_events(events: list[tuple[int, int]], max_gap_samples: int = 10) -> list[tuple[int, int]]:
    if not events:
        return []

    merged = [events[0]]

    for start, end in events[1:]:
        last_start, last_end = merged[-1]
        if start - last_end <= max_gap_samples:
            merged[-1] = (last_start, end)
        else:
            merged.append((start, end))

    return merged


def summarise_braking_events(
    events: list[tuple[int, int]],
    time_s: np.ndarray,
    speed_mps: np.ndarray,
    brake_force_n: np.ndarray,
    brake_torque_nm: np.ndarray,
    brake_energy_j: np.ndarray,
) -> list[dict[str, float]]:
    summaries = []

    for start, end in events:
        event_energy_j = brake_energy_j[end] - brake_energy_j[start]
        speed_drop_mps = speed_mps[start] - speed_mps[end]
        delta_ke_j = 0.5 * (speed_mps[start] ** 2 - speed_mps[end] ** 2)

        summaries.append(
            {
                "start_idx": int(start),
                "end_idx": int(end),
                "start_time_s": float(time_s[start]),
                "end_time_s": float(time_s[end]),
                "duration_s": float(time_s[end] - time_s[start]),
                "start_speed_kmh": float(speed_mps[start] * 3.6),
                "end_speed_kmh": float(speed_mps[end] * 3.6),
                "speed_drop_kmh": float(speed_drop_mps * 3.6),
                "peak_brake_force_N": float(np.nanmax(brake_force_n[start : end + 1])),
                "peak_brake_torque_Nm": float(np.nanmax(brake_torque_nm[start : end + 1])),
                "brake_energy_J": float(event_energy_j),
                "specific_delta_ke_J_per_kg": float(delta_ke_j),
            }
        )

    return summaries


def _replace_nan_for_json(values: np.ndarray) -> list[Any]:
    clean = []
    for value in values:
        if value is None or not np.isfinite(value):
            clean.append(None)
        else:
            clean.append(float(value))
    return clean


def calculate_braking_analysis(df: pd.DataFrame, params: BrakingParams | None = None) -> dict[str, Any]:
    """Calculate braking signals from the processed dashboard dataframe."""
    params = params or BrakingParams()

    time_col = _first_existing_column(df, ["time_s", "timestamp", "time", "Time"])
    speed_col = _first_existing_column(df, ["speed_kmh", "vehicle_speed", "Vehicle Speed Sensor [km/h]"])

    work = df[[time_col, speed_col]].copy().dropna()
    time_s = _to_numeric_array(work[time_col])
    speed_kmh = _to_numeric_array(work[speed_col])

    valid = np.isfinite(time_s) & np.isfinite(speed_kmh)
    time_s = time_s[valid]
    speed_kmh = speed_kmh[valid]

    if len(time_s) == 0:
        raise ValueError("No valid time/speed data found for braking analysis.")

    if len(time_s) > 1 and np.nanmin(time_s) != 0:
        time_s = time_s - time_s[0]

    speed_mps, acceleration_mps2 = compute_acceleration(time_s, speed_kmh, params.smooth_window)
    brake_force_n, drag_force_n, rolling_force_n = compute_brake_force(
        params.vehicle_mass_kg,
        acceleration_mps2,
        speed_mps,
        params.rho_air,
        params.Cd,
        params.frontal_area_m2,
        params.Crr,
        params.g,
    )

    braking_mask = detect_braking(acceleration_mps2, params.braking_threshold_mps2)
    brake_force_masked_n = np.where(braking_mask, brake_force_n, 0.0)

    brake_torque_nm = compute_brake_torque(brake_force_masked_n, params.wheel_radius_m)
    front_torque_nm, rear_torque_nm = split_brake_torque(brake_torque_nm, params.front_bias)
    brake_power_w = compute_brake_power(brake_force_masked_n, speed_mps)
    brake_energy_j = compute_brake_energy(brake_power_w, time_s)

    events = merge_close_events(find_braking_events(braking_mask), params.merge_gap_samples)
    event_summaries = summarise_braking_events(
        events,
        time_s,
        speed_mps,
        brake_force_masked_n,
        brake_torque_nm,
        brake_energy_j,
    )

    return {
        "time_series": {
            "time_s": _replace_nan_for_json(time_s),
            "speed_kmh": _replace_nan_for_json(speed_kmh),
            "speed_mps": _replace_nan_for_json(speed_mps),
            "acceleration_mps2": _replace_nan_for_json(acceleration_mps2),
            "braking_mask": [bool(x) for x in braking_mask],
            "brake_force_N": _replace_nan_for_json(brake_force_masked_n),
            "drag_force_N": _replace_nan_for_json(drag_force_n),
            "rolling_force_N": _replace_nan_for_json(rolling_force_n),
            "brake_torque_Nm": _replace_nan_for_json(brake_torque_nm),
            "front_brake_torque_Nm": _replace_nan_for_json(front_torque_nm),
            "rear_brake_torque_Nm": _replace_nan_for_json(rear_torque_nm),
            "brake_power_W": _replace_nan_for_json(brake_power_w),
            "brake_energy_J": _replace_nan_for_json(brake_energy_j),
        },
        "events": event_summaries,
        "summary": {
            "braking_event_count": int(len(event_summaries)),
            "peak_brake_force_N": float(np.nanmax(brake_force_masked_n)) if len(brake_force_masked_n) else 0.0,
            "peak_brake_torque_Nm": float(np.nanmax(brake_torque_nm)) if len(brake_torque_nm) else 0.0,
            "total_brake_energy_J": float(brake_energy_j[-1]) if len(brake_energy_j) else 0.0,
            "max_braking_mps2": float(np.nanmin(acceleration_mps2)) if len(acceleration_mps2) else 0.0,
        },
    }
