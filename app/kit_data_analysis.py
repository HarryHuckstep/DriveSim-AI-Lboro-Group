"""
kit_data_analysis.py

Reusable fuel/distance calculations converted from Kit_data_extraction.ipynb.
Place this file in the same folder as dashboard_app_with_driver_classifier.py.

The main dashboard function is calculate_kit_outputs(df, ...), which accepts the
processed dashboard dataframe produced by prepare_dashboard_df(). It returns JSON-safe
fuel, distance and model-comparison outputs for Dash graphs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

try:
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVR
except Exception:  # pragma: no cover - lets dashboard still run without sklearn
    Pipeline = None
    StandardScaler = None
    SVR = None


@dataclass
class KitAnalysisParams:
    tank_volume_l: float = 50.0
    fuel_type: str = "petrol"
    afr_lambda: float = 1.0
    stop_speed_kmh: float = 1.0
    speed_mask_kmh: float = 5.0
    clip_negative_predictions: bool = True


def _first_existing_column(df: pd.DataFrame, candidates: list[str]) -> str:
    for col in candidates:
        if col in df.columns:
            return col
    raise KeyError(f"None of these columns were found in the dataframe: {candidates}")


def _optional_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for col in candidates:
        if col in df.columns:
            return col
    return None


def _to_numeric_array(series: pd.Series) -> np.ndarray:
    return pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)


def _replace_nan_for_json(values: np.ndarray) -> list[Any]:
    clean = []
    for value in values:
        if value is None or not np.isfinite(value):
            clean.append(None)
        else:
            clean.append(float(value))
    return clean


def cumulative_trapezoid(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    if len(x) < 2:
        return np.zeros_like(y, dtype=float)

    dx = np.diff(x)
    area = 0.5 * (y[1:] + y[:-1]) * dx
    return np.insert(np.cumsum(area), 0, 0.0)


def mass_air_flow_kgps(maf_gps: np.ndarray) -> np.ndarray:
    return np.asarray(maf_gps, dtype=float) / 1000.0


def fuel_mass_flow_kgps(maf_kgps: np.ndarray, afr_lambda: float = 1.0) -> np.ndarray:
    afr = 14.7 * afr_lambda
    return np.asarray(maf_kgps, dtype=float) / afr


def fuel_tank_flow_lps(
    fuel_mass_flow: np.ndarray,
    time_s: np.ndarray,
    tank_volume_l: float = 50.0,
    fuel_type: str = "petrol",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    fuel_type_normalised = fuel_type.strip().lower()

    if fuel_type_normalised == "petrol":
        density_kg_per_l = 0.745
    elif fuel_type_normalised == "diesel":
        density_kg_per_l = 0.832
    else:
        raise ValueError("fuel_type must be either 'petrol' or 'diesel'.")

    fuel_volume_flow_lps = np.asarray(fuel_mass_flow, dtype=float) / density_kg_per_l

    if len(time_s) < 2:
        fuel_used_l = np.zeros_like(fuel_volume_flow_lps, dtype=float)
    else:
        dt = np.gradient(time_s)
        fuel_used_l = np.cumsum(fuel_volume_flow_lps * dt)

    fuel_remaining_l = tank_volume_l - fuel_used_l
    return fuel_volume_flow_lps, fuel_used_l, fuel_remaining_l


def conventional_fuel_consumption_l_per_100km(
    speed_kmh: np.ndarray,
    maf_gps: np.ndarray,
    stop_speed_kmh: float = 1.0,
) -> np.ndarray:
    # Notebook method: MPG_us = 7.718 * speed / MAF, then convert to L/100 km.
    alpha = 7.718
    beta = 235.215
    eps = 1e-6

    speed_kmh = np.asarray(speed_kmh, dtype=float)
    maf_gps = np.asarray(maf_gps, dtype=float)

    safe_maf = np.maximum(maf_gps, eps)
    mpg_us = alpha * (speed_kmh / safe_maf)

    valid = (speed_kmh >= stop_speed_kmh) & np.isfinite(mpg_us) & (mpg_us > 0)
    fuel_consumption = np.full_like(mpg_us, np.nan, dtype=float)
    fuel_consumption[valid] = beta / mpg_us[valid]
    return fuel_consumption


def fit_fuel_consumption_models(
    fuel_consumption_l_per_100km: np.ndarray,
    rpm: np.ndarray,
    throttle_pct: np.ndarray,
    speed_kmh: np.ndarray,
    maf_gps: np.ndarray,
    speed_mask_kmh: float = 5.0,
    clip_negative: bool = True,
) -> dict[str, Any]:
    def safe_rmse(a: np.ndarray, b: np.ndarray, mask: np.ndarray) -> float:
        m = mask & np.isfinite(a) & np.isfinite(b)
        if not np.any(m):
            return float("nan")
        return float(np.sqrt(np.mean((a[m] - b[m]) ** 2)))

    n = min(
        len(fuel_consumption_l_per_100km),
        len(rpm),
        len(throttle_pct),
        len(speed_kmh),
        len(maf_gps),
    )

    fc = np.asarray(fuel_consumption_l_per_100km, dtype=float)[:n]
    rpm = np.asarray(rpm, dtype=float)[:n]
    throttle_pct = np.asarray(throttle_pct, dtype=float)[:n]
    speed_kmh = np.asarray(speed_kmh, dtype=float)[:n]
    maf_gps = np.asarray(maf_gps, dtype=float)[:n]

    valid = (
        np.isfinite(fc)
        & np.isfinite(rpm)
        & np.isfinite(throttle_pct)
        & (speed_kmh > speed_mask_kmh)
        & (maf_gps > 1e-3)
    )

    surface_prediction = np.full(n, np.nan, dtype=float)
    svr_prediction = np.full(n, np.nan, dtype=float)

    metrics = {
        "n_total": int(n),
        "n_valid": int(np.sum(valid)),
        "valid_samples_%": float(100.0 * np.sum(valid) / n) if n else float("nan"),
        "speed_mask": float(speed_mask_kmh),
        "clip_negative": bool(clip_negative),
        "surface_RMSE": float("nan"),
        "svr_RMSE": float("nan"),
        "median_fc_conv": float(np.nanmedian(fc[valid])) if np.any(valid) else float("nan"),
        "p95_fc_conv": float(np.nanpercentile(fc[valid], 95)) if np.any(valid) else float("nan"),
        "rmse_conv_vs_surface": float("nan"),
        "rmse_conv_vs_svr": float("nan"),
    }

    coefficients: dict[str, Any] = {
        "rpm_quadratic": None,
        "throttle_linear": None,
        "surface": None,
    }

    if np.sum(valid) >= 4:
        rpm_fit = rpm[valid]
        throttle_fit = throttle_pct[valid]
        fc_fit = fc[valid]

        a, b, c = np.polyfit(rpm_fit, fc_fit, 2)
        m, k = np.polyfit(throttle_fit, fc_fit, 1)

        design = np.column_stack([rpm_fit**2, rpm_fit, throttle_fit, np.ones_like(rpm_fit)])
        A, B, C, D = np.linalg.lstsq(design, fc_fit, rcond=None)[0]

        surface_prediction = A * (rpm**2) + B * rpm + C * throttle_pct + D
        surface_prediction = np.where(valid, surface_prediction, np.nan)

        if clip_negative:
            surface_prediction = np.where(surface_prediction >= 0, surface_prediction, np.nan)

        metrics["surface_RMSE"] = safe_rmse(fc, surface_prediction, valid)
        metrics["rmse_conv_vs_surface"] = metrics["surface_RMSE"]

        coefficients["rpm_quadratic"] = [float(a), float(b), float(c)]
        coefficients["throttle_linear"] = [float(m), float(k)]
        coefficients["surface"] = [float(A), float(B), float(C), float(D)]

        if Pipeline is not None and StandardScaler is not None and SVR is not None and np.sum(valid) >= 10:
            model = Pipeline(
                [
                    ("scaler", StandardScaler()),
                    ("svr", SVR(kernel="rbf", C=50.0, epsilon=0.05, gamma="scale")),
                ]
            )
            x_fit = np.column_stack([rpm_fit, throttle_fit])
            model.fit(x_fit, fc_fit)
            svr_prediction = model.predict(np.column_stack([rpm, throttle_pct]))
            svr_prediction = np.where(valid, svr_prediction, np.nan)

            if clip_negative:
                svr_prediction = np.where(svr_prediction >= 0, svr_prediction, np.nan)

            metrics["svr_RMSE"] = safe_rmse(fc, svr_prediction, valid)
            metrics["rmse_conv_vs_svr"] = metrics["svr_RMSE"]

    return {
        "fuelSurface_prediction": surface_prediction,
        "fuelPrediction_svr": svr_prediction,
        "validMask": valid,
        "Metrics": metrics,
        "coefficients": coefficients,
        "residual_surface": surface_prediction - fc,
        "residual_svr": svr_prediction - fc,
    }


def calculate_kit_outputs(df: pd.DataFrame, params: KitAnalysisParams | None = None) -> dict[str, Any]:
    """Calculate KIT notebook outputs from the processed dashboard dataframe."""
    params = params or KitAnalysisParams()

    time_col = _first_existing_column(df, ["time_s", "timestamp", "time", "Time"])
    speed_col = _first_existing_column(df, ["speed_kmh", "vehicle_speed", "Vehicle Speed Sensor [km/h]"])
    maf_col = _first_existing_column(df, ["maf_gps", "Air Flow Rate from Mass Flow Sensor [g/s]"])
    rpm_col = _optional_column(df, ["engine_rpm", "Engine RPM [RPM]"])
    throttle_col = _optional_column(df, ["throttle_pct", "Absolute Throttle Position [%]"])

    columns = [time_col, speed_col, maf_col]
    if rpm_col:
        columns.append(rpm_col)
    if throttle_col:
        columns.append(throttle_col)

    work = df[columns].copy().dropna(subset=[time_col, speed_col, maf_col])

    time_s = _to_numeric_array(work[time_col])
    speed_kmh = _to_numeric_array(work[speed_col])
    maf_gps = _to_numeric_array(work[maf_col])

    valid_base = np.isfinite(time_s) & np.isfinite(speed_kmh) & np.isfinite(maf_gps)
    time_s = time_s[valid_base]
    speed_kmh = speed_kmh[valid_base]
    maf_gps = maf_gps[valid_base]

    if len(time_s) == 0:
        raise ValueError("No valid time/speed/MAF data found for KIT fuel analysis.")

    if len(time_s) > 1 and np.nanmin(time_s) != 0:
        time_s = time_s - time_s[0]

    time_h = time_s / 3600.0
    distance_km = cumulative_trapezoid(time_h, speed_kmh)

    maf_kgps = mass_air_flow_kgps(maf_gps)
    fmf_kgps = fuel_mass_flow_kgps(maf_kgps, params.afr_lambda)
    fuel_lps, fuel_used_l, fuel_remaining_l = fuel_tank_flow_lps(
        fmf_kgps,
        time_s,
        params.tank_volume_l,
        params.fuel_type,
    )

    fuel_consumption = conventional_fuel_consumption_l_per_100km(
        speed_kmh,
        maf_gps,
        params.stop_speed_kmh,
    )

    rpm = None
    throttle_pct = None
    model_output = None

    if rpm_col and throttle_col:
        rpm = _to_numeric_array(work[rpm_col])[valid_base]
        throttle_pct = _to_numeric_array(work[throttle_col])[valid_base]
        model_output = fit_fuel_consumption_models(
            fuel_consumption,
            rpm,
            throttle_pct,
            speed_kmh,
            maf_gps,
            params.speed_mask_kmh,
            params.clip_negative_predictions,
        )
    else:
        model_output = {
            "fuelSurface_prediction": np.full_like(fuel_consumption, np.nan, dtype=float),
            "fuelPrediction_svr": np.full_like(fuel_consumption, np.nan, dtype=float),
            "validMask": np.isfinite(fuel_consumption),
            "Metrics": {
                "n_total": int(len(fuel_consumption)),
                "n_valid": 0,
                "valid_samples_%": 0.0,
                "surface_RMSE": float("nan"),
                "svr_RMSE": float("nan"),
            },
            "coefficients": {},
            "residual_surface": np.full_like(fuel_consumption, np.nan, dtype=float),
            "residual_svr": np.full_like(fuel_consumption, np.nan, dtype=float),
        }

    return {
        "time_series": {
            "time_s": _replace_nan_for_json(time_s),
            "distance_km": _replace_nan_for_json(distance_km),
            "speed_kmh": _replace_nan_for_json(speed_kmh),
            "maf_gps": _replace_nan_for_json(maf_gps),
            "fuel_flow_Lps": _replace_nan_for_json(fuel_lps),
            "fuel_used_L": _replace_nan_for_json(fuel_used_l),
            "fuel_remaining_L": _replace_nan_for_json(fuel_remaining_l),
            "fuel_consumption_L_per_100km": _replace_nan_for_json(fuel_consumption),
            "fuel_surface_prediction_L_per_100km": _replace_nan_for_json(model_output["fuelSurface_prediction"]),
            "fuel_svr_prediction_L_per_100km": _replace_nan_for_json(model_output["fuelPrediction_svr"]),
            "valid_model_mask": [bool(x) for x in model_output["validMask"]],
            "surface_residual_L_per_100km": _replace_nan_for_json(model_output["residual_surface"]),
            "svr_residual_L_per_100km": _replace_nan_for_json(model_output["residual_svr"]),
        },
        "metrics": model_output["Metrics"],
        "coefficients": model_output["coefficients"],
        "summary": {
            "distance_km": float(distance_km[-1]) if len(distance_km) else 0.0,
            "fuel_used_L": float(fuel_used_l[-1]) if len(fuel_used_l) else 0.0,
            "fuel_remaining_L": float(fuel_remaining_l[-1]) if len(fuel_remaining_l) else params.tank_volume_l,
            "median_fuel_consumption_L_per_100km": float(np.nanmedian(fuel_consumption)) if np.any(np.isfinite(fuel_consumption)) else float("nan"),
        },
    }
