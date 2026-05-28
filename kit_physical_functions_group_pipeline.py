"""
Fuel-consumption / vehicle-physics calculations adapted for the group OBD-II pipeline.

Expected input:
    A CSV already processed by the group workflow, usually:
        raw file -> dataHandler.py -> *_clean.csv -> dataSmoother.py -> *_clean_smoothed.csv

This version keeps the physical calculations from the original notebook, but changes the
way data is ingested. It now reads the group-standard column names:
    timestamp, vehicle_speed, maf_gps, engine_rpm, throttle_pct

Example:
    python kit_physical_functions_group_pipeline.py Leon1_clean_smoothed.csv

Optional output:
    python kit_physical_functions_group_pipeline.py Leon1_clean_smoothed.csv --output Leon1_fuel_results.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVR
    SKLEARN_AVAILABLE = True
except ImportError:  # Allows the physical calculations to run even if sklearn is unavailable.
    SKLEARN_AVAILABLE = False


# -----------------------------------------------------------------------------
# Group-pipeline data ingestion
# -----------------------------------------------------------------------------

GROUP_COLUMNS = {
    "time": "timestamp",
    "speed": "vehicle_speed",
    "maf": "maf_gps",
    "rpm": "engine_rpm",
    "throttle": "throttle_pct",
}

TIME_COLUMNS = ["timestamp", "timestep", "time step", "time_step", "t_s"]


def load_group_obd_csv(file_path: str | Path) -> pd.DataFrame:
    """Load a cleaned/smoothed group CSV and coerce expected columns to numeric."""
    file_path = Path(file_path)
    df = pd.read_csv(file_path)

    # Make column names robust to accidental whitespace.
    df.columns = [str(c).strip() for c in df.columns]

    required = [
        GROUP_COLUMNS["time"],
        GROUP_COLUMNS["speed"],
        GROUP_COLUMNS["maf"],
        GROUP_COLUMNS["rpm"],
        GROUP_COLUMNS["throttle"],
    ]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(
            "Input file does not look like a group cleaned/smoothed CSV. "
            f"Missing columns: {missing}. Expected at least: {required}"
        )

    for col in df.columns:
        converted = pd.to_numeric(df[col], errors="coerce")
        if converted.notna().sum() > 0:
            df[col] = converted

    return df


def get_series(df: pd.DataFrame, column: str) -> np.ndarray:
    """Return a numeric numpy array for a column from a group-format dataframe."""
    if column not in df.columns:
        raise KeyError(f"Column {column!r} not found in dataframe.")
    return pd.to_numeric(df[column], errors="coerce").to_numpy(dtype=float)


def get_time_seconds(df: pd.DataFrame) -> np.ndarray:
    """
    Read the group time axis. dataSmoother.py should already convert timestamp to
    seconds from the start, so this mostly just returns df['timestamp'].
    """
    for col in TIME_COLUMNS:
        if col in df.columns:
            time = get_series(df, col)
            return time - time[0] if np.isfinite(time[0]) else time

    # Fallback: use sample index if no time column exists.
    return np.arange(len(df), dtype=float)


# -----------------------------------------------------------------------------
# Plot helper, kept simple and compatible with group dataframes
# -----------------------------------------------------------------------------

def plot_x_against_y(x: np.ndarray, y: np.ndarray, x_label: str, y_label: str) -> None:
    plt.figure(figsize=(10, 6))
    plt.plot(x, y, marker="o", linestyle="-")
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(f"{y_label} against {x_label}")
    plt.grid(True)
    plt.show()


# -----------------------------------------------------------------------------
# Physical functions from the notebook, now dataframe-based rather than file/path-based
# -----------------------------------------------------------------------------

def compute_acceleration(time_s: np.ndarray, speed_kmh: np.ndarray) -> np.ndarray:
    """
    Compute acceleration from vehicle speed.

    Note: because speed is in km/h and time is in seconds, the raw unit is
    (km/h)/s. If SI acceleration is required, multiply speed by 1000/3600 first.
    """
    return np.gradient(speed_kmh, time_s)


def trapezoidal_rule(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Cumulative trapezoidal integration, returning an array aligned with x/y."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    valid = np.isfinite(x) & np.isfinite(y)
    x_valid = x[valid]
    y_valid = y[valid]

    if len(x_valid) < 2:
        return np.full_like(x, np.nan, dtype=float)

    dx = np.diff(x_valid)
    increments = 0.5 * (y_valid[1:] + y_valid[:-1]) * dx
    cumulative = np.insert(np.cumsum(increments), 0, 0.0)

    result = np.full_like(x, np.nan, dtype=float)
    result[np.where(valid)[0]] = cumulative
    return result


def distance_from_speed(time_s: np.ndarray, speed_kmh: np.ndarray) -> np.ndarray:
    """Integrate speed over time to estimate distance in km."""
    time_h = np.asarray(time_s, dtype=float) / 3600.0
    return trapezoidal_rule(time_h, speed_kmh)


def mass_air_flow_kg_s(maf_gps: np.ndarray) -> np.ndarray:
    """Convert MAF from grams/second to kilograms/second."""
    return np.asarray(maf_gps, dtype=float) / 1000.0


def fuel_mass_flow(maf_kg_s: np.ndarray, afr_lambda: float = 1.0) -> np.ndarray:
    """Estimate fuel mass flow using stoichiometric petrol AFR scaled by lambda."""
    afr = 14.7 * afr_lambda
    return np.asarray(maf_kg_s, dtype=float) / afr


def car_fuel_tank_flow(
    fuel_mass_flow_kg_s: np.ndarray,
    fuel_tank_volume_l: float,
    fuel_type: str,
    time_s: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Estimate fuel volume flow and remaining tank volume.

    Densities are kg/L:
        petrol: 0.745
        diesel: 0.832
    """
    fuel_type_norm = fuel_type.strip().lower()
    if fuel_type_norm == "petrol":
        density_kg_l = 0.745
    elif fuel_type_norm == "diesel":
        density_kg_l = 0.832
    else:
        raise ValueError(
            "Fuel type is not supported. Allowed values are: 'petrol' and 'diesel'. "
            f"Received: {fuel_type}"
        )

    fuel_volume_flow_l_s = np.asarray(fuel_mass_flow_kg_s, dtype=float) / density_kg_l
    dt = np.gradient(np.asarray(time_s, dtype=float))
    fuel_used_l = np.cumsum(fuel_volume_flow_l_s * dt)
    fuel_remaining_l = fuel_tank_volume_l - fuel_used_l
    return fuel_volume_flow_l_s, fuel_remaining_l


def fuel_consumption_l_per_100km_conventional(
    speed_kmh: np.ndarray,
    maf_gps: np.ndarray,
    stop_speed_kmh: float = 1.0,
) -> np.ndarray:
    """
    Conventional fuel consumption estimate using speed and MAF.

    This keeps the notebook's constants:
        MPG_us = 7.718 * speed / MAF
        L/100km = 235.215 / MPG_us
    """
    speed = np.asarray(speed_kmh, dtype=float)
    maf = np.asarray(maf_gps, dtype=float)

    alpha = 7.718
    beta = 235.215
    eps = 1e-6

    safe_maf = np.maximum(maf, eps)
    mpg_us = alpha * (speed / safe_maf)

    valid = (speed >= stop_speed_kmh) & np.isfinite(mpg_us) & (mpg_us > 0)
    fuel_consumption = np.full_like(mpg_us, np.nan, dtype=float)
    fuel_consumption[valid] = beta / mpg_us[valid]

    return fuel_consumption


# -----------------------------------------------------------------------------
# Curve fitting / ML improvement from the notebook
# -----------------------------------------------------------------------------

def curve_fitting_improvement_fuel_consumption(
    fuel_consumption: np.ndarray,
    rpm: np.ndarray,
    throttle_pct: np.ndarray,
    speed_kmh: np.ndarray,
    maf_gps: np.ndarray,
    speed_mask: float,
    clip_negative: bool = True,
) -> Dict[str, object]:
    """Fit RPM/TPS surface model and optional SVR model to conventional FC."""

    def safe_rmse(a: np.ndarray, b: np.ndarray, mask: np.ndarray) -> float:
        a = np.asarray(a, dtype=float)
        b = np.asarray(b, dtype=float)
        m = np.asarray(mask, dtype=bool) & np.isfinite(a) & np.isfinite(b)
        if not np.any(m):
            return np.nan
        return float(np.sqrt(np.mean((a[m] - b[m]) ** 2)))

    rpm = np.asarray(rpm, dtype=float)
    throttle_pct = np.asarray(throttle_pct, dtype=float)
    fuel_consumption = np.asarray(fuel_consumption, dtype=float)
    speed_kmh = np.asarray(speed_kmh, dtype=float)
    maf_gps = np.asarray(maf_gps, dtype=float)

    n = min(len(rpm), len(throttle_pct), len(fuel_consumption), len(speed_kmh), len(maf_gps))
    rpm = rpm[:n]
    throttle_pct = throttle_pct[:n]
    fuel_consumption = fuel_consumption[:n]
    speed_kmh = speed_kmh[:n]
    maf_gps = maf_gps[:n]

    valid = (
        np.isfinite(fuel_consumption)
        & np.isfinite(rpm)
        & np.isfinite(throttle_pct)
        & (speed_kmh > speed_mask)
        & (maf_gps > 1e-3)
    )

    if np.sum(valid) < 4:
        raise ValueError("Not enough valid samples to fit the fuel-consumption models.")

    rpm_fit = rpm[valid]
    throttle_fit = throttle_pct[valid]
    fc_fit = fuel_consumption[valid]

    # Equation 2: quadratic fit to RPM.
    a, b, c = np.polyfit(rpm_fit, fc_fit, 2)

    # Equation 3: linear fit to throttle position.
    m, k = np.polyfit(throttle_fit, fc_fit, 1)

    # Equation 4: combined surface fit.
    phi = np.column_stack([rpm_fit**2, rpm_fit, throttle_fit, np.ones_like(rpm_fit)])
    A, B, C, D = np.linalg.lstsq(phi, fc_fit, rcond=None)[0]

    # Equation 5: surface prediction.
    fuel_surface_prediction = A * (rpm**2) + B * rpm + C * throttle_pct + D
    fuel_surface_prediction = np.where(valid, fuel_surface_prediction, np.nan)
    if clip_negative:
        fuel_surface_prediction = np.where(fuel_surface_prediction >= 0, fuel_surface_prediction, np.nan)

    surface_rmse = safe_rmse(fuel_consumption, fuel_surface_prediction, valid)

    fuel_prediction_svr = np.full_like(fuel_consumption, np.nan, dtype=float)
    svr_model = None
    svr_rmse = np.nan

    if SKLEARN_AVAILABLE:
        svr_model = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("svr", SVR(kernel="rbf", C=50.0, epsilon=0.05, gamma="scale")),
            ]
        )
        x_fit = np.column_stack([rpm_fit, throttle_fit])
        svr_model.fit(x_fit, fc_fit)

        x_all = np.column_stack([rpm, throttle_pct])
        fuel_prediction_svr = svr_model.predict(x_all)
        fuel_prediction_svr = np.where(valid, fuel_prediction_svr, np.nan)
        if clip_negative:
            fuel_prediction_svr = np.where(fuel_prediction_svr >= 0, fuel_prediction_svr, np.nan)
        svr_rmse = safe_rmse(fuel_consumption, fuel_prediction_svr, valid)

    fc_valid = fuel_consumption[valid]
    metrics = {
        "surface_RMSE": surface_rmse,
        "svr_RMSE": svr_rmse,
        "valid_samples_%": float(100.0 * np.sum(valid) / n) if n else np.nan,
        "median_fc_conv": float(np.nanmedian(fc_valid)) if fc_valid.size else np.nan,
        "p95_fc_conv": float(np.nanpercentile(fc_valid, 95)) if fc_valid.size else np.nan,
        "rmse_conv_vs_surface": safe_rmse(fuel_consumption, fuel_surface_prediction, valid),
        "rmse_conv_vs_svr": safe_rmse(fuel_consumption, fuel_prediction_svr, valid),
        "n_total": int(n),
        "n_valid": int(np.sum(valid)),
        "speed_mask": float(speed_mask),
        "clip_negative": bool(clip_negative),
    }

    return {
        "fuelSurface_prediction": fuel_surface_prediction,
        "fuelPrediction_svr": fuel_prediction_svr,
        "RPM quadratic coefficients": (a, b, c),
        "TPS linear coefficients": (m, k),
        "surface coefficients": (A, B, C, D),
        "svr model": svr_model,
        "validMask": valid,
        "Metrics": metrics,
        "RMSE_conv_vs_SVR": metrics["rmse_conv_vs_svr"],
        "RMSE_conv_vs_surface": metrics["rmse_conv_vs_surface"],
    }


# -----------------------------------------------------------------------------
# Result table builder for easy integration with plotHandler.py
# -----------------------------------------------------------------------------

def add_physical_outputs(
    df: pd.DataFrame,
    fuel_tank_volume_l: float = 50.0,
    fuel_type: str = "petrol",
    afr_lambda: float = 1.0,
    stop_speed_kmh: float = 1.0,
    fit_speed_mask_kmh: float = 5.0,
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    """Add your calculated columns to the group dataframe without renaming existing columns."""
    out = df.copy()

    time_s = get_time_seconds(out)
    speed_kmh = get_series(out, GROUP_COLUMNS["speed"])
    maf_gps = get_series(out, GROUP_COLUMNS["maf"])
    rpm = get_series(out, GROUP_COLUMNS["rpm"])
    throttle = get_series(out, GROUP_COLUMNS["throttle"])

    acceleration = compute_acceleration(time_s, speed_kmh)
    distance_km = distance_from_speed(time_s, speed_kmh)
    maf_kg_s = mass_air_flow_kg_s(maf_gps)
    fmf_kg_s = fuel_mass_flow(maf_kg_s, afr_lambda=afr_lambda)
    fuel_l_s, fuel_remaining_l = car_fuel_tank_flow(
        fmf_kg_s,
        fuel_tank_volume_l=fuel_tank_volume_l,
        fuel_type=fuel_type,
        time_s=time_s,
    )
    fuel_consumption = fuel_consumption_l_per_100km_conventional(
        speed_kmh,
        maf_gps,
        stop_speed_kmh=stop_speed_kmh,
    )

    fit_output = curve_fitting_improvement_fuel_consumption(
        fuel_consumption=fuel_consumption,
        rpm=rpm,
        throttle_pct=throttle,
        speed_kmh=speed_kmh,
        maf_gps=maf_gps,
        speed_mask=fit_speed_mask_kmh,
    )

    # New columns are added; original group columns are preserved.
    out["acceleration_kmh_per_s"] = acceleration
    out["distance_km"] = distance_km
    out["maf_kg_s"] = maf_kg_s
    out["fuel_mass_flow_kg_s"] = fmf_kg_s
    out["fuel_volume_flow_l_s"] = fuel_l_s
    out["fuel_remaining_l"] = fuel_remaining_l
    out["fuel_consumption_l_per_100km"] = fuel_consumption
    out["fuel_surface_prediction_l_per_100km"] = fit_output["fuelSurface_prediction"]
    out["fuel_svr_prediction_l_per_100km"] = fit_output["fuelPrediction_svr"]
    out["fuel_model_valid"] = fit_output["validMask"]

    return out, fit_output


class FuelConsumptionPlots:
    """Plot class kept from the notebook, now using arrays from the result dataframe."""

    def __init__(
        self,
        time_s: np.ndarray,
        speed_kmh: np.ndarray,
        fuel_consumption: np.ndarray,
        output_curve_fitting: Dict[str, object],
        max_scatter_points: int = 20_000,
    ):
        self.time_s = time_s
        self.speed_kmh = speed_kmh
        self.fuel_consumption = fuel_consumption
        self.output_curve_fitting = output_curve_fitting
        self.max_scatter_points = max_scatter_points

        self.fuel_surface_prediction = output_curve_fitting["fuelSurface_prediction"]
        self.fuel_prediction_svr = output_curve_fitting["fuelPrediction_svr"]
        self.valid = output_curve_fitting["validMask"]

    def time_series_comparison(self) -> None:
        plt.figure(figsize=(10, 5))
        plt.plot(self.time_s[self.valid], self.fuel_consumption[self.valid], label="Conventional Speed+MAF")
        plt.plot(self.time_s[self.valid], self.fuel_surface_prediction[self.valid], label="Surface Fit RPM+TPS")
        plt.plot(self.time_s[self.valid], self.fuel_prediction_svr[self.valid], label="SVR RPM+TPS")
        plt.xlabel("Time (s)")
        plt.ylabel("Fuel consumption (L/100km)")
        plt.title("Fuel consumption vs time")
        plt.grid(True)
        plt.legend()
        plt.show()

    def parity_plots(self) -> None:
        idx = np.where(self.valid)[0]
        if len(idx) > self.max_scatter_points:
            rng = np.random.default_rng(0)
            idx = rng.choice(idx, size=self.max_scatter_points, replace=False)

        for prediction, title in [
            (self.fuel_surface_prediction, "Surface fit vs conventional"),
            (self.fuel_prediction_svr, "SVR prediction vs conventional"),
        ]:
            finite = np.isfinite(self.fuel_consumption[idx]) & np.isfinite(prediction[idx])
            plot_idx = idx[finite]
            if len(plot_idx) == 0:
                continue

            plt.figure(figsize=(6, 6))
            plt.scatter(self.fuel_consumption[plot_idx], prediction[plot_idx], s=8, alpha=0.3)
            mn = np.nanmin([self.fuel_consumption[plot_idx], prediction[plot_idx]])
            mx = np.nanmax([self.fuel_consumption[plot_idx], prediction[plot_idx]])
            plt.plot([mn, mx], [mn, mx], linewidth=1)
            plt.xlabel("Conventional fuel consumption (L/100km)")
            plt.ylabel("Predicted fuel consumption (L/100km)")
            plt.title(f"Parity: {title}")
            plt.grid(True)
            plt.show()

    def residual_plot(self) -> None:
        residual_surface = self.fuel_surface_prediction - self.fuel_consumption
        residual_svr = self.fuel_prediction_svr - self.fuel_consumption

        plt.figure(figsize=(10, 5))
        plt.plot(self.time_s[self.valid], residual_surface[self.valid], label="Residual Surface")
        plt.plot(self.time_s[self.valid], residual_svr[self.valid], label="Residual SVR")
        plt.axhline(0, linewidth=1)
        plt.xlabel("Time (s)")
        plt.ylabel("Residual (L/100km)")
        plt.title("Residuals vs time")
        plt.grid(True)
        plt.legend()
        plt.show()


# -----------------------------------------------------------------------------
# Command-line runner
# -----------------------------------------------------------------------------

def run_file(
    input_csv: str | Path,
    output_csv: Optional[str | Path] = None,
    fuel_tank_volume_l: float = 50.0,
    fuel_type: str = "petrol",
    make_plots: bool = False,
) -> Path:
    df = load_group_obd_csv(input_csv)
    results, fit_output = add_physical_outputs(
        df,
        fuel_tank_volume_l=fuel_tank_volume_l,
        fuel_type=fuel_type,
    )

    input_csv = Path(input_csv)
    if output_csv is None:
        output_csv = input_csv.with_name(f"{input_csv.stem}_fuel_results{input_csv.suffix}")
    else:
        output_csv = Path(output_csv)

    results.to_csv(output_csv, index=False)

    print("\nFuel model metrics:")
    for key, value in fit_output["Metrics"].items():
        print(f"  {key}: {value}")

    print(f"\nSaved fuel results: {output_csv}\n")

    if make_plots:
        plots = FuelConsumptionPlots(
            time_s=get_time_seconds(results),
            speed_kmh=get_series(results, GROUP_COLUMNS["speed"]),
            fuel_consumption=get_series(results, "fuel_consumption_l_per_100km"),
            output_curve_fitting=fit_output,
        )
        plots.time_series_comparison()
        plots.parity_plots()
        plots.residual_plot()

    return Path(output_csv)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run fuel/physics calculations on group-format OBD CSV data.")
    parser.add_argument("input_csv", help="Group cleaned/smoothed CSV, e.g. Leon1_clean_smoothed.csv")
    parser.add_argument("--output", help="Optional output CSV path")
    parser.add_argument("--fuel-tank-volume-l", type=float, default=50.0)
    parser.add_argument("--fuel-type", choices=["petrol", "diesel"], default="petrol")
    parser.add_argument("--plots", action="store_true", help="Show diagnostic plots")
    args = parser.parse_args()

    run_file(
        input_csv=args.input_csv,
        output_csv=args.output,
        fuel_tank_volume_l=args.fuel_tank_volume_l,
        fuel_type=args.fuel_type,
        make_plots=args.plots,
    )


if __name__ == "__main__":
    main()
