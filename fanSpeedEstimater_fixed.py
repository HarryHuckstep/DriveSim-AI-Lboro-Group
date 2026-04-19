from pathlib import Path
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


DEFAULT_DT = 0.1

FAN_OFF_TEMP = 88.0
FAN_LOW_TEMP = 92.0
FAN_HIGH_TEMP = 96.0

SPEED_FULL_FAN_EFFECT_KPH = 50.0

FAN_MIN_RPM = 0.0
FAN_MAX_RPM = 2500.0

USE_ENGINE_HEAT_ASSIST = True
HEAT_ASSIST_GAIN = 0.15

AFR_STOICH = 14.7
FUEL_LHV_J_PER_KG = 44e6
FRACTION_FUEL_TO_COOLANT = 0.30

ASSUMED_MAX_ENGINE_POWER_W = 100e3
ASSUMED_THERMAL_EFFICIENCY = 0.28


def get_time_axis(df: pd.DataFrame) -> pd.Series:
    if "timestamp" in df.columns:
        t = pd.to_numeric(df["timestamp"], errors="coerce")
        if t.notna().sum() > 1:
            return t

    if "t_s" in df.columns:
        t = pd.to_numeric(df["t_s"], errors="coerce")
        if t.notna().sum() > 1:
            return t

    return pd.Series(np.arange(len(df)) * DEFAULT_DT, index=df.index, dtype=float)


def smooth_series(series: pd.Series, window: int = 5) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    s1 = s.rolling(window=window, center=True, min_periods=1).median()
    s2 = s1.rolling(window=window, center=True, min_periods=1).mean()
    return s2


def pick_air_temp(df: pd.DataFrame) -> pd.Series:
    if "ambient_temp" in df.columns:
        return pd.to_numeric(df["ambient_temp"], errors="coerce").ffill().bfill()

    if "intake_temp" in df.columns:
        return pd.to_numeric(df["intake_temp"], errors="coerce").ffill().bfill()

    return pd.Series(np.full(len(df), 20.0), index=df.index, dtype=float)


def estimate_engine_heat_to_coolant(df: pd.DataFrame) -> pd.Series:
    if "maf_gps" in df.columns:
        maf_gps = pd.to_numeric(df["maf_gps"], errors="coerce")
        maf_kgps = maf_gps / 1000.0

        fuel_kgps = maf_kgps / AFR_STOICH
        fuel_power = fuel_kgps * FUEL_LHV_J_PER_KG
        q_coolant = FRACTION_FUEL_TO_COOLANT * fuel_power

        if q_coolant.notna().sum() > 0:
            return q_coolant.ffill().bfill().fillna(0.0)

    if "engine_load_pct" in df.columns:
        load = pd.to_numeric(df["engine_load_pct"], errors="coerce") / 100.0
        load = load.clip(lower=0.0, upper=1.2)

        shaft_power = load * ASSUMED_MAX_ENGINE_POWER_W
        fuel_power = shaft_power / max(ASSUMED_THERMAL_EFFICIENCY, 1e-6)
        q_coolant = FRACTION_FUEL_TO_COOLANT * fuel_power
        return q_coolant.fillna(0.0)

    return pd.Series(np.zeros(len(df)), index=df.index, dtype=float)


def estimate_fan_speed(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "coolant_temp" not in df.columns:
        raise ValueError("Missing required column: 'coolant_temp'")

    coolant = pd.to_numeric(df["coolant_temp"], errors="coerce").ffill().bfill()
    coolant_smooth = smooth_series(coolant, window=7)

    air_temp = pick_air_temp(df)
    air_temp_smooth = smooth_series(air_temp, window=7)

    if "vehicle_speed" in df.columns:
        vehicle_speed = pd.to_numeric(df["vehicle_speed"], errors="coerce").fillna(0.0)
    else:
        vehicle_speed = pd.Series(np.zeros(len(df)), index=df.index, dtype=float)

    vehicle_speed = vehicle_speed.clip(lower=0.0)

    q_engine = estimate_engine_heat_to_coolant(df)
    q_engine_smooth = smooth_series(q_engine, window=7)

    fan_temp_demand = pd.Series(np.zeros(len(df)), index=df.index, dtype=float)

    mask1 = (coolant_smooth >= FAN_OFF_TEMP) & (coolant_smooth < FAN_LOW_TEMP)
    fan_temp_demand.loc[mask1] = (
        (coolant_smooth.loc[mask1] - FAN_OFF_TEMP) /
        (FAN_LOW_TEMP - FAN_OFF_TEMP)
    ) * 0.4

    mask2 = (coolant_smooth >= FAN_LOW_TEMP) & (coolant_smooth < FAN_HIGH_TEMP)
    fan_temp_demand.loc[mask2] = 0.4 + (
        (coolant_smooth.loc[mask2] - FAN_LOW_TEMP) /
        (FAN_HIGH_TEMP - FAN_LOW_TEMP)
    ) * 0.6

    mask3 = coolant_smooth >= FAN_HIGH_TEMP
    fan_temp_demand.loc[mask3] = 1.0

    speed_factor = 1.0 - (vehicle_speed / SPEED_FULL_FAN_EFFECT_KPH)
    speed_factor = speed_factor.clip(lower=0.15, upper=1.0)

    if USE_ENGINE_HEAT_ASSIST:
        heat_norm = q_engine_smooth / max(q_engine_smooth.max(), 1.0)
        heat_assist = heat_norm * HEAT_ASSIST_GAIN
    else:
        heat_assist = pd.Series(np.zeros(len(df)), index=df.index, dtype=float)

    delta_t = (coolant_smooth - air_temp_smooth).clip(lower=0.0)
    delta_t_factor = (delta_t / 25.0).clip(lower=0.2, upper=1.0)

    fan_demand = (fan_temp_demand * speed_factor * delta_t_factor) + heat_assist
    fan_demand = fan_demand.clip(lower=0.0, upper=1.0)

    fan_speed_est = FAN_MIN_RPM + fan_demand * (FAN_MAX_RPM - FAN_MIN_RPM)
    fan_speed_est = smooth_series(fan_speed_est, window=5)

    fan_state = pd.Series(np.zeros(len(df)), index=df.index, dtype=int)
    fan_state.loc[fan_speed_est > 200] = 1
    fan_state.loc[fan_speed_est > 1200] = 2

    df["coolant_temp_smooth"] = coolant_smooth
    df["air_temp_used"] = air_temp_smooth
    df["q_engine_to_coolant_W"] = q_engine_smooth
    df["fan_temp_demand"] = fan_temp_demand
    df["speed_factor"] = speed_factor
    df["delta_t_factor"] = delta_t_factor
    df["fan_demand"] = fan_demand
    df["fan_speed_est"] = fan_speed_est
    df["fan_state"] = fan_state

    return df


def plot_fan_speed_vs_time(df: pd.DataFrame) -> None:
    t = get_time_axis(df)

    plt.figure(figsize=(10, 4))
    plt.plot(t, df["fan_speed_est"])
    plt.xlabel("Time")
    plt.ylabel("Estimated Fan Speed (RPM)")
    plt.title("Estimated Fan Speed vs Time")
    plt.grid(True)


def plot_coolant_and_fan(df: pd.DataFrame) -> None:
    t = get_time_axis(df)

    fig, ax1 = plt.subplots(figsize=(10, 4))

    ax1.plot(t, df["coolant_temp"], label="Coolant Temp", alpha=0.7)
    ax1.plot(t, df["coolant_temp_smooth"], label="Coolant Temp Smoothed", linewidth=2)
    ax1.set_xlabel("Time")
    ax1.set_ylabel("Coolant Temp (C)")
    ax1.grid(True)

    ax2 = ax1.twinx()
    ax2.plot(t, df["fan_speed_est"], label="Fan Speed Est", linestyle="--")
    ax2.set_ylabel("Fan Speed Est (RPM)")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="best")

    plt.title("Coolant Temperature and Estimated Fan Speed")


def plot_fan_demand_terms(df: pd.DataFrame) -> None:
    t = get_time_axis(df)

    plt.figure(figsize=(10, 4))
    plt.plot(t, df["fan_temp_demand"], label="Temp Demand")
    plt.plot(t, df["speed_factor"], label="Speed Factor")
    plt.plot(t, df["delta_t_factor"], label="Delta-T Factor")
    plt.plot(t, df["fan_demand"], label="Final Fan Demand", linewidth=2)
    plt.xlabel("Time")
    plt.ylabel("Normalised Value")
    plt.title("Fan Demand Terms")
    plt.legend()
    plt.grid(True)


def process_file(input_path: str) -> Path:
    input_path = Path(input_path)

    df = pd.read_csv(input_path)
    df = estimate_fan_speed(df)

    output_path = input_path.with_name(f"{input_path.stem}_fan{input_path.suffix}")
    df.to_csv(output_path, index=False)

    print(f"\nSaved fan-estimated file: {output_path}")

    print("\nFan speed summary:")
    print(df["fan_speed_est"].describe())

    print("\nFan state counts:")
    print(df["fan_state"].value_counts().sort_index())

    plot_fan_speed_vs_time(df)
    plot_coolant_and_fan(df)
    plot_fan_demand_terms(df)
    plt.show()

    return output_path


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python fanSpeedEstimater.py <input.csv>")
        sys.exit(1)

    input_csv = sys.argv[1]
    process_file(input_csv)
