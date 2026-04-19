from pathlib import Path
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans



# 1) Config


N_GEARS = 7             # Change this if the car is 6-speed etc.
MIN_SPEED = 5.0          # Ignore very low speed rows for gear detection
MIN_RPM = 500.0          # Ignore engine-off / idle-noise rows
RANDOM_STATE = 42
SHIFT_SMOOTHING_WINDOW = 3   # Median smoothing on final gear trace



# 2) Time axis helper


def get_time_axis(df: pd.DataFrame) -> pd.Series:
    """
    Use the best available time axis for plotting.
    """
    if "t_s" in df.columns:
        return pd.to_numeric(df["t_s"], errors="coerce")

    if "timestamp" in df.columns:
        ts = pd.to_numeric(df["timestamp"], errors="coerce")
        if ts.notna().sum() > 0:
            return ts

    return pd.Series(np.arange(len(df)), index=df.index, dtype=float)



# 3) Core gear estimation


def estimate_gears(df: pd.DataFrame, n_gears: int = N_GEARS) -> pd.DataFrame:
    """
    Estimate gear from the ratio:
        gear_proxy = engine_rpm / vehicle_speed

    Logic:
    - filter out low-speed / low-rpm rows
    - cluster the ratio values using K-means
    - sort clusters by mean ratio
    - highest ratio = lowest gear
    - lowest ratio = highest gear
    """
    df = df.copy()

    # Required columns
    if "engine_rpm" not in df.columns:
        raise ValueError("Missing required column: 'engine_rpm'")
    if "vehicle_speed" not in df.columns:
        raise ValueError("Missing required column: 'vehicle_speed'")

    rpm = pd.to_numeric(df["engine_rpm"], errors="coerce")
    speed = pd.to_numeric(df["vehicle_speed"], errors="coerce")

    # Create valid mask for rows suitable for gear estimation
    valid_mask = (
        rpm.notna()
        & speed.notna()
        & (rpm >= MIN_RPM)
        & (speed >= MIN_SPEED)
    )

    # Gear proxy
    gear_ratio_proxy = pd.Series(np.nan, index=df.index, dtype=float)
    gear_ratio_proxy.loc[valid_mask] = rpm.loc[valid_mask] / speed.loc[valid_mask]
    df["gear_ratio_proxy"] = gear_ratio_proxy

    # Start with NaN gear column
    df["gear"] = np.nan

    # Not enough points to cluster
    valid_ratios = gear_ratio_proxy.loc[valid_mask].dropna()
    if len(valid_ratios) < n_gears:
        print("Not enough valid points for clustering.")
        return df

    # K-means expects 2D array
    X = valid_ratios.to_numpy().reshape(-1, 1)

    kmeans = KMeans(
        n_clusters=n_gears,
        random_state=RANDOM_STATE,
        n_init=20
    )
    cluster_labels = kmeans.fit_predict(X)

    # Mean ratio for each cluster
    cluster_means = {}
    for c in range(n_gears):
        cluster_means[c] = float(valid_ratios.iloc[cluster_labels == c].mean())

    # Sort clusters by ratio descending:
    # highest ratio -> 1st gear
    # lowest ratio -> top gear
    sorted_clusters = sorted(cluster_means.items(), key=lambda x: x[1], reverse=True)

    cluster_to_gear = {}
    for gear_number, (cluster_id, _) in enumerate(sorted_clusters, start=1):
        cluster_to_gear[cluster_id] = gear_number

    # Map cluster labels back to gear numbers
    estimated_gears = pd.Series(index=valid_ratios.index, dtype=float)
    for idx, cluster_id in zip(valid_ratios.index, cluster_labels):
        estimated_gears.loc[idx] = cluster_to_gear[cluster_id]

    df.loc[estimated_gears.index, "gear"] = estimated_gears

    return df



# 4) Gear smoothing / cleanup


def smooth_gear_trace(df: pd.DataFrame) -> pd.DataFrame:
    """
    Light cleanup of the gear trace.

    We use a rolling median on the gear column to suppress
    one-sample misclassifications during shifts/noise.
    """
    df = df.copy()

    if "gear" not in df.columns:
        return df

    gear = pd.to_numeric(df["gear"], errors="coerce")

    # Median smoothing only on valid values
    gear_smooth = gear.rolling(
        window=SHIFT_SMOOTHING_WINDOW,
        center=True,
        min_periods=1
    ).median()

    df["gear"] = gear_smooth.round()

    return df



# 5) Plotting


def plot_rpm_vs_speed(df: pd.DataFrame) -> None:
    """
    Scatter plot of RPM vs speed, coloured by estimated gear.
    """
    if "gear" not in df.columns:
        return

    speed = pd.to_numeric(df["vehicle_speed"], errors="coerce")
    rpm = pd.to_numeric(df["engine_rpm"], errors="coerce")
    gear = pd.to_numeric(df["gear"], errors="coerce")

    valid = speed.notna() & rpm.notna() & gear.notna()

    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(
        speed[valid],
        rpm[valid],
        c=gear[valid],
        s=8,
        alpha=0.7
    )
    plt.xlabel("Vehicle Speed")
    plt.ylabel("Engine RPM")
    plt.title("RPM vs Speed coloured by estimated gear")
    plt.grid(True)
    plt.colorbar(scatter, label="Gear")


def plot_ratio_distribution(df: pd.DataFrame) -> None:
    """
    Histogram of the gear ratio proxy.
    """
    if "gear_ratio_proxy" not in df.columns:
        return

    ratio = pd.to_numeric(df["gear_ratio_proxy"], errors="coerce").dropna()

    if len(ratio) == 0:
        return

    plt.figure(figsize=(8, 4))
    plt.hist(ratio, bins=60)
    plt.xlabel("RPM / Speed ratio")
    plt.ylabel("Count")
    plt.title("Gear ratio proxy distribution")
    plt.grid(True)


def plot_gear_vs_time(df: pd.DataFrame) -> None:
    """
    Plot estimated gear over time.
    """
    if "gear" not in df.columns:
        return

    t = get_time_axis(df)
    gear = pd.to_numeric(df["gear"], errors="coerce")

    valid = t.notna() & gear.notna()

    plt.figure(figsize=(10, 4))
    plt.plot(t[valid], gear[valid])
    plt.xlabel("Time")
    plt.ylabel("Gear")
    plt.title("Estimated gear vs time")
    plt.grid(True)



# 6) File wrapper


def process_file(input_path: str, n_gears: int = N_GEARS) -> Path:
    input_path = Path(input_path)

    df = pd.read_csv(input_path)

    df = estimate_gears(df, n_gears=n_gears)
    df = smooth_gear_trace(df)

    output_path = input_path.with_name(f"{input_path.stem}_gears{input_path.suffix}")
    df.to_csv(output_path, index=False)

    print(f"\nSaved gear-estimated file: {output_path}")

    if "gear" in df.columns:
        print("\nEstimated gear counts:")
        print(df["gear"].value_counts(dropna=False).sort_index())

    # Plots
    plot_rpm_vs_speed(df)
    plot_ratio_distribution(df)
    plot_gear_vs_time(df)
    plt.show()

    return output_path



# 7) Main runner


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python gearEstimater.py <input.csv> [number_of_gears]")
        sys.exit(1)

    input_csv = sys.argv[1]

    if len(sys.argv) >= 3:
        n_gears = int(sys.argv[2])
    else:
        n_gears = N_GEARS

    process_file(input_csv, n_gears=n_gears)