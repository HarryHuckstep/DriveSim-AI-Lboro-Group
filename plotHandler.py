from pathlib import Path
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def load_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def get_time_axis(df: pd.DataFrame) -> pd.Series:
    if "timestamp" in df.columns:
        t = pd.to_numeric(df["timestamp"], errors="coerce")
        if t.notna().sum() > 0:
            return t

    return pd.Series(np.arange(len(df)), index=df.index, dtype=float)


def get_numeric_series(df: pd.DataFrame, column: str) -> pd.Series:
    return pd.to_numeric(df[column], errors="coerce")


def plot_overlay(raw_df: pd.DataFrame, processed_df: pd.DataFrame, column: str, title: str) -> None:
    if column not in raw_df.columns or column not in processed_df.columns:
        return

    t_raw = get_time_axis(raw_df)
    t_processed = get_time_axis(processed_df)

    y_raw = get_numeric_series(raw_df, column)
    y_processed = get_numeric_series(processed_df, column)

    plt.figure(figsize=(10, 4))
    plt.plot(t_raw, y_raw, label="raw", alpha=0.6)
    plt.plot(t_processed, y_processed, label="processed", linewidth=2)
    plt.xlabel("time (s)")
    plt.ylabel(column)
    plt.title(title)
    plt.grid(True)
    plt.legend()


def plot_raw_vs_smoothed(raw_df: pd.DataFrame, smoothed_df: pd.DataFrame) -> None:
    columns = [
        "engine_rpm",
        "vehicle_speed",
        "coolant_temp",
        "maf_gps",
        "map_kpa",
        "throttle_pct",
        "ambient_temp",
        "intake_temp",
        "pedal_pct_d",
        "pedal_pct_e",
        "engine_load_pct",
    ]

    for column in columns:
        if column in raw_df.columns and column in smoothed_df.columns:
            plot_overlay(raw_df, smoothed_df, column, f"{column}: raw vs smoothed")


def plot_single_file_overview(df: pd.DataFrame, file_label: str) -> None:
    t = get_time_axis(df)

    basic_columns = [
        "engine_rpm",
        "vehicle_speed",
        "coolant_temp",
        "maf_gps",
        "throttle_pct",
    ]

    for column in basic_columns:
        if column in df.columns:
            y = get_numeric_series(df, column)

            plt.figure(figsize=(10, 4))
            plt.plot(t, y)
            plt.xlabel("time (s)")
            plt.ylabel(column)
            plt.title(f"{file_label}: {column}")
            plt.grid(True)


def plot_gear_results(df: pd.DataFrame, file_label: str) -> None:
    if "gear" not in df.columns:
        return

    t = get_time_axis(df)
    gear = get_numeric_series(df, "gear")

    plt.figure(figsize=(10, 4))
    plt.plot(t, gear)
    plt.xlabel("time (s)")
    plt.ylabel("gear")
    plt.title(f"{file_label}: gear vs time")
    plt.grid(True)

    if "engine_rpm" in df.columns and "vehicle_speed" in df.columns:
        rpm = get_numeric_series(df, "engine_rpm")
        speed = get_numeric_series(df, "vehicle_speed")
        valid = rpm.notna() & speed.notna() & gear.notna()

        plt.figure(figsize=(8, 6))
        scatter = plt.scatter(speed[valid], rpm[valid], c=gear[valid], s=8, alpha=0.7)
        plt.xlabel("vehicle_speed")
        plt.ylabel("engine_rpm")
        plt.title(f"{file_label}: RPM vs speed coloured by gear")
        plt.grid(True)
        plt.colorbar(scatter, label="gear")

    if "gear_ratio_proxy" in df.columns:
        ratio = get_numeric_series(df, "gear_ratio_proxy").dropna()

        plt.figure(figsize=(8, 4))
        plt.hist(ratio, bins=60)
        plt.xlabel("gear_ratio_proxy")
        plt.ylabel("count")
        plt.title(f"{file_label}: gear ratio proxy distribution")
        plt.grid(True)


def plot_fan_results(df: pd.DataFrame, file_label: str) -> None:
    if "fan_speed_est" not in df.columns:
        return

    t = get_time_axis(df)
    fan = get_numeric_series(df, "fan_speed_est")

    plt.figure(figsize=(10, 4))
    plt.plot(t, fan)
    plt.xlabel("time (s)")
    plt.ylabel("fan_speed_est")
    plt.title(f"{file_label}: fan speed vs time")
    plt.grid(True)

    if "coolant_temp" in df.columns:
        coolant = get_numeric_series(df, "coolant_temp")

        fig, ax1 = plt.subplots(figsize=(10, 4))
        ax1.plot(t, coolant, label="coolant_temp", alpha=0.7)
        ax1.set_xlabel("time (s)")
        ax1.set_ylabel("coolant_temp")
        ax1.grid(True)

        ax2 = ax1.twinx()
        ax2.plot(t, fan, label="fan_speed_est", linestyle="--")
        ax2.set_ylabel("fan_speed_est")

        plt.title(f"{file_label}: coolant and fan")

    if "fan_demand" in df.columns:
        demand = get_numeric_series(df, "fan_demand")

        plt.figure(figsize=(10, 4))
        plt.plot(t, demand)
        plt.xlabel("time (s)")
        plt.ylabel("fan_demand")
        plt.title(f"{file_label}: fan demand")
        plt.grid(True)

    if "coolant_temp" in df.columns:
        coolant = get_numeric_series(df, "coolant_temp")
        valid = coolant.notna() & fan.notna()

        plt.figure(figsize=(8, 5))
        plt.scatter(coolant[valid], fan[valid], s=5)
        plt.xlabel("coolant_temp")
        plt.ylabel("fan_speed_est")
        plt.title(f"{file_label}: fan vs coolant temp")
        plt.grid(True)


def plot_file_by_columns(df: pd.DataFrame, file_label: str) -> None:
    plot_single_file_overview(df, file_label)
    plot_gear_results(df, file_label)
    plot_fan_results(df, file_label)


def plot_results(file_paths: list[str]) -> None:
    if len(file_paths) == 1:
        df = load_csv(file_paths[0])
        plot_file_by_columns(df, Path(file_paths[0]).name)
        plt.show()
        return

    if len(file_paths) >= 2:
        raw_df = load_csv(file_paths[0])
        smoothed_df = load_csv(file_paths[1])

        plot_raw_vs_smoothed(raw_df, smoothed_df)

        for extra_path in file_paths[2:]:
            df = load_csv(extra_path)
            plot_file_by_columns(df, Path(extra_path).name)

        plt.show()


def print_usage() -> None:
    print(
        "\nUsage:\n"
        "python plotResults.py <file1.csv> [file2.csv] [file3.csv] ...\n\n"
        "Examples:\n"
        "python plotResults.py NewDrive_clean_smoothed.csv\n"
        "python plotResults.py NewDrive.csv NewDrive_clean_smoothed.csv\n"
        "python plotResults.py NewDrive.csv NewDrive_clean_smoothed.csv NewDrive_clean_smoothed_gears.csv\n"
        "python plotResults.py NewDrive.csv NewDrive_clean_smoothed.csv NewDrive_clean_smoothed_fan.csv\n"
    )


def main(argv: list[str]) -> None:
    if len(argv) < 2:
        print_usage()
        raise SystemExit(1)

    file_paths = argv[1:]
    plot_results(file_paths)


if __name__ == "__main__":
    main(sys.argv)