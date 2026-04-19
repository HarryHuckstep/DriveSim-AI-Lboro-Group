# OBD-II CSV Smoother
# Creates abs time and replaces the Timestamp column. Lightly smooths data before analysis.
# Usage: python dataSmoother.py Leon1.csv (or whatever the CSV is called. Remember to use put file types after e.g. .csv .py etc...)
# WILL INWOOD

from pathlib import Path
import sys
import numpy as np
import pandas as pd



# Convert timestamp to seconds from start


def timestamp_to_seconds_from_start(df: pd.DataFrame) -> pd.DataFrame:

    df = df.copy()

    if "timestamp" not in df.columns:
        return df

    ts = df["timestamp"].astype(str).str.strip()

    parts = ts.str.split(":", n=1, expand=True)

    if parts.shape[1] != 2:
        raise ValueError(
            "Timestamp format not recognised. Expected format like '16:31.2'"
        )

    mins = pd.to_numeric(parts[0], errors="coerce")
    secs = pd.to_numeric(parts[1], errors="coerce")

    abs_time = (mins * 60.0) + secs

    # make time start at 0
    t_from_start = abs_time - abs_time.iloc[0]

    df["timestamp"] = t_from_start

    # also update common timestep columns if present
    for col in ["timestep", "time step", "time_step", "t_s"]:
        if col in df.columns:
            df[col] = t_from_start

    return df



# Light smoothing function


def light_smooth_series(series: pd.Series) -> pd.Series:

    s = pd.to_numeric(series, errors="coerce")

    # remove small spikes
    s1 = s.rolling(window=3, center=True, min_periods=1).median()

    # gentle smoothing
    s2 = s1.ewm(alpha=0.35, adjust=False).mean()

    # tidy result
    s3 = s2.rolling(window=3, center=True, min_periods=1).mean()

    return s3



# Apply smoothing to dataframe


def smooth_obd_dataframe(df: pd.DataFrame):

    df = df.copy()
    out = df.copy()

    numeric_cols = []

    for col in df.columns:

        converted = pd.to_numeric(df[col], errors="coerce")

        if converted.notna().sum() > 0:
            numeric_cols.append(col)

    change_report = {}

    for col in numeric_cols:

        before = pd.to_numeric(df[col], errors="coerce")

        # do NOT smooth time columns
        if col in ["timestamp", "timestep", "time step", "time_step", "t_s"]:
            after = before
        else:
            after = light_smooth_series(before)

        out[col] = after

        changed_mask = ~np.isclose(before, after, equal_nan=True, atol=1e-12)
        change_report[col] = int(changed_mask.sum())

    return out, change_report



# File wrapper


def smooth_csv_file(input_path: str):

    input_path = Path(input_path)

    df = pd.read_csv(input_path)

    # convert timestamp first
    df = timestamp_to_seconds_from_start(df)

    # smooth signals
    df_smoothed, change_report = smooth_obd_dataframe(df)

    output_path = input_path.with_name(
        f"{input_path.stem}_smoothed{input_path.suffix}"
    )

    df_smoothed.to_csv(output_path, index=False)

    print("\nChanged values per column:")
    for col, n in change_report.items():
        print(f"  {col}: {n}")

    print("\nFirst 10 timestamps:")
    if "timestamp" in df_smoothed.columns:
        print(df_smoothed["timestamp"].head(10))

    return output_path



# Main runner/CLI


if __name__ == "__main__":

    if len(sys.argv) < 2:
        print("Usage: python dataSmoother.py <input.csv>")
        sys.exit(1)

    input_path = sys.argv[1]

    output_path = smooth_csv_file(input_path)

    print(f"\nSaved smoothed file: {output_path}\n")