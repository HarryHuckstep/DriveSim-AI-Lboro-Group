from __future__ import annotations
import pandas as pd
#creates an array for elapsed time in seconds
def add_elapsed_time(df: pd.DataFrame, *, time_col: str = "time", out_col: str = "elapsed_time_s") -> pd.DataFrame:
    t = pd.to_datetime(df[time_col], format="%H:%M:%S.%f", errors="coerce")
    if t.isna().all():
        t = pd.to_datetime(df[time_col], errors="coerce")
    df[out_col] = (t - t.iloc[0]).dt.total_seconds()
    return df
