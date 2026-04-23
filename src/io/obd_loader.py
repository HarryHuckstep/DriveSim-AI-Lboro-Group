from __future__ import annotations
import pandas as pd

DEFAULT_COLUMN_MAP = {
    "Time": "time",
    "timestamp": "time",
    "Vehicle Speed Sensor [km/h]": "speed_kmh",
    "vehicle_speed": "speed_kmh",
    "Engine RPM [RPM]": "engine_rpm",
    # add more as needed
}

def load_obd_csv(path: str, *, column_map: dict[str, str] | None = None) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip() 
    cmap = DEFAULT_COLUMN_MAP if column_map is None else column_map
    return df.rename(columns=cmap)

def require_columns(df: pd.DataFrame, required: list[str]) -> None: #??
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"Missing columns {missing}. Available: {list(df.columns)}")











'''
def load_obd_csv(path: str, *, column_map: dict[str, str] | None = None) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip() 
    cmap = DEFAULT_COLUMN_MAP if column_map is None else column_map
    return df.rename(columns=cmap)

def require_columns(df: pd.DataFrame, required: list[str]) -> None: #??
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"Missing columns {missing}. Available: {list(df.columns)}")
'''