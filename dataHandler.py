# OBD-II CSV Cleaner
# Standardises headers and removes rows with missing values
# Usage: python dataHandler.py Leon1.csv (or whatever the CSV is called. Remember to use put file types after e.g. .csv .py etc...)
# WILL INWOOD

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd


# Header names

def normalise_header_name(name: str) -> str:
    """Make header text easier to match."""
    if name is None:
        return ""

    s = str(name).strip().lower()
    s = re.sub(r"[\[\]\(\)\/\\\|\-_\:\,\.\%\°]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


# Header map

def build_synonym_map() -> Dict[str, List[str]]:
    return {
        "timestamp": [
            "timestamp",
            "date time",
            "datetime",
            "time stamp",
            "time",
            "time step",
            "timestep",
            "time_step",
        ],
        "engine_rpm": [
            "engine rpm",
            "rpm",
            "motor rpm",
            "eng rpm",
            "Engine RPM [RPM]",
        ],
        "vehicle_speed": [
            "vehicle speed",
            "speed",
            "vehicle speed sensor",
            "vss",
            "gps speed",
            "Vehicle Speed Sensor [km/h]",
        ],
        "coolant_temp": [
            "engine coolant temperature",
            "coolant temperature",
            "coolant temp",
            "ect",
            "Engine Coolant Temperature [Ã‚Â°C]",
            "Engine Coolant Temperature [Â°C]",
        ],
        "intake_temp": [
            "intake air temperature",
            "iat",
            "intake temp",
            "Intake Air Temperature [Ã‚Â°C]",
            "Intake Air Temperature [Â°C]",
        ],
        "ambient_temp": [
            "ambient air temperature",
            "ambient temp",
            "outside temp",
            "external temperature",
            "Ambient Air Temperature [Ã‚Â°C]",
            "Ambient Air Temperature [Â°C]",
        ],
        "throttle_pct": [
            "absolute throttle position",
            "throttle position",
            "throttle",
            "Absolute Throttle Position [%]",
        ],
        "pedal_pct_d": [
            "accelerator pedal position d",
            "pedal position d",
            "Accelerator Pedal Position D [%]",
        ],
        "pedal_pct_e": [
            "accelerator pedal position e",
            "pedal position e",
            "Accelerator Pedal Position E [%]",
        ],
        "pedal_pct": [
            "accelerator pedal position",
            "pedal position",
        ],
        "engine_load_pct": [
            "engine load",
            "calculated load",
            "Calculated Engine Load [%]",
        ],
        "maf_gps": [
            "mass air flow",
            "maf",
            "Air Flow Rate from Mass Flow Sensor [g/s]",
        ],
        "map_kpa": [
            "manifold absolute pressure",
            "intake manifold absolute pressure",
            "map",
            "Intake Manifold Absolute Pressure [kPa]",
        ],
    }


# Header match

def match_canonical_name(raw_header: str, synonym_map: Dict[str, List[str]]) -> Optional[str]:
    h = normalise_header_name(raw_header)

    best_match = None
    best_length = 0

    for canonical, synonyms in synonym_map.items():
        for syn in synonyms:
            syn_norm = normalise_header_name(syn)
            if syn_norm and syn_norm in h:
                if len(syn_norm) > best_length:
                    best_length = len(syn_norm)
                    best_match = canonical

    return best_match


# Standardise headers

def standardise_headers(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, str]]:
    synonym_map = build_synonym_map()

    rename_map: Dict[str, str] = {}
    counts: Dict[str, int] = {}

    for col in df.columns:
        canonical = match_canonical_name(col, synonym_map)
        if canonical is None:
            continue

        counts[canonical] = counts.get(canonical, 0) + 1
        new_name = canonical if counts[canonical] == 1 else f"{canonical}_{counts[canonical]}"
        rename_map[col] = new_name

    df = df.rename(columns=rename_map)
    return df, rename_map


# Numeric columns

def coerce_numeric_if_present(df: pd.DataFrame) -> pd.DataFrame:
    """Convert known numeric columns."""
    df = df.copy()

    candidates = [
        "engine_rpm",
        "engine_rpm_2",
        "vehicle_speed",
        "vehicle_speed_2",
        "coolant_temp",
        "coolant_temp_2",
        "intake_temp",
        "ambient_temp",
        "throttle_pct",
        "pedal_pct",
        "pedal_pct_d",
        "pedal_pct_e",
        "engine_load_pct",
        "maf_gps",
        "map_kpa",
        "timestamp",
    ]

    for col in candidates:
        if col in df.columns and col != "timestamp":
            cleaned = df[col].astype(str).str.replace(",", "", regex=False).str.strip()
            df[col] = pd.to_numeric(cleaned, errors="coerce")

    return df


# Blank rows

def remove_rows_with_blanks(df: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
    before = len(df)
    df = df.copy()

    df = df.replace(r"^\s*$", pd.NA, regex=True)
    df = df.dropna(axis=0, how="any")

    removed = before - len(df)
    return df, removed


# File load

def load_input_file(input_path: Path) -> pd.DataFrame:
    """Load csv or excel."""
    suffix = input_path.suffix.lower()

    if suffix == ".csv":
        return pd.read_csv(input_path, dtype=str, encoding_errors="ignore")

    if suffix in [".xlsx", ".xls"]:
        return pd.read_excel(input_path, dtype=str)

    raise ValueError(
        f"Unsupported file type: {input_path.suffix}. "
        "Please provide a .csv, .xlsx, or .xls file."
    )


# Main clean

def clean_obd_csv(input_path: str, output_path: Optional[str] = None) -> Path:
    """Load, clean, save csv."""
    in_path = Path(input_path)

    if output_path is None:
        out_path = in_path.with_name(f"{in_path.stem}_clean.csv")
    else:
        out_path = Path(output_path)

    df = load_input_file(in_path)

    print(f"\nLoaded: {in_path.name}")
    print(f"Rows: {len(df):,} | Cols: {len(df.columns):,}")

    df, rename_map = standardise_headers(df)

    print("\nHeader standardisation:")
    if rename_map:
        for old, new in rename_map.items():
            print(f"  - {old!r} -> {new!r}")
    else:
        print("  (No headers matched known synonyms.)")

    df = coerce_numeric_if_present(df)

    df, removed = remove_rows_with_blanks(df)
    print(f"\nRemoved rows with blanks: {removed:,}")
    print(f"Remaining rows: {len(df):,}")

    df.to_csv(out_path, index=False)
    print(f"\nSaved cleaned CSV: {out_path}\n")

    return out_path


# CLI

def main(argv: List[str]) -> None:
    if len(argv) < 2:
        print("Usage: python dataHandlerV2.py <input.csv_or_xlsx>")
        raise SystemExit(1)

    input_path = argv[1]
    clean_obd_csv(input_path)


if __name__ == "__main__":
    main(sys.argv)
