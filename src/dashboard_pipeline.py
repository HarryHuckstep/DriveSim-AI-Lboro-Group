import sys
from pathlib import Path
import matplotlib.pyplot as plt
HERE = Path().resolve()
PROJECT_ROOT = next(p for p in [HERE, *HERE.parents] if (p / "src").exists())
sys.path.insert(0, str(PROJECT_ROOT)) 
from src.io.obd_loader import load_obd_csv, require_columns
from src.io.timebase import add_elapsed_time
from src.physics.kinematics import add_speed_ms, add_acceleration
from src.physics.longitudinal import VehicleParams, add_longitudinal_forces, plot_longitudinal_forces
from src.physics.power_energy import add_power_terms, add_energy_terms, plot_power, plot_cumulative_energy
from src.features.efficiency import add_chemical_efficiency
from dataHandler import clean_obd_csv
from dataSmoother import smooth_csv_file
import tempfile


def prepare_dashboard_df(df_raw, params): #add option for params later
    
    
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
        df_raw.to_csv(tmp.name, index=False)

    
    smoothed_data_path = smooth_csv_file(tmp.name)
    cleaned_data_path = clean_obd_csv(smoothed_data_path)
    preprocessed_df = load_obd_csv(cleaned_data_path)

    require_columns(preprocessed_df, ["time", "speed_kmh", "engine_rpm"])

    preprocessed_df = add_elapsed_time(preprocessed_df)
    preprocessed_df = add_speed_ms(preprocessed_df)
    preprocessed_df = add_acceleration(preprocessed_df)

#    params = VehicleParams(
#        mass_kg=1300, Cd=0.3, area_m2=2.2, crr=0.012,
#        tyre_radius_m=0.318, rho_air=1.17
#    )

    preprocessed_df = add_longitudinal_forces(preprocessed_df, params, grade_rad=0.0)
    preprocessed_df = add_power_terms(preprocessed_df)
    preprocessed_df = add_energy_terms(preprocessed_df)
    
    processed_df = preprocessed_df
    

#==================================
    

    

#=================================

    dt = processed_df["elapsed_time_s"].diff().fillna(0)
    dt = dt.clip(lower=0, upper=2)
    processed_df = add_chemical_efficiency(processed_df)
    
    valid_efficiency = processed_df["chemical_efficiency"].between(0, 0.45)
    valid = (valid_efficiency)
    

    processed_df["E_drive_step_J"] = processed_df["P_drive_W"] * dt
    processed_df["E_fuel_step_J"] = processed_df["Pfuel"] * dt

    window = 1000
    min_period = 100

    processed_df["eff_rolling_energy"] = (processed_df["E_drive_step_J"].where(valid).rolling(window, min_periods=min_period).sum()/ processed_df["E_fuel_step_J"].where(valid).rolling(window, min_periods=min_period).sum())

    processed_df["eff_rolling_energy"] = processed_df["eff_rolling_energy"].where(processed_df["eff_rolling_energy"].between(0, 0.45))

#============================================================

    
    return processed_df