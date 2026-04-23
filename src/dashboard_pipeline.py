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
    
    return processed_df