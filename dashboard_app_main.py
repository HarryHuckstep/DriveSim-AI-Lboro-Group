#=====================================================================

# Will's imports

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import Dash, Input, Output, State, callback_context, dcc, html
from dash.exceptions import PreventUpdate
from plotly.subplots import make_subplots

#======================================================================

# Julian's imports

import base64
import io
from src.dashboard_pipeline import prepare_dashboard_df
from src.physics.longitudinal import VehicleParams
from src.physics.power_energy import plot_power, plot_cumulative_energy
from src.physics.longitudinal import plot_longitudinal_forces

#======================================================================

# Ryan's imports

from src.ai.graph_analysis import build_graph_context
from src.ai.graph_payloads import build_payloads_from_demo_dataframe
from src.ai.ibm_granite import GraniteClient

#======================================================================

# Adrian's imports

from driver_classifier import analyse_driver_behaviour
from kit_physical_functions_group_pipeline import add_physical_outputs

#=======================================================================

# Julian's code for CSV parsing and vehicle parameter setup.

def parse_uploaded_csv(contents, filename):
    content_type, content_string = contents.split(",")
    decoded = base64.b64decode(content_string)

    if not filename.lower().endswith(".csv"):
        raise ValueError("Only CSV files are supported.")

    return pd.read_csv(io.StringIO(decoded.decode("utf-8")))

params = VehicleParams(
    mass_kg=1300,
    Cd=0.3,
    area_m2=2.2,
    crr=0.012,
    tyre_radius_m=0.318,
    rho_air=1.17,
)

#=======================================================================

# Adrian's code to prepare the dashboard dataframe for the driver classifier.

def prepare_driver_classifier_df(frame):
    classifier_df = frame.copy()

    rename_map = {}

    if "vehicle_speed" not in classifier_df.columns and "speed_kmh" in classifier_df.columns:
        rename_map["speed_kmh"] = "vehicle_speed"

    if "timestamp" not in classifier_df.columns and "time_s" in classifier_df.columns:
        rename_map["time_s"] = "timestamp"

    if rename_map:
        classifier_df = classifier_df.rename(columns=rename_map)

    return classifier_df


def make_driver_analysis_json_safe(driver_analysis):
    safe_features = {}

    for key, value in driver_analysis["features"].items():
        if isinstance(value, (np.integer, int)):
            safe_features[key] = int(value)
        elif isinstance(value, (np.floating, float)):
            safe_features[key] = float(value)
        else:
            safe_features[key] = value

    return {
        "driver_type": driver_analysis["driver_type"],
        "aggression_score": float(driver_analysis["aggression_score"]),
        "features": safe_features,
    }

#=======================================================================

# Wiil's code for dummy dataset - To be removed once CSV upload and processing is working with real data.

def build_demo_drive():
    dt = 0.25
    time_s = np.arange(0, 300 + dt, dt)

    base_speed = (
        34
        + 24 * np.sin(2 * np.pi * time_s / 180)
        + 10 * np.sin(2 * np.pi * time_s / 42)
        + 4 * np.sin(2 * np.pi * time_s / 13)
    )

    stop_zones = (
        22 * np.exp(-((time_s - 65) / 9) ** 2)
        + 18 * np.exp(-((time_s - 145) / 11) ** 2)
        + 26 * np.exp(-((time_s - 232) / 10) ** 2)
    )

    speed_kmh = np.clip(base_speed - stop_zones, 0, 122)
    speed_gradient = np.gradient(speed_kmh, dt)

    throttle_pct = np.clip(
        18
        + 1.7 * speed_gradient
        + 14 * np.sin(2 * np.pi * time_s / 31)
        + 8 * np.sin(2 * np.pi * time_s / 9),
        0,
        100,
    )

    gear = np.select(
        [
            speed_kmh < 2,
            speed_kmh < 15,
            speed_kmh < 30,
            speed_kmh < 45,
            speed_kmh < 65,
            speed_kmh < 90,
        ],
        [0, 1, 2, 3, 4, 5],
        default=6,
    ).astype(int)

    rpm_per_kmh = {0: 0, 1: 155, 2: 98, 3: 72, 4: 56, 5: 46, 6: 38}
    rpm_gain = np.vectorize(rpm_per_kmh.get)(gear)

    engine_rpm = np.where(
        gear == 0,
        780 + 11 * throttle_pct + 45 * np.sin(2 * np.pi * time_s / 6),
        850 + speed_kmh * rpm_gain + 7 * throttle_pct + 55 * np.sin(2 * np.pi * time_s / 8),
    )

    engine_rpm = np.clip(engine_rpm, 700, 6500)

    ambient_temp_c = np.full_like(time_s, 14.0)

    intake_temp_c = np.clip(
        ambient_temp_c + 3.5 + 0.035 * throttle_pct + 0.0012 * engine_rpm,
        14,
        52,
    )

    coolant_temp_c = np.clip(
        24
        + 70 * (1 - np.exp(-time_s / 85))
        + 0.02 * throttle_pct
        + 1.8 * np.sin(2 * np.pi * time_s / 95),
        20,
        106,
    )

    map_kpa = np.clip(
        24 + 0.55 * throttle_pct + 0.085 * speed_kmh + 2.5 * np.sin(2 * np.pi * time_s / 17),
        18,
        101,
    )

    maf_gps = np.clip(1.8 + 0.0105 * engine_rpm + 0.11 * throttle_pct, 1.5, 180)

    fan_speed_pct = np.piecewise(
        coolant_temp_c,
        [
            coolant_temp_c < 90,
            (coolant_temp_c >= 90) & (coolant_temp_c < 96),
            (coolant_temp_c >= 96) & (coolant_temp_c < 101),
            coolant_temp_c >= 101,
        ],
        [
            0,
            lambda x: 18 + (x - 90) * 9,
            lambda x: 72 + (x - 96) * 5,
            100,
        ],
    )

    fan_speed_pct = np.clip(fan_speed_pct, 0, 100)

    df = pd.DataFrame(
        {
            "time_s": time_s,
            "speed_kmh": speed_kmh,
            "engine_rpm": engine_rpm,
            "gear": gear,
            "coolant_temp_c": coolant_temp_c,
            "fan_speed_pct": fan_speed_pct,
            "throttle_pct": throttle_pct,
            "map_kpa": map_kpa,
            "intake_temp_c": intake_temp_c,
            "ambient_temp_c": ambient_temp_c,
            "maf_gps": maf_gps,
        }
    )

    return df.round(
        {
            "time_s": 2,
            "speed_kmh": 1,
            "engine_rpm": 0,
            "coolant_temp_c": 1,
            "fan_speed_pct": 0,
            "throttle_pct": 1,
            "map_kpa": 1,
            "intake_temp_c": 1,
            "ambient_temp_c": 1,
            "maf_gps": 1,
        }
    )


df = build_demo_drive()

#=======================================================================

# Ryan's code to build graph payloads for Granite analysis based on the demo dataframe - To be replaced with real data once CSV upload and processing is working.

graph_payloads = build_payloads_from_demo_dataframe(df, file_label="demo_drive")

#=======================================================================

# Adrian's code to build initial driver classifier analysis from the demo dataframe.

driver_analysis = make_driver_analysis_json_safe(
    analyse_driver_behaviour(
        prepare_driver_classifier_df(df)
    )
)

#=======================================================================

# Will's code for dashboard setup, styling, and callbacks.

max_index = len(df) - 1
playback_steps = {"1x": 1, "2x": 2, "4x": 4}

app = Dash(__name__)
app.title = "DriveSim AI Dashboard Prototype"

colors = {
    "page": "#0d1117",
    "card": "#161b22",
    "border": "#30363d",
    "text": "#e6edf3",
    "muted": "#8b949e",
    "accent": "#58a6ff",
    "success": "#3fb950",
    "warning": "#d29922",
    "danger": "#f85149",
}


def format_time(seconds):
    minutes = int(seconds // 60)
    remainder = seconds % 60
    return f"{minutes:02d}:{remainder:05.2f}"


def slider_marks(frame):
    markers = [0, int(max_index * 0.25), int(max_index * 0.5), int(max_index * 0.75), max_index]
    return {i: format_time(frame.loc[i, "time_s"]) for i in markers}


def card_style():
    return {
        "backgroundColor": colors["card"],
        "border": f"1px solid {colors['border']}",
        "borderRadius": "18px",
        "padding": "16px",
        "boxShadow": "0 8px 24px rgba(0, 0, 0, 0.18)",
    }


def stat_card(label, value_id, unit_id):
    return html.Div(
        [
            html.Div(label, style={"color": colors["muted"], "fontSize": "13px", "marginBottom": "8px"}),
            html.Div(
                [
                    html.Span(id=value_id, style={"fontSize": "30px", "fontWeight": "700", "marginRight": "8px"}),
                    html.Span(id=unit_id, style={"fontSize": "14px", "color": colors["muted"]}),
                ],
                style={"display": "flex", "alignItems": "baseline"},
            ),
        ],
        style=card_style(),
    )


def make_speedometer(speed_value):
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=float(speed_value),
            number={"suffix": " km/h", "font": {"size": 34}},
            title={"text": "Vehicle speed"},
            gauge={
                "axis": {"range": [0, 140], "tickwidth": 1},
                "bar": {"color": colors["accent"]},
                "bgcolor": "#0d1117",
                "steps": [
                    {"range": [0, 50], "color": "#15202b"},
                    {"range": [50, 100], "color": "#1f2937"},
                    {"range": [100, 140], "color": "#312e1f"},
                ],
                "threshold": {
                    "line": {"color": colors["danger"], "width": 4},
                    "thickness": 0.8,
                    "value": 120,
                },
            },
        )
    )

    fig.update_layout(
        margin=dict(l=20, r=20, t=60, b=20),
        paper_bgcolor=colors["card"],
        font={"color": colors["text"]},
        height=300,
    )

    return fig


def make_rpm_gauge(rpm_value):
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=float(rpm_value),
            number={"suffix": " rpm", "font": {"size": 30}},
            title={"text": "Engine RPM"},
            gauge={
                "axis": {"range": [0, 7000], "tickwidth": 1},
                "bar": {"color": colors["success"]},
                "bgcolor": "#0d1117",
                "steps": [
                    {"range": [0, 3000], "color": "#15202b"},
                    {"range": [3000, 5000], "color": "#1f2937"},
                    {"range": [5000, 7000], "color": "#33211f"},
                ],
                "threshold": {
                    "line": {"color": colors["danger"], "width": 4},
                    "thickness": 0.8,
                    "value": 6000,
                },
            },
        )
    )

    fig.update_layout(
        margin=dict(l=20, r=20, t=60, b=20),
        paper_bgcolor=colors["card"],
        font={"color": colors["text"]},
        height=300,
    )

    return fig


def apply_common_layout(fig, title):
    fig.update_layout(
        title=title,
        paper_bgcolor=colors["card"],
        plot_bgcolor=colors["card"],
        font={"color": colors["text"]},
        margin=dict(l=50, r=40, t=60, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        hovermode="x unified",
        xaxis=dict(gridcolor="#21262d", zeroline=False),
        yaxis=dict(gridcolor="#21262d", zeroline=False),
    )
    return fig


def add_current_time_marker(fig, current_time):
    fig.add_vline(x=current_time, line_width=2, line_dash="dash", line_color=colors["warning"])
    return fig


def make_speed_rpm_plot(frame, idx):
    current_time = frame.loc[idx, "time_s"]

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(
        go.Scatter(
            x=frame["time_s"],
            y=frame["speed_kmh"],
            mode="lines",
            name="Speed",
            line={"width": 3, "color": colors["accent"]},
        ),
        secondary_y=False,
    )

    fig.add_trace(
        go.Scatter(
            x=frame["time_s"],
            y=frame["engine_rpm"],
            mode="lines",
            name="RPM",
            line={"width": 2, "color": colors["success"]},
        ),
        secondary_y=True,
    )

    add_current_time_marker(fig, current_time)
    fig.update_xaxes(title_text="Time (s)")
    fig.update_yaxes(title_text="Speed (km/h)", secondary_y=False)
    fig.update_yaxes(title_text="RPM", secondary_y=True, gridcolor="#21262d")
    apply_common_layout(fig, "Drive trace")

    return fig


def make_temperature_plot(frame, idx):
    current_time = frame.loc[idx, "time_s"]

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=frame["time_s"],
            y=frame["coolant_temp_c"],
            mode="lines",
            name="Coolant",
            line={"width": 3, "color": "#ff7b72"},
        )
    )

    fig.add_trace(
        go.Scatter(
            x=frame["time_s"],
            y=frame["intake_temp_c"],
            mode="lines",
            name="Intake",
            line={"width": 2, "color": "#79c0ff"},
        )
    )

    fig.add_trace(
        go.Scatter(
            x=frame["time_s"],
            y=frame["ambient_temp_c"],
            mode="lines",
            name="Ambient",
            line={"width": 2, "color": "#a5d6ff"},
        )
    )

    add_current_time_marker(fig, current_time)
    fig.update_xaxes(title_text="Time (s)")
    fig.update_yaxes(title_text="Temperature (°C)")
    apply_common_layout(fig, "Temperatures")

    return fig


def make_load_plot(frame, idx):
    current_time = frame.loc[idx, "time_s"]

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(
        go.Scatter(
            x=frame["time_s"],
            y=frame["throttle_pct"],
            mode="lines",
            name="Throttle",
            line={"width": 3, "color": "#c297ff"},
        ),
        secondary_y=False,
    )

    fig.add_trace(
        go.Scatter(
            x=frame["time_s"],
            y=frame["fan_speed_pct"],
            mode="lines",
            name="Fan speed",
            line={"width": 3, "color": "#ffa657"},
        ),
        secondary_y=False,
    )

    fig.add_trace(
        go.Scatter(
            x=frame["time_s"],
            y=frame["map_kpa"],
            mode="lines",
            name="MAP",
            line={"width": 2, "color": "#3fb950"},
        ),
        secondary_y=True,
    )

    add_current_time_marker(fig, current_time)
    fig.update_xaxes(title_text="Time (s)")
    fig.update_yaxes(title_text="Throttle / fan (%)", secondary_y=False)
    fig.update_yaxes(title_text="MAP (kPa)", secondary_y=True, gridcolor="#21262d")
    apply_common_layout(fig, "Driver demand and cooling")

    return fig


#=======================================================================

# Direct integration of KIT group-pipeline physical calculations.
# This keeps the original dashboard pipeline unchanged. These helpers adapt the
# processed dataframe stored by the dashboard into the column names expected by
# kit_physical_functions_group_pipeline.py.

def make_empty_physics_fig(title, message):
    fig = go.Figure()
    fig.add_annotation(
        text=message,
        xref="paper",
        yref="paper",
        x=0.5,
        y=0.5,
        showarrow=False,
        font={"size": 15, "color": colors["muted"]},
    )
    apply_common_layout(fig, title)
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    return fig


def prepare_group_physics_frame(frame):
    """Return a dataframe with the group-standard names used by the KIT file."""
    out = frame.copy()
    out.columns = [str(c).strip() for c in out.columns]

    if "timestamp" not in out.columns:
        for candidate in ["elapsed_time_s", "time_s", "timestep", "time_step", "t_s"]:
            if candidate in out.columns:
                out["timestamp"] = out[candidate]
                break

    if "vehicle_speed" not in out.columns:
        if "speed_kmh" in out.columns:
            out["vehicle_speed"] = out["speed_kmh"]
        elif "speed_ms" in out.columns:
            out["vehicle_speed"] = pd.to_numeric(out["speed_ms"], errors="coerce") * 3.6

    if "speed_kmh" not in out.columns and "vehicle_speed" in out.columns:
        out["speed_kmh"] = out["vehicle_speed"]

    if "speed_ms" not in out.columns and "vehicle_speed" in out.columns:
        out["speed_ms"] = pd.to_numeric(out["vehicle_speed"], errors="coerce") / 3.6

    for col in ["timestamp", "vehicle_speed", "speed_kmh", "speed_ms", "maf_gps", "engine_rpm", "throttle_pct"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")

    return out


def cumulative_trapz_dashboard(y_values, x_values):
    y = np.nan_to_num(np.asarray(y_values, dtype=float), nan=0.0)
    x = np.asarray(x_values, dtype=float)
    result = np.zeros_like(y, dtype=float)
    if len(y) < 2:
        return result
    increments = 0.5 * (y[1:] + y[:-1]) * np.diff(x)
    result[1:] = np.cumsum(increments)
    return result


def calculate_braking_from_processed_frame(frame):
    """Calculate braking force outputs directly from the processed dashboard data."""
    out = prepare_group_physics_frame(frame)

    required = ["timestamp", "speed_ms"]
    missing = [col for col in required if col not in out.columns]
    if missing:
        raise ValueError(f"Missing required columns for braking calculation: {missing}")

    work = out[["timestamp", "speed_ms"]].copy()
    work = work.replace([np.inf, -np.inf], np.nan).dropna()
    work = work.sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)

    if len(work) < 3:
        raise ValueError("Not enough valid time/speed samples for braking calculation.")

    t = work["timestamp"].to_numpy(dtype=float)
    t = t - t[0]
    v = work["speed_ms"].to_numpy(dtype=float)

    acceleration = np.gradient(v, t)
    braking_mask = acceleration < -0.05
    braking_force = np.where(braking_mask, -params.mass_kg * acceleration, 0.0)
    brake_torque = braking_force * params.tyre_radius_m
    brake_power = braking_force * v
    brake_energy_j = cumulative_trapz_dashboard(brake_power, t)

    braking_df = pd.DataFrame(
        {
            "elapsed_time_s": t,
            "speed_ms": v,
            "acceleration_mps2": acceleration,
            "F_brake_masked_N": braking_force,
            "T_brake_Nm": brake_torque,
            "T_front_brake_Nm": brake_torque * 0.70,
            "T_rear_brake_Nm": brake_torque * 0.30,
            "P_brake_W": brake_power,
            "E_brake_kJ": brake_energy_j / 1000.0,
        }
    )

    summary = {
        "n_braking_events": int(np.sum((braking_mask[1:] == True) & (braking_mask[:-1] == False))) if len(braking_mask) > 1 else int(np.any(braking_mask)),
        "peak_brake_force_N": float(np.nanmax(braking_force)) if len(braking_force) else 0.0,
        "peak_brake_torque_Nm": float(np.nanmax(brake_torque)) if len(brake_torque) else 0.0,
        "peak_brake_power_W": float(np.nanmax(brake_power)) if len(brake_power) else 0.0,
        "total_brake_energy_J": float(brake_energy_j[-1]) if len(brake_energy_j) else 0.0,
        "min_acceleration_mps2": float(np.nanmin(acceleration)) if len(acceleration) else 0.0,
    }
    return braking_df, summary


def calculate_kit_from_processed_frame(frame):
    """Run the uploaded KIT physical-functions file on the processed dashboard data."""
    group_df = prepare_group_physics_frame(frame)
    kit_df, fit_output = add_physical_outputs(
        group_df,
        fuel_tank_volume_l=50.0,
        fuel_type="petrol",
    )

    if "elapsed_time_s" not in kit_df.columns:
        kit_df["elapsed_time_s"] = pd.to_numeric(kit_df["timestamp"], errors="coerce")
        kit_df["elapsed_time_s"] = kit_df["elapsed_time_s"] - kit_df["elapsed_time_s"].iloc[0]

    if "fuel_surface_prediction_l_per_100km" in kit_df.columns:
        kit_df["fuel_surface_residual_l_per_100km"] = (
            kit_df["fuel_surface_prediction_l_per_100km"] - kit_df["fuel_consumption_l_per_100km"]
        )

    if "fuel_svr_prediction_l_per_100km" in kit_df.columns:
        kit_df["fuel_svr_residual_l_per_100km"] = (
            kit_df["fuel_svr_prediction_l_per_100km"] - kit_df["fuel_consumption_l_per_100km"]
        )

    metrics = fit_output.get("Metrics", {}) if isinstance(fit_output, dict) else {}
    distance_km = float(pd.to_numeric(kit_df.get("distance_km"), errors="coerce").max()) if "distance_km" in kit_df else 0.0
    fuel_remaining = float(pd.to_numeric(kit_df.get("fuel_remaining_l"), errors="coerce").iloc[-1]) if "fuel_remaining_l" in kit_df else np.nan
    fuel_used = 50.0 - fuel_remaining if np.isfinite(fuel_remaining) else np.nan

    summary = {
        "distance_km": distance_km,
        "fuel_used_l": fuel_used,
        "fuel_remaining_l": fuel_remaining,
        "median_fuel_consumption_l_per_100km": metrics.get("median_fc_conv", np.nan),
        "n_valid_model_samples": metrics.get("n_valid", 0),
        "rmse_conv_vs_surface": metrics.get("rmse_conv_vs_surface", np.nan),
        "rmse_conv_vs_svr": metrics.get("rmse_conv_vs_svr", np.nan),
    }
    return kit_df, summary

def make_braking_force_plot(braking_df):
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=braking_df["elapsed_time_s"],
            y=braking_df["F_brake_masked_N"],
            mode="lines",
            name="Braking force",
            line={"width": 2, "color": colors["danger"]},
        )
    )

    fig.update_xaxes(title_text="Time (s)")
    fig.update_yaxes(title_text="Braking force (N)")

    return apply_common_layout(fig, "Braking force")


def make_brake_torque_plot(braking_df):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=braking_df["elapsed_time_s"], y=braking_df["T_front_brake_Nm"], mode="lines", name="Front brake torque", line={"width": 2, "color": colors["warning"]}))
    fig.add_trace(go.Scatter(x=braking_df["elapsed_time_s"], y=braking_df["T_rear_brake_Nm"], mode="lines", name="Rear brake torque", line={"width": 2, "color": colors["success"]}))
    fig.add_trace(go.Scatter(x=braking_df["elapsed_time_s"], y=braking_df["T_brake_Nm"], mode="lines", name="Total brake torque", line={"width": 2, "color": colors["danger"]}))
    fig.update_xaxes(title_text="Time (s)")
    fig.update_yaxes(title_text="Brake torque (Nm)")
    return apply_common_layout(fig, "Brake torque split")

def make_braking_energy_plot(braking_df):
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=braking_df["elapsed_time_s"],
            y=braking_df["E_brake_kJ"],
            mode="lines",
            name="Cumulative braking energy",
            line={"width": 2, "color": colors["accent"]},
        )
    )

    fig.update_xaxes(title_text="Time (s)")
    fig.update_yaxes(title_text="Braking energy (kJ)")

    return apply_common_layout(fig, "Cumulative braking energy")

def make_braking_power_plot(braking_df):
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=braking_df["elapsed_time_s"],
            y=braking_df["P_brake_W"],
            mode="lines",
            name="Braking power",
            line={"width": 2, "color": colors["danger"]},
        )
    )

    fig.update_xaxes(title_text="Time (s)")
    fig.update_yaxes(title_text="Braking power (W)")

    return apply_common_layout(fig, "Braking power")


def make_kit_fuel_consumption_plot(kit_df):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=kit_df["elapsed_time_s"], y=kit_df["fuel_consumption_l_per_100km"], mode="lines", name="Conventional speed + MAF", line={"width": 2, "color": colors["accent"]}))
    if "fuel_surface_prediction_l_per_100km" in kit_df.columns:
        fig.add_trace(go.Scatter(x=kit_df["elapsed_time_s"], y=kit_df["fuel_surface_prediction_l_per_100km"], mode="lines", name="RPM + throttle surface fit", line={"width": 2, "color": colors["warning"]}))
    if "fuel_svr_prediction_l_per_100km" in kit_df.columns and kit_df["fuel_svr_prediction_l_per_100km"].notna().any():
        fig.add_trace(go.Scatter(x=kit_df["elapsed_time_s"], y=kit_df["fuel_svr_prediction_l_per_100km"], mode="lines", name="SVR prediction", line={"width": 2, "color": colors["success"]}))
    fig.update_xaxes(title_text="Time (s)")
    fig.update_yaxes(title_text="Fuel consumption (L/100 km)")
    return apply_common_layout(fig, "KIT fuel consumption")


def make_kit_fuel_remaining_plot(kit_df):
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Scatter(x=kit_df["elapsed_time_s"], y=kit_df["fuel_remaining_l"], mode="lines", name="Fuel remaining", line={"width": 2, "color": colors["success"]}), secondary_y=False)
    fig.add_trace(go.Scatter(x=kit_df["elapsed_time_s"], y=kit_df["distance_km"], mode="lines", name="Distance", line={"width": 2, "color": colors["accent"]}), secondary_y=True)
    fig.update_xaxes(title_text="Time (s)")
    fig.update_yaxes(title_text="Fuel remaining (L)", secondary_y=False)
    fig.update_yaxes(title_text="Distance (km)", secondary_y=True, gridcolor="#21262d")
    return apply_common_layout(fig, "Fuel remaining and distance")


def make_kit_residual_plot(kit_df):
    fig = go.Figure()
    if "fuel_surface_residual_l_per_100km" in kit_df.columns:
        fig.add_trace(go.Scatter(x=kit_df["elapsed_time_s"], y=kit_df["fuel_surface_residual_l_per_100km"], mode="lines", name="Surface residual", line={"width": 2, "color": colors["warning"]}))
    if "fuel_svr_residual_l_per_100km" in kit_df.columns and kit_df["fuel_svr_residual_l_per_100km"].notna().any():
        fig.add_trace(go.Scatter(x=kit_df["elapsed_time_s"], y=kit_df["fuel_svr_residual_l_per_100km"], mode="lines", name="SVR residual", line={"width": 2, "color": colors["success"]}))
    fig.add_hline(y=0, line_width=1, line_dash="dash", line_color=colors["muted"])
    fig.update_xaxes(title_text="Time (s)")
    fig.update_yaxes(title_text="Prediction residual (L/100 km)")
    return apply_common_layout(fig, "Fuel model residuals")


app.layout = html.Div(
    [
        dcc.Store(id="playback-store", data={"playing": False}),
        dcc.Store(id="processed-data-store"),
        dcc.Store(id="graph-payloads-store", data=graph_payloads),
        dcc.Store(id="driver-analysis-store", data=driver_analysis),
        dcc.Store(id="granite-answer-store"),
        dcc.Interval(id="playback-interval", interval=250, n_intervals=0),

        html.Div(
            [
                html.H1("DriveSim AI", style={"margin": "0", "fontSize": "34px"}),
                html.P(
                    "Dash prototype for replaying a completed drive with live indicators, evolving plots, CSV upload, and Granite graph analysis.",
                    style={"margin": "8px 0 0 0", "color": colors["muted"]},
                ),
            ],
            style={"marginBottom": "18px"},
        ),

#=======================================================================

# Julian's code for CSV upload component and status message display.

        dcc.Upload(
            id="upload-data",
            children=html.Div(["Drag and drop or ", html.A("select a CSV file")]),
            style={
                "width": "100%",
                "height": "80px",
                "lineHeight": "80px",
                "borderWidth": "1px",
                "borderStyle": "dashed",
                "borderRadius": "12px",
                "textAlign": "center",
                "marginBottom": "18px",
                "color": colors["text"],
                "borderColor": colors["border"],
                "backgroundColor": colors["card"],
            },
            multiple=False,
        ),

        html.Div(
            id="upload-status",
            style={"marginBottom": "14px", "color": colors["muted"]},
        ),

#=======================================================================

# Will's code for playback controls, timeline label, and main dashboard indicators and plots.

        html.Div(
            [
                html.Button(
                    "Play",
                    id="play-button",
                    n_clicks=0,
                    style={
                        "backgroundColor": colors["accent"],
                        "color": "#0d1117",
                        "border": "none",
                        "borderRadius": "12px",
                        "padding": "12px 18px",
                        "fontWeight": "700",
                        "cursor": "pointer",
                    },
                ),

                html.Button(
                    "Reset",
                    id="reset-button",
                    n_clicks=0,
                    style={
                        "backgroundColor": colors["card"],
                        "color": colors["text"],
                        "border": f"1px solid {colors['border']}",
                        "borderRadius": "12px",
                        "padding": "12px 18px",
                        "fontWeight": "700",
                        "cursor": "pointer",
                    },
                ),

                html.Div(
                    [
                        html.Div("Playback", style={"fontSize": "12px", "color": colors["muted"], "marginBottom": "6px"}),
                        dcc.Dropdown(
                            id="playback-rate",
                            options=[{"label": label, "value": value} for label, value in playback_steps.items()],
                            value=1,
                            clearable=False,
                            searchable=False,
                            style={"width": "120px", "color": "#111111"},
                        ),
                    ]
                ),

                html.Div(
                    id="timeline-label",
                    style={
                        "marginLeft": "auto",
                        "fontWeight": "600",
                        "fontSize": "15px",
                        "color": colors["text"],
                    },
                ),
            ],
            style={
                "display": "flex",
                "gap": "12px",
                "alignItems": "center",
                "marginBottom": "14px",
                **card_style(),
            },
        ),

        html.Div(
            [
                dcc.Slider(
                    id="time-slider",
                    min=0,
                    max=max_index,
                    step=1,
                    value=0,
                    marks=slider_marks(df),
                    updatemode="drag",
                    tooltip={"placement": "bottom", "always_visible": False},
                )
            ],
            style={"marginBottom": "18px", **card_style()},
        ),

        html.Div(
            [
                stat_card("Current gear", "gear-value", "gear-unit"),
                stat_card("Coolant temp", "coolant-value", "coolant-unit"),
                stat_card("Fan speed", "fan-value", "fan-unit"),
                stat_card("Throttle", "throttle-value", "throttle-unit"),
                stat_card("MAP", "map-value", "map-unit"),
                stat_card("MAF", "maf-value", "maf-unit"),
            ],
            style={
                "display": "grid",
                "gridTemplateColumns": "repeat(auto-fit, minmax(180px, 1fr))",
                "gap": "14px",
                "marginBottom": "18px",
            },
        ),

        html.Div(
            [
                html.Div(
                    [dcc.Graph(id="speedometer-graph", config={"displayModeBar": False})],
                    style=card_style(),
                ),
                html.Div(
                    [dcc.Graph(id="rpm-gauge-graph", config={"displayModeBar": False})],
                    style=card_style(),
                ),
            ],
            style={
                "display": "grid",
                "gridTemplateColumns": "repeat(auto-fit, minmax(320px, 1fr))",
                "gap": "14px",
                "marginBottom": "18px",
            },
        ),

        html.Div(
            [
                html.Div([dcc.Graph(id="speed-rpm-plot")], style=card_style()),
                html.Div([dcc.Graph(id="temperature-plot")], style=card_style()),
                html.Div([dcc.Graph(id="load-plot")], style=card_style()),
               
                #=======================================================================
                
                # Julian's code to add power, force, and energy plots based on the uploaded CSV data.

                html.Div([dcc.Graph(id="power-plot")], style=card_style()),
                html.Div([dcc.Graph(id="force-plot")], style=card_style()),
                html.Div([dcc.Graph(id="energy-plot")], style=card_style()),

                #=======================================================================
            ],
            style={
                "display": "grid",
                "gridTemplateColumns": "1fr",
                "gap": "14px",
                "marginBottom": "18px", # Added by Ryan. This margin will create space between the plots and the Granite analysis section below. Added by Ryan.
            },
        ),



#=======================================================================

# Direct KIT physical-functions / braking integration section.

        html.Div(
            [
                html.H2("Braking force analysis", style={"marginTop": "0", "marginBottom": "10px"}),
                html.Div(
                    id="braking-physics-summary",
                    style={
                        "whiteSpace": "pre-wrap",
                        "lineHeight": "1.5",
                        "padding": "14px",
                        "borderRadius": "12px",
                        "border": f"1px solid {colors['border']}",
                        "backgroundColor": "#0d1117",
                        "marginBottom": "14px",
                    },
                ),
                dcc.Graph(id="braking-force-plot"),
                dcc.Graph(id="brake-torque-plot"),
                dcc.Graph(id="braking-energy-plot"),
                dcc.Graph(id="braking-power-plot"),
            ],
            style={**card_style(), "marginBottom": "18px"},
        ),

        html.Div(
            [
                html.H2("KIT fuel analysis", style={"marginTop": "0", "marginBottom": "10px"}),
                html.Div(
                    id="kit-physics-summary",
                    style={
                        "whiteSpace": "pre-wrap",
                        "lineHeight": "1.5",
                        "padding": "14px",
                        "borderRadius": "12px",
                        "border": f"1px solid {colors['border']}",
                        "backgroundColor": "#0d1117",
                        "marginBottom": "14px",
                    },
                ),
                dcc.Graph(id="kit-fuel-consumption-plot"),
                dcc.Graph(id="kit-fuel-remaining-plot"),
                dcc.Graph(id="kit-fuel-residual-plot"),
            ],
            style={**card_style(), "marginBottom": "18px"},
        ),

#=======================================================================

# Ryan's code to build the Granite graph analysis section of the dashboard, including the graph selector, question input, and response display.
        html.Div(
            [
                html.H2("IBM Granite graph analysis", style={"marginTop": "0", "marginBottom": "14px"}),

                html.Div(
                    [
                        html.Div(
                            [
                                html.Div(
                                    "Graph",
                                    style={
                                        "fontSize": "12px",
                                        "color": colors["muted"],
                                        "marginBottom": "6px",
                                    },
                                ),
                                dcc.Dropdown(
                                    id="graph-selector",
                                    options=[
                                        {"label": payload["graph_name"], "value": payload["graph_id"]}
                                        for payload in graph_payloads
                                    ],
                                    value=graph_payloads[0]["graph_id"] if graph_payloads else None,
                                    clearable=False,
                                    searchable=True,
                                    style={"color": "#111111"},
                                ),
                            ],
                            style={"flex": "1"},
                        ),

                        html.Div(
                            [
                                html.Div(
                                    "Focus x-value (optional)",
                                    style={
                                        "fontSize": "12px",
                                        "color": colors["muted"],
                                        "marginBottom": "6px",
                                    },
                                ),
                                dcc.Input(
                                    id="focus-x",
                                    type="number",
                                    placeholder="e.g. 120",
                                    style={
                                        "width": "180px",
                                        "padding": "10px",
                                        "borderRadius": "10px",
                                        "border": f"1px solid {colors['border']}",
                                    },
                                ),
                            ]
                        ),
                    ],
                    style={
                        "display": "flex",
                        "gap": "12px",
                        "marginBottom": "12px",
                    },
                ),

                dcc.Textarea(
                    id="granite-question",
                    placeholder="Ask a question about the selected graph...",
                    style={
                        "width": "100%",
                        "height": "110px",
                        "padding": "12px",
                        "borderRadius": "12px",
                        "border": f"1px solid {colors['border']}",
                        "marginBottom": "12px",
                        "color": "#000000",
                        "backgroundColor": "#ffffff",
                    },
                ),

                html.Button(
                    "Analyse with Granite",
                    id="granite-button",
                    n_clicks=0,
                    style={
                        "backgroundColor": colors["accent"],
                        "color": "#0d1117",
                        "border": "none",
                        "borderRadius": "12px",
                        "padding": "12px 18px",
                        "fontWeight": "700",
                        "cursor": "pointer",
                        "marginBottom": "12px",
                    },
                ),

                html.Div(
                    id="granite-response",
                    style={
                        "whiteSpace": "pre-wrap",
                        "lineHeight": "1.5",
                        "padding": "14px",
                        "borderRadius": "12px",
                        "border": f"1px solid {colors['border']}",
                        "backgroundColor": "#0d1117",
                    },
                ),

                html.Div(
                    [
                        html.Button(
                            "Read answer aloud",
                            id="tts-read-button",
                            n_clicks=0,
                            style={
                                "backgroundColor": colors["success"],
                                "color": "#0d1117",
                                "border": "none",
                                "borderRadius": "12px",
                                "padding": "12px 18px",
                                "fontWeight": "700",
                                "cursor": "pointer",
                                "marginTop": "12px",
                                "marginRight": "8px",
                            },
                        ),

                        html.Button(
                            "Stop reading",
                            id="tts-stop-button",
                            n_clicks=0,
                            style={
                                "backgroundColor": colors["danger"],
                                "color": "#ffffff",
                                "border": "none",
                                "borderRadius": "12px",
                                "padding": "12px 18px",
                                "fontWeight": "700",
                                "cursor": "pointer",
                                "marginTop": "12px",
                            },
                        ),
                    ]
                ),

                html.Div(
                    id="tts-status",
                    style={
                        "marginTop": "8px",
                        "color": colors["muted"],
                        "fontSize": "13px",
                    },
                ),
            ],
            style=card_style(),
        ),

#=======================================================================

# Adrian's code to display the driver classifier results at the end of the dashboard.

        html.Div(
            [
                html.H2("Driver behaviour classifier", style={"marginTop": "0", "marginBottom": "14px"}),

                html.Div(
                    [
                        stat_card("Driver type", "driver-type-value", "driver-type-unit"),
                        stat_card("Aggression score", "aggression-score-value", "aggression-score-unit"),
                        stat_card("Hard brakes", "hard-brakes-value", "hard-brakes-unit"),
                        stat_card("Throttle spikes", "throttle-spikes-value", "throttle-spikes-unit"),
                    ],
                    style={
                        "display": "grid",
                        "gridTemplateColumns": "repeat(auto-fit, minmax(180px, 1fr))",
                        "gap": "14px",
                        "marginBottom": "14px",
                    },
                ),

                html.Div(
                    id="driver-classifier-summary",
                    style={
                        "whiteSpace": "pre-wrap",
                        "lineHeight": "1.5",
                        "padding": "14px",
                        "borderRadius": "12px",
                        "border": f"1px solid {colors['border']}",
                        "backgroundColor": "#0d1117",
                    },
                ),
            ],
            style=card_style(),
        ),
    ],
    style={
        "backgroundColor": colors["page"],
        "color": colors["text"],
        "minHeight": "100vh",
        "padding": "24px",
        "fontFamily": "Inter, Arial, sans-serif",
    },
)
#========================================================================

# Julian's code to process the uploaded CSV file and prepare the data for plotting.

# Updated by Ryan to also update the graph payloads store with new payloads generated from the uploaded CSV data, and to display a status message about the upload and processing results.

@app.callback(
    Output("processed-data-store", "data"),
    Output("graph-payloads-store", "data"),
    Output("driver-analysis-store", "data"),
    Output("upload-status", "children"),
    Input("upload-data", "contents"),
    State("upload-data", "filename"),
    prevent_initial_call=True,
)
def process_uploaded_file(contents, filename):
    if contents is None:
        raise PreventUpdate

    try:
        raw_df = parse_uploaded_csv(contents, filename)
        processed_df = prepare_dashboard_df(raw_df, params)

        uploaded_payloads = build_payloads_from_demo_dataframe(
            processed_df,
            file_label=filename,
        )

        classifier_df = prepare_driver_classifier_df(processed_df)
        driver_analysis = analyse_driver_behaviour(classifier_df)
        driver_analysis = make_driver_analysis_json_safe(driver_analysis)

        print(processed_df.columns.tolist()) # Tempoary debug line.

        return (
            processed_df.to_dict("records"),
            uploaded_payloads,
            driver_analysis,
            f"Loaded {filename} with {len(processed_df)} rows. Granite graph analysis and driver behaviour analysis have been updated for this file.",
        )

    except Exception as e:
        return None, graph_payloads, None, f"Upload failed: {e}"
    
#=======================================================================

# Will's code to update the graph selector options and default value based on the graph payloads generated from the uploaded CSV data. This ensures that the Granite analysis section always has the correct graphs available for selection based on the most recently uploaded data.

@app.callback(
    Output("graph-selector", "options"),
    Output("graph-selector", "value"),
    Input("graph-payloads-store", "data"),
)
def update_graph_selector(payloads):
    if not payloads:
        return [], None

    options = [
        {"label": payload["graph_name"], "value": payload["graph_id"]}
        for payload in payloads
    ]

    return options, payloads[0]["graph_id"]

#=======================================================================

# Julian's lines corresponding to plotting power.

@app.callback(
    Output("power-plot", "figure"),
    Input("processed-data-store", "data"),
)
def update_power_plot(stored_data):
    if stored_data is None:
        raise PreventUpdate

    uploaded_df = pd.DataFrame(stored_data)
    return plot_power(uploaded_df)

#========================================================================

# Julian's lines corresponding to plotting longitudinal forces.
@app.callback(
    Output("force-plot", "figure"),
    Input("processed-data-store", "data"),
)
def update_force_plot(stored_data):
    if stored_data is None:
        raise PreventUpdate

    uploaded_df = pd.DataFrame(stored_data)
    return plot_longitudinal_forces(uploaded_df)

#=========================================================================

# Julian's lines corresponding to plotting cumulative energy.

@app.callback(
    Output("energy-plot", "figure"),
    Input("processed-data-store", "data"),
)
def update_energy_plot(stored_data):
    if stored_data is None:
        raise PreventUpdate

    uploaded_df = pd.DataFrame(stored_data)
    return plot_cumulative_energy(uploaded_df)



#=======================================================================

# Direct KIT physical-functions / braking integration callbacks.
# These callbacks only read processed-data-store and do not alter the existing
# power, force, or cumulative-energy callbacks.

@app.callback(
    Output("braking-force-plot", "figure"),
    Output("brake-torque-plot", "figure"),
    Output("braking-energy-plot", "figure"),
    Output("braking-power-plot", "figure"),
    Output("braking-physics-summary", "children"),
    Input("processed-data-store", "data"),
)
def update_braking_physics_outputs(stored_data):
    if stored_data is None:
        message = "Upload a CSV file to run the braking force analysis."
        return (
            make_empty_physics_fig("Braking force", message),
            make_empty_physics_fig("Brake torque split", message),
            make_empty_physics_fig("Cumulative braking energy", message),
            make_empty_physics_fig("Braking power", message),
            message,
        )

    try:
        uploaded_df = pd.DataFrame(stored_data)
        braking_df, braking_summary = calculate_braking_from_processed_frame(uploaded_df)
        summary_text = (
            f"Braking events detected: {braking_summary.get('n_braking_events', 0)}\n"
            f"Peak braking force: {braking_summary.get('peak_brake_force_N', 0):.0f} N\n"
            f"Peak brake torque: {braking_summary.get('peak_brake_torque_Nm', 0):.0f} Nm\n"
            f"Peak braking power: {braking_summary.get('peak_brake_power_W', 0) / 1000:.2f} kW\n"
            f"Total braking energy: {braking_summary.get('total_brake_energy_J', 0) / 1000:.2f} kJ\n"
            f"Maximum braking acceleration: {braking_summary.get('min_acceleration_mps2', 0):.2f} m/s²"
        )
        return (
            make_braking_force_plot(braking_df),
            make_brake_torque_plot(braking_df),
            make_braking_power_plot(braking_df),
            make_braking_energy_plot(braking_df),
            summary_text,
        )
    except Exception as e:
        message = f"Braking force analysis failed: {e}"
        return (
            make_empty_physics_fig("Braking force and acceleration", message),
            make_empty_physics_fig("Brake torque split", message),
            make_empty_physics_fig("Braking power and energy", message),
            message,
        )


@app.callback(
    Output("kit-fuel-consumption-plot", "figure"),
    Output("kit-fuel-remaining-plot", "figure"),
    Output("kit-fuel-residual-plot", "figure"),
    Output("kit-physics-summary", "children"),
    Input("processed-data-store", "data"),
)
def update_kit_physics_outputs(stored_data):
    if stored_data is None:
        message = "Upload a CSV file to run the KIT fuel analysis."
        return (
            make_empty_physics_fig("KIT fuel consumption", message),
            make_empty_physics_fig("Fuel remaining and distance", message),
            make_empty_physics_fig("Fuel model residuals", message),
            message,
        )

    try:
        uploaded_df = pd.DataFrame(stored_data)
        kit_df, kit_summary = calculate_kit_from_processed_frame(uploaded_df)
        surface_rmse = kit_summary.get("rmse_conv_vs_surface", np.nan)
        surface_rmse_text = "not available" if pd.isna(surface_rmse) else f"{surface_rmse:.2f} L/100 km"
        summary_text = (
            f"Distance covered: {kit_summary.get('distance_km', 0):.2f} km\n"
            f"Fuel used: {kit_summary.get('fuel_used_l', 0):.3f} L\n"
            f"Fuel remaining estimate: {kit_summary.get('fuel_remaining_l', 0):.2f} L\n"
            f"Median conventional fuel consumption: {kit_summary.get('median_fuel_consumption_l_per_100km', 0):.2f} L/100 km\n"
            f"Valid model samples: {kit_summary.get('n_valid_model_samples', 0)}\n"
            f"Surface model RMSE: {surface_rmse_text}"
        )
        return (
            make_kit_fuel_consumption_plot(kit_df),
            make_kit_fuel_remaining_plot(kit_df),
            make_kit_residual_plot(kit_df),
            summary_text,
        )
    except Exception as e:
        message = f"KIT fuel analysis failed: {e}"
        return (
            make_empty_physics_fig("KIT fuel consumption", message),
            make_empty_physics_fig("Fuel remaining and distance", message),
            make_empty_physics_fig("Fuel model residuals", message),
            message,
        )

#=======================================================================

# Will's code to handle playback controls, timeline advancement, and dashboard refresh based on the current time index.

@app.callback(
    Output("playback-store", "data"),
    Output("play-button", "children"),
    Input("play-button", "n_clicks"),
    State("playback-store", "data"),
    prevent_initial_call=True,
)
def toggle_playback(_, playback_state):
    playing = not playback_state["playing"]
    return {"playing": playing}, ("Pause" if playing else "Play")


@app.callback(
    Output("time-slider", "value"),
    Input("playback-interval", "n_intervals"),
    Input("reset-button", "n_clicks"),
    State("playback-store", "data"),
    State("playback-rate", "value"),
    State("time-slider", "value"),
)
def advance_timeline(_, __, playback_state, playback_rate, current_value):
    trigger = callback_context.triggered[0]["prop_id"] if callback_context.triggered else ""

    if trigger == "reset-button.n_clicks":
        return 0

    if not playback_state["playing"]:
        raise PreventUpdate

    next_value = current_value + playback_rate

    if next_value > max_index:
        return max_index

    return next_value


@app.callback(
    Output("timeline-label", "children"),
    Output("gear-value", "children"),
    Output("gear-unit", "children"),
    Output("coolant-value", "children"),
    Output("coolant-unit", "children"),
    Output("fan-value", "children"),
    Output("fan-unit", "children"),
    Output("throttle-value", "children"),
    Output("throttle-unit", "children"),
    Output("map-value", "children"),
    Output("map-unit", "children"),
    Output("maf-value", "children"),
    Output("maf-unit", "children"),
    Output("speedometer-graph", "figure"),
    Output("rpm-gauge-graph", "figure"),
    Output("speed-rpm-plot", "figure"),
    Output("temperature-plot", "figure"),
    Output("load-plot", "figure"),
    Input("time-slider", "value"),
)
def refresh_dashboard(index):
    row = df.iloc[index]
    timeline_text = f"Elapsed {format_time(row['time_s'])} / {format_time(df.iloc[-1]['time_s'])}"
    gear_display = "N" if int(row["gear"]) == 0 else str(int(row["gear"]))

    return (
        timeline_text,
        gear_display,
        "",
        f"{row['coolant_temp_c']:.1f}",
        "°C",
        f"{row['fan_speed_pct']:.0f}",
        "%",
        f"{row['throttle_pct']:.1f}",
        "%",
        f"{row['map_kpa']:.1f}",
        "kPa",
        f"{row['maf_gps']:.1f}",
        "g/s",
        make_speedometer(row["speed_kmh"]),
        make_rpm_gauge(row["engine_rpm"]),
        make_speed_rpm_plot(df, index),
        make_temperature_plot(df, index),
        make_load_plot(df, index),
    )

#=======================================================================

# Ryan's code to handle the Granite graph analysis when the "Analyse with Granite" button is clicked, including extracting the selected graph payload, building the graph context, sending the question to Granite, and displaying the response.

@app.callback(
    Output("granite-response", "children"),
    Output("granite-answer-store", "data"),
    Input("granite-button", "n_clicks"),
    State("graph-selector", "value"),
    State("granite-question", "value"),
    State("focus-x", "value"),
    State("graph-payloads-store", "data"),
    prevent_initial_call=True,
)
def analyse_with_granite(_, graph_id, question, focus_x, payloads):
    if not graph_id or not question:
        raise PreventUpdate

    selected_payload = None

    for payload in payloads:
        if payload["graph_id"] == graph_id:
            selected_payload = payload
            break

    if selected_payload is None:
        message = "Selected graph payload was not found."
        return message, message

    graph_context = build_graph_context(
        graph_payload=selected_payload,
        user_question=question,
        selected_x=focus_x,
    )

    client = GraniteClient()
    answer = client.answer_graph_question(graph_context)

    if answer:
        return answer, answer

    message = "No response received from Granite."
    return message, message

#=======================================================================

# Adrian's code to update the driver behaviour classifier display using the latest dataframe already processed by the dashboard.

@app.callback(
    Output("driver-type-value", "children"),
    Output("driver-type-unit", "children"),
    Output("aggression-score-value", "children"),
    Output("aggression-score-unit", "children"),
    Output("hard-brakes-value", "children"),
    Output("hard-brakes-unit", "children"),
    Output("throttle-spikes-value", "children"),
    Output("throttle-spikes-unit", "children"),
    Output("driver-classifier-summary", "children"),
    Input("driver-analysis-store", "data"),
)
def update_driver_classifier_display(driver_analysis):
    if not driver_analysis:
        return (
            "No data",
            "",
            "--",
            "",
            "--",
            "events",
            "--",
            "events",
            "Upload a CSV file to run the driver behaviour classifier.",
        )

    features = driver_analysis.get("features", {})

    driver_type = driver_analysis.get("driver_type", "Unknown")
    aggression_score = driver_analysis.get("aggression_score", 0)
    hard_brake_count = features.get("hard_brake_count", 0)
    throttle_spike_count = features.get("throttle_spike_count", 0)

    summary = (
        f"Average speed: {features.get('avg_speed', 0):.1f} km/h\n"
        f"Maximum speed: {features.get('max_speed', 0):.1f} km/h\n"
        f"Maximum acceleration: {features.get('max_accel', 0):.2f} m/s²\n"
        f"Maximum braking: {features.get('max_brake', 0):.2f} m/s²\n"
        f"Aggressive acceleration events: {features.get('aggressive_accel_count', 0)}\n"
        f"Average RPM: {features.get('avg_rpm', 0):.0f} rpm\n"
    )

    return (
        driver_type,
        "",
        f"{aggression_score:.2f}",
        "",
        f"{hard_brake_count}",
        "events",
        f"{throttle_spike_count}",
        "events",
        summary,
    )

#=======================================================================
app.clientside_callback(
    """
    function(readClicks, stopClicks, answerText) {
        const triggered = dash_clientside.callback_context.triggered;

        if (!triggered || triggered.length === 0) {
            return "";
        }

        const triggerId = triggered[0].prop_id;

        if (triggerId === "tts-stop-button.n_clicks") {
            window.speechSynthesis.cancel();
            return "Stopped reading.";
        }

        if (triggerId === "tts-read-button.n_clicks") {
            if (!answerText) {
                return "No Granite answer available to read.";
            }

            window.speechSynthesis.cancel();

            const utterance = new SpeechSynthesisUtterance(answerText);
            utterance.rate = 1.0;
            utterance.pitch = 1.0;
            utterance.volume = 1.0;

            window.speechSynthesis.speak(utterance);

            return "Reading Granite answer aloud.";
        }

        return "";
    }
    """,
    Output("tts-status", "children"),
    Input("tts-read-button", "n_clicks"),
    Input("tts-stop-button", "n_clicks"),
    State("granite-answer-store", "data"),
    prevent_initial_call=True,
)

if __name__ == "__main__":
    app.run(debug=True)