import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import Dash, Input, Output, State, callback_context, dcc, html
from dash.exceptions import PreventUpdate
from plotly.subplots import make_subplots

#Julian import
#==============================================================
import base64
import io
from src.dashboard_pipeline import prepare_dashboard_df
from src.physics.longitudinal import VehicleParams
from src.physics.power_energy import plot_power, plot_cumulative_energy
from src.physics.longitudinal import plot_longitudinal_forces
#===============================================================



#JULIANS SECTION
#=============================================================================================================#


'''
#=== code for giving user option to plug in their own parameters=====

html.Div(
    [
        html.Div("Vehicle parameters", style={"fontWeight": "700", "marginBottom": "10px"}),

        html.Div(
            [
                html.Div([
                    html.Label("Mass (kg)", style={"color": colors["muted"]}),
                    dcc.Input(id="mass-kg", type="number", value=1300, debounce=True,
                              style={"width": "100%", "padding": "8px", "borderRadius": "8px"})
                ]),
                html.Div([
                    html.Label("Drag coefficient Cd", style={"color": colors["muted"]}),
                    dcc.Input(id="cd", type="number", value=0.30, debounce=True,
                              style={"width": "100%", "padding": "8px", "borderRadius": "8px"})
                ]),
                html.Div([
                    html.Label("Frontal area (m²)", style={"color": colors["muted"]}),
                    dcc.Input(id="area-m2", type="number", value=2.2, debounce=True,
                              style={"width": "100%", "padding": "8px", "borderRadius": "8px"})
                ]),
                html.Div([
                    html.Label("Rolling resistance crr", style={"color": colors["muted"]}),
                    dcc.Input(id="crr", type="number", value=0.012, debounce=True,
                              style={"width": "100%", "padding": "8px", "borderRadius": "8px"})
                ]),
                html.Div([
                    html.Label("Tyre radius (m)", style={"color": colors["muted"]}),
                    dcc.Input(id="tyre-radius-m", type="number", value=0.318, debounce=True,
                              style={"width": "100%", "padding": "8px", "borderRadius": "8px"})
                ]),
                html.Div([
                    html.Label("Air density (kg/m³)", style={"color": colors["muted"]}),
                    dcc.Input(id="rho-air", type="number", value=1.17, debounce=True,
                              style={"width": "100%", "padding": "8px", "borderRadius": "8px"})
                ]),
            ],
            style={
                "display": "grid",
                "gridTemplateColumns": "repeat(auto-fit, minmax(180px, 1fr))",
                "gap": "12px",
            },
        ),
    ],
    style={**card_style(), "marginBottom": "18px"},
)

#============================================================
'''

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






























#code for dummy dataset 
#=============================================================================================================#



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



#=================================================================#

app.layout = html.Div(
    [
        dcc.Store(id="playback-store", data={"playing": False}),
        dcc.Store(id="processed-data-store"), #new line
        dcc.Interval(id="playback-interval", interval=250, n_intervals=0),
        html.Div(
            [
                html.H1("DriveSim AI", style={"margin": "0", "fontSize": "34px"}),
                html.P(
                    "Dash prototype for replaying a completed drive with live indicators and evolving plots.",
                    style={"margin": "8px 0 0 0", "color": colors["muted"]},
                ),
            ],
            style={"marginBottom": "18px"},
        ),
        
        
        
        
        
        
        
        
        
        
      
#===== NEW lINES
        
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
        
 #=====
        
        
        
        
        
        
        
        
        
        
        

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
                html.Div([dcc.Graph(id="power-plot")], style=card_style()), #new line
                html.Div([dcc.Graph(id="force-plot")], style=card_style()), #new line
                html.Div([dcc.Graph(id="energy-plot")], style=card_style()), #new line
            ],
            style={
                "display": "grid",
                "gridTemplateColumns": "1fr",
                "gap": "14px",
            },
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



#=== NEW lines corresponding to the uploading CSV file
        
@app.callback(
    Output("processed-data-store", "data"),
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

        print(processed_df.columns.tolist())  # temporary debug line

        return (
            processed_df.to_dict("records"),
            f"Loaded {filename} with {len(processed_df)} rows.",
        )

    except Exception as e:
        return None, f"Upload failed: {e}"        



#===



#===== New lines corresponding to plotting power
@app.callback(
    Output("power-plot", "figure"),
    Input("processed-data-store", "data"),
)
def update_power_plot(stored_data):
    if stored_data is None:
        raise PreventUpdate

    df = pd.DataFrame(stored_data)
    return plot_power(df)
#==============


#==================
@app.callback(
    Output("force-plot", "figure"),
    Input("processed-data-store", "data"),
)
def update_force_plot(stored_data):
    if stored_data is None:
        raise PreventUpdate

    df = pd.DataFrame(stored_data)
    return plot_longitudinal_forces(df)
#=================


#=================
@app.callback(
    Output("energy-plot", "figure"),
    Input("processed-data-store", "data"),
)
def update_energy_plot(stored_data):
    if stored_data is None:
        raise PreventUpdate

    df = pd.DataFrame(stored_data)
    return plot_cumulative_energy(df)

#===============















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


if __name__ == "__main__":
    app.run(debug=True)
