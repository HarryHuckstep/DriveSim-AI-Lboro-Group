#!/usr/bin/env python3

import pandas as pd


def get_numeric_series(df, column):
    if column not in df.columns:
        return None
    return pd.to_numeric(df[column], errors="coerce")


def clean_xy(x_series, y_series):
    valid = x_series.notna() & y_series.notna()
    return x_series[valid].tolist(), y_series[valid].tolist()


def build_line_graph_payload(df, graph_id, graph_name, y_column, x_column="time_s", x_axis_name="time (s)"):
    x = get_numeric_series(df, x_column)
    y = get_numeric_series(df, y_column)

    if x is None or y is None:
        return None

    x_values, y_values = clean_xy(x, y)

    if not x_values or not y_values:
        return None

    return {
        "graph_id": graph_id,
        "graph_name": graph_name,
        "graph_type": "line",
        "x_axis_name": x_axis_name,
        "y_axis_name": y_column,
        "source_x_column": x_column,
        "source_y_column": y_column,
        "x_values": x_values,
        "y_values": y_values,
    }


def build_xy_graph_payload(df, graph_id, graph_name, x_column, y_column):
    x = get_numeric_series(df, x_column)
    y = get_numeric_series(df, y_column)

    if x is None or y is None:
        return None

    x_values, y_values = clean_xy(x, y)

    if not x_values or not y_values:
        return None

    return {
        "graph_id": graph_id,
        "graph_name": graph_name,
        "graph_type": "xy",
        "x_axis_name": x_column,
        "y_axis_name": y_column,
        "source_x_column": x_column,
        "source_y_column": y_column,
        "x_values": x_values,
        "y_values": y_values,
    }


def build_payloads_from_demo_dataframe(df, file_label="demo_drive"):
    payloads = []

    time_series_columns = [
        "speed_kmh",
        "engine_rpm",
        "gear",
        "coolant_temp_c",
        "fan_speed_pct",
        "throttle_pct",
        "map_kpa",
        "intake_temp_c",
        "ambient_temp_c",
        "maf_gps",
    ]

    for column in time_series_columns:
        if column in df.columns:
            payload = build_line_graph_payload(
                df=df,
                graph_id="{}:{}_vs_time".format(file_label, column),
                graph_name="{}: {} vs time".format(file_label, column),
                y_column=column,
                x_column="time_s",
                x_axis_name="time (s)",
            )
            if payload:
                payloads.append(payload)

    xy_pairs = [
        ("speed_kmh", "engine_rpm"),
        ("speed_kmh", "throttle_pct"),
        ("speed_kmh", "map_kpa"),
        ("engine_rpm", "maf_gps"),
        ("coolant_temp_c", "fan_speed_pct"),
        ("throttle_pct", "engine_rpm"),
        ("throttle_pct", "speed_kmh"),
        ("map_kpa", "engine_rpm"),
        ("intake_temp_c", "engine_rpm"),
    ]

    for x_column, y_column in xy_pairs:
        if x_column in df.columns and y_column in df.columns:
            payload = build_xy_graph_payload(
                df=df,
                graph_id="{}:{}_vs_{}".format(file_label, y_column, x_column),
                graph_name="{}: {} vs {}".format(file_label, y_column, x_column),
                x_column=x_column,
                y_column=y_column,
            )
            if payload:
                payloads.append(payload)

    return payloads
