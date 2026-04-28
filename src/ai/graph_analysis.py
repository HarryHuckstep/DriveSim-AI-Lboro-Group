#!/usr/bin/env python3

import math


def is_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def numeric_values(values):
    return [v for v in values if is_number(v)]


def safe_min(values):
    nums = numeric_values(values)
    return min(nums) if nums else None


def safe_max(values):
    nums = numeric_values(values)
    return max(nums) if nums else None


def safe_avg(values):
    nums = numeric_values(values)
    return float(sum(nums)) / len(nums) if nums else None


def safe_range(values):
    nums = numeric_values(values)
    if not nums:
        return None
    return max(nums) - min(nums)


def basic_trend(y_values):
    if len(y_values) < 2:
        return "unknown"

    first = y_values[0]
    last = y_values[-1]

    if not is_number(first) or not is_number(last):
        return "unknown"

    delta = last - first

    if abs(delta) < 1e-9:
        return "roughly flat"
    if delta > 0:
        return "overall increasing"
    return "overall decreasing"


def detect_spikes(x_values, y_values, threshold_ratio=2.0):
    spikes = []

    if len(x_values) != len(y_values) or len(y_values) < 3:
        return spikes

    diffs = []
    for i in range(1, len(y_values)):
        prev_y = y_values[i - 1]
        curr_y = y_values[i]
        if is_number(prev_y) and is_number(curr_y):
            diffs.append(abs(curr_y - prev_y))

    avg_diff = safe_avg(diffs)
    if not avg_diff or avg_diff == 0:
        return spikes

    for i in range(1, len(y_values)):
        prev_y = y_values[i - 1]
        curr_y = y_values[i]

        if not is_number(prev_y) or not is_number(curr_y):
            continue

        jump = abs(curr_y - prev_y)
        if jump >= threshold_ratio * avg_diff:
            spikes.append(
                {
                    "index": i,
                    "x": x_values[i],
                    "y": curr_y,
                    "previous_y": prev_y,
                    "jump": jump,
                }
            )

    return spikes


def nearest_point_index(x_values, selected_x):
    if selected_x is None:
        return None

    best_index = None
    best_dist = None

    for i, x in enumerate(x_values):
        if not is_number(x):
            continue
        dist = abs(x - selected_x)
        if best_dist is None or dist < best_dist:
            best_dist = dist
            best_index = i

    return best_index


def local_window(x_values, y_values, center_index, window_size=4):
    if center_index is None:
        return []

    start = max(0, center_index - window_size)
    end = min(len(x_values), center_index + window_size + 1)

    window = []
    for i in range(start, end):
        window.append({"index": i, "x": x_values[i], "y": y_values[i]})
    return window


def pearson_correlation(x_values, y_values):
    if len(x_values) != len(y_values):
        return None

    paired = []
    for x, y in zip(x_values, y_values):
        if is_number(x) and is_number(y):
            paired.append((float(x), float(y)))

    if len(paired) < 3:
        return None

    xs = [p[0] for p in paired]
    ys = [p[1] for p in paired]

    mean_x = sum(xs) / float(len(xs))
    mean_y = sum(ys) / float(len(ys))

    num = 0.0
    den_x = 0.0
    den_y = 0.0

    for x, y in paired:
        dx = x - mean_x
        dy = y - mean_y
        num += dx * dy
        den_x += dx * dx
        den_y += dy * dy

    if den_x <= 0.0 or den_y <= 0.0:
        return None

    return num / math.sqrt(den_x * den_y)


def classify_correlation(corr):
    if corr is None:
        return "unknown"

    abs_corr = abs(corr)

    if abs_corr >= 0.9:
        strength = "very strong"
    elif abs_corr >= 0.7:
        strength = "strong"
    elif abs_corr >= 0.5:
        strength = "moderate"
    elif abs_corr >= 0.3:
        strength = "weak"
    else:
        strength = "very weak"

    if corr > 0.0:
        direction = "positive"
    elif corr < 0.0:
        direction = "negative"
    else:
        direction = "no"

    if direction == "no":
        return "no clear linear correlation"

    return "{} {} correlation".format(strength, direction)


def estimate_linear_slope(x_values, y_values):
    if len(x_values) != len(y_values):
        return None

    paired = []
    for x, y in zip(x_values, y_values):
        if is_number(x) and is_number(y):
            paired.append((float(x), float(y)))

    if len(paired) < 2:
        return None

    xs = [p[0] for p in paired]
    ys = [p[1] for p in paired]

    mean_x = sum(xs) / float(len(xs))
    mean_y = sum(ys) / float(len(ys))

    num = 0.0
    den = 0.0

    for x, y in paired:
        dx = x - mean_x
        num += dx * (y - mean_y)
        den += dx * dx

    if den <= 0.0:
        return None

    return num / den


def summarise_line_graph(graph_payload):
    x_values = graph_payload.get("x_values", [])
    y_values = graph_payload.get("y_values", [])

    return {
        "graph_id": graph_payload.get("graph_id"),
        "graph_name": graph_payload.get("graph_name"),
        "graph_type": graph_payload.get("graph_type", "line"),
        "x_axis_name": graph_payload.get("x_axis_name"),
        "y_axis_name": graph_payload.get("y_axis_name"),
        "point_count": len(y_values),
        "min_y": safe_min(y_values),
        "max_y": safe_max(y_values),
        "avg_y": safe_avg(y_values),
        "range_y": safe_range(y_values),
        "trend": basic_trend(y_values),
        "spikes": detect_spikes(x_values, y_values)[:10],
    }


def summarise_xy_graph(graph_payload):
    x_values = graph_payload.get("x_values", [])
    y_values = graph_payload.get("y_values", [])

    corr = pearson_correlation(x_values, y_values)
    slope = estimate_linear_slope(x_values, y_values)

    return {
        "graph_id": graph_payload.get("graph_id"),
        "graph_name": graph_payload.get("graph_name"),
        "graph_type": graph_payload.get("graph_type", "xy"),
        "x_axis_name": graph_payload.get("x_axis_name"),
        "y_axis_name": graph_payload.get("y_axis_name"),
        "point_count": min(len(x_values), len(y_values)),
        "min_x": safe_min(x_values),
        "max_x": safe_max(x_values),
        "avg_x": safe_avg(x_values),
        "range_x": safe_range(x_values),
        "min_y": safe_min(y_values),
        "max_y": safe_max(y_values),
        "avg_y": safe_avg(y_values),
        "range_y": safe_range(y_values),
        "correlation_coefficient": corr,
        "correlation_description": classify_correlation(corr),
        "estimated_linear_slope": slope,
    }


def summarise_graph(graph_payload):
    graph_type = graph_payload.get("graph_type", "line")

    if graph_type == "xy":
        return summarise_xy_graph(graph_payload)

    return summarise_line_graph(graph_payload)


def build_graph_context(graph_payload, user_question, selected_x=None):
    x_values = graph_payload.get("x_values", [])
    y_values = graph_payload.get("y_values", [])
    graph_type = graph_payload.get("graph_type", "line")

    summary = summarise_graph(graph_payload)

    selected_point = None
    nearby_points = []

    if graph_type == "line":
        selected_index = nearest_point_index(x_values, selected_x)

        if selected_index is not None:
            selected_point = {
                "index": selected_index,
                "x": x_values[selected_index],
                "y": y_values[selected_index],
            }
            nearby_points = local_window(x_values, y_values, selected_index)

    sample_points = []
    max_points = 40

    if len(x_values) <= max_points:
        for i in range(min(len(x_values), len(y_values))):
            sample_points.append({"x": x_values[i], "y": y_values[i]})
    else:
        step = max(1, len(x_values) // max_points)
        for i in range(0, min(len(x_values), len(y_values)), step):
            sample_points.append({"x": x_values[i], "y": y_values[i]})

    return {
        "question": user_question,
        "graph_type": graph_type,
        "graph_summary": summary,
        "selected_point": selected_point,
        "nearby_points": nearby_points,
        "sample_points": sample_points,
    }
