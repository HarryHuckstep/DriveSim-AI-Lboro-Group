import numpy as np
import pandas as pd


# ==========================================================
# FEATURE EXTRACTION
# ==========================================================

def extract_driver_features(df):

    if "vehicle_speed" not in df.columns:
        raise ValueError("vehicle_speed column missing")

    # ------------------------------------------------------
    # Create synthetic time axis if none exists
    # ------------------------------------------------------

    if "timestamp" in df.columns:
        time_s = np.arange(len(df))
    else:
        time_s = np.arange(len(df))

    # ------------------------------------------------------
    # Speed
    # ------------------------------------------------------

    speed_ms = df["vehicle_speed"].astype(float) / 3.6

    acceleration = np.gradient(speed_ms, time_s)

    jerk = np.gradient(acceleration, time_s)

    # ------------------------------------------------------
    # Optional signals
    # ------------------------------------------------------

    throttle = (
        df["throttle_pct"].astype(float)
        if "throttle_pct" in df.columns
        else np.zeros(len(df))
    )

    rpm = (
        df["engine_rpm"].astype(float)
        if "engine_rpm" in df.columns
        else np.zeros(len(df))
    )

    engine_load = (
        df["engine_load_pct"].astype(float)
        if "engine_load_pct" in df.columns
        else np.zeros(len(df))
    )

    # ------------------------------------------------------
    # Events
    # ------------------------------------------------------

    hard_brakes = acceleration < -1.5

    aggressive_accels = acceleration > 1.0

    throttle_spikes = np.abs(np.gradient(throttle, time_s)) > 30

    # ------------------------------------------------------
    # Metrics
    # ------------------------------------------------------

    features = {

        "avg_speed":
            np.mean(speed_ms) * 3.6,

        "max_speed":
            np.max(speed_ms) * 3.6,

        "avg_rpm":
            np.mean(rpm),

        "rpm_std":
            np.std(rpm),

        "avg_throttle":
            np.mean(throttle),

        "throttle_std":
            np.std(throttle),

        "avg_engine_load":
            np.mean(engine_load),

        "max_accel":
            np.max(acceleration),

        "max_brake":
            np.min(acceleration),

        "jerk_std":
            np.std(jerk),

        "hard_brake_count":
            np.sum(hard_brakes),

        "aggressive_accel_count":
            np.sum(aggressive_accels),

        "throttle_spike_count":
            np.sum(throttle_spikes),
    }

    return features


# ==========================================================
# DRIVER CLASSIFICATION
# ==========================================================

def classify_driver(features):

    aggression_score = (
        features["hard_brake_count"] * 2
        + features["aggressive_accel_count"] * 2
        + features["throttle_spike_count"] * 0.5
        + features["throttle_std"] * 0.3
        + features["jerk_std"] * 5
    )

    # ------------------------------------------------------
    # Classifications
    # ------------------------------------------------------

    if (
        features["hard_brake_count"] > 20
        or features["max_brake"] < -2.5
    ):
        label = "Heavy Braker"

    elif aggression_score > 60:
        label = "Aggressive Driver"

    elif (
        features["avg_rpm"] > 3200
        and features["avg_speed"] > 60
    ):
        label = "High-Revs Driver"

    elif (
        features["avg_throttle"] < 18
        and features["hard_brake_count"] < 5
        and features["jerk_std"] < 0.5
    ):
        label = "Efficient Driver"

    elif (
        features["jerk_std"] < 0.8
        and features["throttle_std"] < 10
    ):
        label = "Calm Driver"

    else:
        label = "Normal Driver"

    return label, aggression_score


# ==========================================================
# MAIN API FUNCTION
# ==========================================================

def analyse_driver_behaviour(df):

    features = extract_driver_features(df)

    label, aggression_score = classify_driver(features)

    return {
        "driver_type": label,
        "aggression_score": round(aggression_score, 2),
        "features": features,
    }
