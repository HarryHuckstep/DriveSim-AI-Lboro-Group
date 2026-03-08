from __future__ import annotations
import pandas as pd

def convert_mdot_air_to_kgs(df):
    df["Air Flow Rate from Mass Flow Sensor [kg/s]"] = (df["Air Flow Rate from Mass Flow Sensor [g/s]"] / 1000)
    return df