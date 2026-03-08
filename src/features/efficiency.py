from __future__ import annotations
import pandas as pd

def add_mdot_air_kgs(df):
    df["Air Flow Rate from Mass Flow Sensor [kg/s]"] = (df["Air Flow Rate from Mass Flow Sensor [g/s]"] / 1000)
    return df

def add_mdot_fuel(df, AFR = 14.7):
    df["mdot_fuel"] = df["Air Flow Rate from Mass Flow Sensor [kg/s]"]  / AFR

    return df

def add_chemical_power(df, LHV = 43e6):
    df["Pfuel"]  = df["mdot_fuel"] * LHV


def add_chemical_efficiency(df):
    add_mdot_air_kgs(df)
    add_mdot_fuel(df)
    add_chemical_power(df)
    df["chemical_efficiency"] = df["P_drive_W"] / df["Pfuel"]

    return df
