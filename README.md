# DriveSim-AI
# OBD-II Drive Analytics Dashboard

## Overview

The OBD-II Drive Analytics Dashboard is an interactive Dash application designed to transform raw vehicle telemetry data into actionable driving insights.

The dashboard accepts a raw CSV export containing OBD-II sensor data collected during a vehicle journey and automatically generates a suite of analytics, visualisations, and performance metrics. These insights help users understand vehicle behaviour, driving style, fuel usage, thermal performance, braking characteristics, and overall journey efficiency.

The application is intended for:

* Automotive data analysis
* Vehicle performance evaluation
* Driver behaviour assessment
* Fleet and transport analytics
* Motorsport and track-day analysis
* Academic and engineering projects involving vehicle telemetry

---

## Features

### Vehicle Motion Analysis

Visualise how the vehicle behaved throughout the journey.

* Distance travelled over time
* Speed profile analysis
* Acceleration analysis
* Drive trace visualisation
* Journey statistics and summaries

### Thermodynamics Analysis

Monitor engine and cooling system behaviour.

* Engine coolant temperature trends
* Intake air temperature analysis
* Engine load monitoring
* Fan and cooling system analytics
* Thermal performance insights

### Fuel and Energy Analytics

Evaluate vehicle efficiency and fuel consumption.

* Instantaneous fuel consumption
* Fuel remaining estimates
* Fuel residual calculations
* Engine power estimation
* Longitudinal force analysis
* Cumulative energy expenditure

### Braking Analysis

Understand braking performance during a drive.

* Braking event detection
* Braking force estimation
* Brake torque calculations
* Braking power analysis
* Regenerated or dissipated braking energy

### Driver Behaviour Classification

Identify driving characteristics from telemetry patterns.

Potential classifications include:

* Eco driving
* Normal driving
* Aggressive driving
* Mixed driving behaviour

### Granite AI Analysis

Generate AI-powered insights from processed drive data.

The Granite analysis module can be used to provide:

* Journey summaries
* Driving behaviour observations
* Vehicle performance commentary
* Potential efficiency recommendations

---

## Supported Input Data

The dashboard expects a CSV file containing OBD-II telemetry data collected during a drive.

Typical parameters include:

| Parameter                  | Description                |
| -------------------------- | -------------------------- |
| Time                       | Timestamp or elapsed time  |
| Vehicle Speed              | Vehicle speed measurements |
| Engine RPM                 | Engine rotational speed    |
| Engine Load                | Calculated engine load     |
| Coolant Temperature        | Engine coolant temperature |
| Intake Air Temperature     | Intake air temperature     |
| Fuel Level                 | Remaining fuel level       |
| Mass Air Flow (MAF)        | Airflow into engine        |
| Throttle Position          | Accelerator pedal demand   |
| GPS Coordinates (optional) | Vehicle location tracking  |

Additional parameters may be utilised if available.

---

## Processing Pipeline

The dashboard performs the following workflow:

### 1. Data Upload

The user uploads a raw OBD-II CSV dataset.

### 2. Data Validation

The application validates:

* Required columns
* Data quality
* Missing values
* Timestamp consistency

### 3. Feature Engineering

Derived metrics are calculated, including:

* Distance travelled
* Acceleration
* Engine power
* Longitudinal force
* Fuel consumption
* Braking force
* Thermal indicators

### 4. Dashboard Generation

Interactive visualisations and KPI cards are generated automatically.

### 5. AI Analysis

Optional AI-driven narrative summaries can be produced using the processed data.

---

## Installation

### Clone the Repository

```bash
git clone <repository-url>
cd <repository-folder>
```

### Create a Virtual Environment

```bash
python -m venv .venv
```

Activate:

Windows:

```bash
.venv\Scripts\activate
```

Linux / macOS:

```bash
source .venv/bin/activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Running the Dashboard

Launch the application:

```bash
python dashboard_app_final_V5.py
```

The dashboard will start locally and can be accessed through:

```text
http://127.0.0.1:8050
```

---

## Usage

1. Launch the dashboard.
2. Upload a raw OBD-II CSV file.
3. Wait for preprocessing to complete.
4. Explore the available analytics sections.
5. Review generated visualisations and performance metrics.
6. Export results if required.

---

## Project Structure

```text
project/
│
├── dashboard_app_final_V5.py
├── requirements.txt
├── README.md
│
├── src/
│   ├── features/
│   ├── analytics/
│   ├── visualisation/
│   └── utils/
│
└── data/
```

---

## Example Applications

### Driver Behaviour Analysis

Determine whether a journey exhibits:

* Aggressive acceleration
* Harsh braking
* Consistent cruising
* Fuel-efficient driving

### Vehicle Performance Monitoring

Assess:

* Engine performance
* Power demand
* Thermal stability
* Fuel utilisation

### Fleet Management

Analyse:

* Driver performance
* Fuel efficiency trends
* Vehicle operating conditions
* Route characteristics

---

## Limitations

* Results are dependent on sensor quality and sampling frequency.
* Missing OBD-II parameters may reduce the number of available analytics.
* Different vehicle manufacturers may expose different PID sets.
* Fuel consumption calculations may rely on estimated values when certain sensors are unavailable.

---

## Future Improvements

Potential enhancements include:

* Live OBD-II streaming support
* GPS route mapping
* Predictive maintenance analytics
* Advanced machine learning driver profiling
* Multi-drive comparison tools
* Automated reporting and export functionality

---

## License

This project is provided for educational, research, and analytical purposes. Ensure compliance with applicable vehicle data privacy and usage regulations when collecting and processing telemetry data.
