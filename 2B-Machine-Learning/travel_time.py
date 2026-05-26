"""
travel_time.py
--------------
Converts ML-predicted normalised traffic volume into a travel time (seconds)
for an edge between two SCATS sites.

Flow -> Speed conversion uses the quadratic model:
    flow = -1.4648375 * speed^2 + 93.75 * speed

Rearranged for speed (under-capacity / green line assumed):
    speed = (-B + sqrt(B^2 + 4*A*x)) / (2*A)
    where A = -1.4648375, B = 93.75, x = flow (veh/hour)

Assumptions:
  - Speed limit: 60 km/h
  - Traffic is always under capacity (green line)
  - Flow <= 351 veh/hr -> speed capped at 60 km/h
  - Average intersection delay: 30 seconds per SCATS site
  - Distance between sites calculated from lat/lon in sites.csv
"""

import json
import math
import numpy as np
import pandas as pd
import tensorflow as tf


import config

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SPEED_LIMIT_KMH   = 60.0      # km/h — cap when flow is low
FLOW_CAP_BOUNDARY = 351.0     # veh/hr — below this, road is under speed limit
INTERSECTION_DELAY = 30.0     # seconds added per intersection traversed
SPEED_MIN_KMH     = 1.0       # floor to avoid division by zero

# Quadratic coefficients: flow = A*speed^2 + B*speed
A_COEFF = -1.4648375
B_COEFF =  93.75


# ---------------------------------------------------------------------------
# Flow -> Speed conversion
# ---------------------------------------------------------------------------

def flow_to_speed(flow_veh_per_hour: float) -> float:
    """Convert predicted traffic flow (veh/hr) to speed (km/h).

    Uses the under-capacity (green line) solution of the quadratic:
        speed = (-B + sqrt(B^2 + 4*A*flow)) / (2*A)

    Speed is capped at SPEED_LIMIT_KMH when flow is low, and floored at
    SPEED_MIN_KMH to avoid division by zero on very congested links.
    """
    if flow_veh_per_hour <= FLOW_CAP_BOUNDARY:
        return SPEED_LIMIT_KMH

    # Quadratic formula — under capacity branch (+ discriminant)
    discriminant = B_COEFF ** 2 + 4 * A_COEFF * flow_veh_per_hour

    if discriminant < 0:
        # Flow is beyond the parabola's range — road is extremely congested.
        # Return minimum driveable speed.
        return SPEED_MIN_KMH

    speed = (-B_COEFF + math.sqrt(discriminant)) / (2 * A_COEFF)
    speed = max(speed, SPEED_MIN_KMH)       # floor
    speed = min(speed, SPEED_LIMIT_KMH)     # cap at speed limit
    return speed


# ---------------------------------------------------------------------------
# Distance between two lat/lon points (Haversine)
# ---------------------------------------------------------------------------

def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Return the great-circle distance in kilometres between two points."""
    R = 6371.0  # Earth radius in km
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi  = math.radians(lat2 - lat1)
    dlam  = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
    return R * 2 * math.asin(math.sqrt(a))


# ---------------------------------------------------------------------------
# Denormalise ML prediction back to real vehicle counts
# ---------------------------------------------------------------------------

def denormalise(normalised_value: float, vmin: float, vmax: float) -> float:
    """Invert min-max normalisation to get vehicles per 15 minutes."""
    return normalised_value * (vmax - vmin) + vmin


def volume_to_flow(volume_per_15min: float) -> float:
    """Convert vehicles/15min to vehicles/hour."""
    return volume_per_15min * 4


# ---------------------------------------------------------------------------
# Scaler and site loader
# ---------------------------------------------------------------------------

def load_scalers() -> dict:
    """Load per-site scalers saved by run.py."""
    path = config.OUTPUT_DIR / "scalers.json"
    with open(path) as f:
        return json.load(f)


def load_sites() -> pd.DataFrame:
    """Load SCATS site coordinates saved by run.py.

    Returns a DataFrame with columns:
        SCATS Number, Location, NB_LATITUDE, NB_LONGITUDE
    Averaged across approaches so each SCATS Number has one lat/lon.
    """
    sites = pd.read_csv(config.OUTPUT_DIR / "sites.csv")
    # Average lat/lon across approaches for a single representative coordinate
    coords = (sites.groupby("SCATS Number")[["NB_LATITUDE", "NB_LONGITUDE"]]
              .mean()
              .reset_index())
    return coords


# ---------------------------------------------------------------------------
# Model loader
# ---------------------------------------------------------------------------

def load_model(model_name: str):
    """Load a trained Keras model by name ('lstm', 'gru', or 'saes').

    model_name: one of 'lstm', 'gru', 'saes'
    """
    model_name = model_name.lower()
    path = config.OUTPUT_DIR / model_name / f"{model_name}_model.keras"
    if not path.exists():
        raise FileNotFoundError(
            f"Model not found at {path}. Run {model_name}.py first."
        )
    return tf.keras.models.load_model(path)


# ---------------------------------------------------------------------------
# Predict normalised volume for a single site
# ---------------------------------------------------------------------------

def predict_volume(model, model_name: str, recent_window: np.ndarray) -> float:
    """Run the model on a single window of normalised volume readings.

    recent_window: 1D array of length WINDOW (normalised [0,1] values)
    Returns: predicted normalised volume (scalar)
    """
    model_name = model_name.lower()
    window = recent_window.reshape(1, -1)   # (1, 12)

    if model_name in ("lstm", "gru"):
        window = window[..., np.newaxis]    # (1, 12, 1) for recurrent models
    # SAEs expects (1, 12)

    pred = model.predict(window, verbose=0)
    return float(pred.flatten()[0])


# ---------------------------------------------------------------------------
# Main travel time function
# ---------------------------------------------------------------------------

def estimate_travel_time(
    site_a: int,
    site_b: int,
    model,
    model_name: str,
    scalers: dict,
    sites: pd.DataFrame,
    recent_windows: dict,
) -> float:
    """Estimate travel time in seconds from SCATS site A to site B.

    Flow is calculated from the *starting* site (A).

    Arguments:
        site_a:         Origin SCATS Number
        site_b:         Destination SCATS Number
        model:          Loaded Keras model
        model_name:     'lstm', 'gru', or 'saes'
        scalers:        Dict loaded from scalers.json
        sites:          DataFrame from load_sites()
        recent_windows: Dict mapping SCATS Number -> np.ndarray of shape (WINDOW,)
                        containing the most recent normalised volume readings

    Returns:
        Travel time in seconds (float), or a large penalty if data is missing.
    """
    PENALTY = 9999.0    # returned when a site has no data

    # --- Get coordinates ---
    row_a = sites[sites["SCATS Number"] == site_a]
    row_b = sites[sites["SCATS Number"] == site_b]
    if row_a.empty or row_b.empty:
        return PENALTY

    lat_a, lon_a = float(row_a["NB_LATITUDE"].iloc[0]), float(row_a["NB_LONGITUDE"].iloc[0])
    lat_b, lon_b = float(row_b["NB_LATITUDE"].iloc[0]), float(row_b["NB_LONGITUDE"].iloc[0])
    distance_km = haversine_km(lat_a, lon_a, lat_b, lon_b)

    # --- Predict volume at site A (flow is from the starting site) ---
    if site_a not in recent_windows:
        return PENALTY

    window = recent_windows[site_a]

    # Find the scaler for site A — average across approaches if multiple exist
    site_scalers = {k: v for k, v in scalers.items()
                    if k.startswith(f"{site_a}|")}
    if not site_scalers:
        return PENALTY

    avg_min = float(np.mean([v["min"] for v in site_scalers.values()]))
    avg_max = float(np.mean([v["max"] for v in site_scalers.values()]))

    # Predict and denormalise
    pred_normalised  = predict_volume(model, model_name, window)
    pred_veh_15min   = denormalise(pred_normalised, avg_min, avg_max)
    pred_veh_per_hr  = volume_to_flow(pred_veh_15min)

    # --- Flow -> Speed -> Travel time ---
    speed_kmh    = flow_to_speed(pred_veh_per_hr)
    speed_km_per_sec = speed_kmh / 3600.0

    if speed_km_per_sec <= 0:
        return PENALTY

    drive_time_sec    = distance_km / speed_km_per_sec
    total_time_sec    = drive_time_sec + INTERSECTION_DELAY

    return total_time_sec


# ---------------------------------------------------------------------------
# For testing purposes - checking travel time converter
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Flow -> Speed conversion checks:")
    for flow in [0, 100, 351, 500, 1000, 1500]:
        speed = flow_to_speed(flow)
        print(f"  flow={flow:5d} veh/hr  ->  speed={speed:.2f} km/h")
