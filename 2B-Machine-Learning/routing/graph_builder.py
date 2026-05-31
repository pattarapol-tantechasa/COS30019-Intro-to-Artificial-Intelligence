"""
graph_builder.py
----------------
Builds a Graph object from the SCATS dataset for use with A* search.

Each SCATS intersection becomes a node. Edges are created between each site
and its k nearest neighbours (default k=5, configurable in config.py).
Edge costs are travel time in seconds, estimated using the ML model and the
flow->speed conversion from PDF provided on Canvas

Usage:
    from graph_builder import build_graph
    from travel_time import load_model, load_scalers, load_sites

    model    = load_model('saes')
    scalers  = load_scalers()
    sites    = load_sites()
    graph    = build_graph(model, 'saes', scalers, sites,
                           origin=2000, destinations=[3002])
"""

import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree

# Parent folder path
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))



import config
from graph import Graph
from travel_time import (
    estimate_travel_time,
    load_scalers,
    load_sites,
    haversine_km,
)

# ---------------------------------------------------------------------------
# How many nearest neighbours each site connects to.
# Increase for a denser graph, decrease for faster search.
# ---------------------------------------------------------------------------
K_NEIGHBOURS = getattr(config, 'K_NEIGHBOURS', 5)


# ---------------------------------------------------------------------------
# Build recent windows from scats_long.csv
# ---------------------------------------------------------------------------

def build_recent_windows(long_csv_path, window: int = config.WINDOW) -> dict:
    """Extract the most recent `window` normalised volume readings per site.

    The scats_long.csv file has already-cleaned raw volumes. We normalise
    per-site here (same min-max as dataset.py) so the window is in [0,1]
    space ready for the model.

    Returns:
        Dict mapping SCATS Number (int) -> np.ndarray of shape (window,)
    """
    df = pd.read_csv(long_csv_path, parse_dates=['Timestamp'])
    df = df.sort_values(['SCATS Number', 'Timestamp'])

    windows = {}
    for site_id, group in df.groupby('SCATS Number'):
        # Average across all approaches at this site for a single time series
        ts = (group.groupby('Timestamp')['Volume']
              .mean()
              .sort_index()
              .values
              .astype(float))

        if len(ts) < window:
            continue    # not enough data

        recent = ts[-window:]

        # Min-max normalise using the same window of data
        vmin, vmax = recent.min(), recent.max()
        if vmax == vmin:
            normalised = np.zeros(window)
        else:
            normalised = (recent - vmin) / (vmax - vmin)

        windows[int(site_id)] = normalised

    return windows


# ---------------------------------------------------------------------------
# K-nearest neighbour graph construction
# ---------------------------------------------------------------------------

def build_graph(
    model,
    model_name: str,
    scalers: dict,
    sites: pd.DataFrame,
    origin: int,
    destinations: list,
    k: int = K_NEIGHBOURS,
) -> Graph:
    """Build and return a Graph populated with SCATS sites and travel time edges.

    Arguments:
        model:        Loaded Keras model (lstm, gru, or saes)
        model_name:   'lstm', 'gru', or 'saes'
        scalers:      Dict from load_scalers()
        sites:        DataFrame from load_sites() — one row per SCATS Number
        origin:       SCATS Number of the trip origin
        destinations: List of SCATS Numbers for the destination(s)
        k:            Number of nearest neighbours per site

    Returns:
        A populated Graph object ready for astar()
    """
    long_csv = config.OUTPUT_DIR / 'scats_long.csv'
    print(f"Building recent windows from {long_csv.name} ...")
    recent_windows = build_recent_windows(long_csv)
    print(f"  {len(recent_windows)} sites with sufficient data")

    # Only keep sites that have recent window data
    valid_sites = sites[sites['SCATS Number'].isin(recent_windows.keys())].copy()
    valid_sites = valid_sites.reset_index(drop=True)

    site_ids = valid_sites['SCATS Number'].tolist()
    lats     = valid_sites['NB_LATITUDE'].values
    lons     = valid_sites['NB_LONGITUDE'].values

    print(f"Building graph with {len(site_ids)} nodes, k={k} neighbours ...")

    # Use BallTree with haversine metric for efficient nearest-neighbour lookup
    # BallTree expects coordinates in radians
    coords_rad = np.radians(np.column_stack([lats, lons]))
    tree = BallTree(coords_rad, metric='haversine')

    # Query k+1 neighbours because the closest match is always the site itself
    distances, indices = tree.query(coords_rad, k=k + 1)

    graph = Graph()

    # --- Add all valid sites as nodes ---
    # graph.nodes expects (x, y) — we use (longitude, latitude) so that the
    # A* heuristic (straight-line distance) works correctly geographically.
    for site_id, lat, lon in zip(site_ids, lats, lons):
        graph.add_node(site_id, lon, lat)

    # --- Add directed edges to k nearest neighbours ---
    edges_added = 0
    edges_skipped = 0

    for i, site_a in enumerate(site_ids):
        neighbour_indices = indices[i][1:]  # skip index 0 (self)

        for j in neighbour_indices:
            site_b = site_ids[j]

            travel_time = estimate_travel_time(
                site_a, site_b,
                model, model_name,
                scalers, valid_sites,
                recent_windows,
            )

            if travel_time >= 9999.0:
                edges_skipped += 1
                continue

            # Add edge in both directions (undirected road network)
            graph.add_edge(site_a, site_b, travel_time)
            graph.add_edge(site_b, site_a, travel_time)
            edges_added += 2

    print(f"  Edges added:   {edges_added}")
    print(f"  Edges skipped: {edges_skipped} (missing data)")

    # --- Set origin and destinations ---
    if origin not in graph.nodes:
        raise ValueError(
            f"Origin site {origin} not found in graph. "
            f"Check that it exists in sites.csv and has sufficient data."
        )
    for dest in destinations:
        if dest not in graph.nodes:
            raise ValueError(
                f"Destination site {dest} not found in graph. "
                f"Check that it exists in sites.csv and has sufficient data."
            )

    graph.origin       = origin
    graph.destinations = destinations

    print(f"  Origin:      {origin}")
    print(f"  Destination: {destinations}")
    print("Graph built successfully.")

    return graph


# ---------------------------------------------------------------------------
# Quick sanity check - testing purposes. Run this directly to ensure it works
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    
    from travel_time import load_model, load_scalers, load_sites

    MODEL_NAME = "saes"

    print(f"Loading model ({MODEL_NAME}) ...")
    model   = load_model(MODEL_NAME)
    scalers = load_scalers()
    sites   = load_sites()

    # Example origin/destination from the assignment brief
    graph = build_graph(
        model, MODEL_NAME, scalers, sites,
        origin=2000,
        destinations=[3002],
    )

    print(f"\nNode count: {len(graph.nodes)}")
    print(f"Sample neighbours of 2000: {graph.get_neighbours(2000)[:3]}")
