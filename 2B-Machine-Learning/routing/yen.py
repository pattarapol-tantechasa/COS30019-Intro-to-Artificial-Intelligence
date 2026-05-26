"""
yen.py
------
Implements Yen's k-shortest paths algorithm using the existing A* search from 2A.

The assignment brief requires returning up to 5 routes from origin to destination
with estimated travel time. A* only returns the single best path, so Yen's
algorithm wraps it to find alternatives by systematically blocking edges
from previously found paths and re-running the search.

Usage:
    from yen import k_shortest_paths

    paths = k_shortest_paths(graph, k=5)
    for i, (path, cost) in enumerate(paths):
        print(f"Route {i+1}: {path}  ({cost:.1f} seconds)")
"""

import copy
import heapq

from graph import Graph
from astar import astar


def path_cost(graph: Graph, path: list) -> float:
    """Sum edge costs along a path to get total travel time in seconds."""
    total = 0.0
    for i in range(len(path) - 1):
        from_node = path[i]
        to_node   = path[i + 1]

        # Find the cost of this specific edge
        cost = None
        for neighbour, edge_cost in graph.edges.get(from_node, []):
            if neighbour == to_node:
                cost = edge_cost
                break

        if cost is None:
            # Edge not found — path is invalid
            return float('inf')
        total += cost

    # Add intersection delay for each node traversed (already included in
    # edge costs via travel_time.py, so we don't double-count here)
    return total


def _remove_edge(graph: Graph, u: int, v: int):
    """Remove the directed edge u->v from the graph in place."""
    graph.edges[u] = [(n, c) for n, c in graph.edges.get(u, []) if n != v]


def _remove_node(graph: Graph, node: int):
    """Remove a node and all its edges from the graph in place."""
    if node in graph.nodes:
        del graph.nodes[node]
    if node in graph.edges:
        del graph.edges[node]
    # Remove any edges pointing to this node
    for u in graph.edges:
        graph.edges[u] = [(n, c) for n, c in graph.edges[u] if n != node]


def k_shortest_paths(graph: Graph, k: int = 5) -> list:
    """Find up to k shortest paths from graph.origin to graph.destinations.

    Uses Yen's algorithm — runs A* repeatedly on modified copies of the graph
    to find alternative routes.

    Arguments:
        graph:  Populated Graph object with origin and destinations set
        k:      Maximum number of paths to return (default 5)

    Returns:
        List of (path, cost) tuples sorted by ascending cost.
        path is a list of SCATS Numbers from origin to destination.
        cost is total travel time in seconds.
        Returns fewer than k paths if the graph doesn't have enough routes.
    """
    origin      = graph.origin
    destination = graph.destinations[0]  # single destination for TBRGS queries

    # --- Step 1: Find the first (shortest) path using A* ---
    _, _, first_path = astar(graph)

    if not first_path:
        print("No path found between origin and destination.")
        return []

    # A* list of confirmed shortest paths
    confirmed = [(first_path, path_cost(graph, first_path))]

    # Candidate heap: (cost, path) — paths found but not yet confirmed
    candidates = []

    for i in range(1, k):
        prev_path, _ = confirmed[-1]

        # --- Step 2: Generate spur paths from each node in the previous path ---
        for spur_idx in range(len(prev_path) - 1):
            spur_node  = prev_path[spur_idx]
            root_path  = prev_path[:spur_idx + 1]
            root_cost  = path_cost(graph, root_path)

            # Work on a deep copy so we can remove edges/nodes freely
            g_copy = copy.deepcopy(graph)
            g_copy.origin      = spur_node
            g_copy.destinations = [destination]

            # Remove edges that are shared with previously found paths
            # to force the search to find a different route
            for confirmed_path, _ in confirmed:
                if (len(confirmed_path) > spur_idx and
                        confirmed_path[:spur_idx + 1] == root_path):
                    u = confirmed_path[spur_idx]
                    v = confirmed_path[spur_idx + 1]
                    _remove_edge(g_copy, u, v)

            # Remove all nodes in the root path except the spur node
            # to prevent the spur from looping back through the root
            for node in root_path[:-1]:
                _remove_node(g_copy, node)

            # Run A* on the modified graph to find a spur path
            _, _, spur_path = astar(g_copy)

            if not spur_path or spur_path[0] != spur_node:
                continue

            # Reconstruct the full path: root + spur (excluding duplicate spur_node)
            full_path = root_path[:-1] + spur_path
            full_cost = path_cost(graph, full_path)  # cost on original graph

            if full_cost == float('inf'):
                continue

            # Avoid duplicate candidates
            if not any(full_path == p for p, _ in candidates):
                heapq.heappush(candidates, (full_cost, full_path))

        if not candidates:
            break   # no more alternative paths exist

        # Move the best candidate into confirmed
        best_cost, best_path = heapq.heappop(candidates)

        # Skip duplicates already in confirmed
        while any(best_path == p for p, _ in confirmed):
            if not candidates:
                best_path = None
                break
            best_cost, best_path = heapq.heappop(candidates)

        if best_path is None:
            break

        confirmed.append((best_path, best_cost))

    return confirmed


def format_paths(paths: list, sites_df=None) -> str:
    """Format k-shortest paths for display.

    Arguments:
        paths:    Output from k_shortest_paths()
        sites_df: Optional DataFrame from load_sites() to show location names

    Returns:
        Formatted string for printing or GUI display
    """
    if not paths:
        return "No routes found."

    lines = []
    for i, (path, cost) in enumerate(paths):
        minutes = cost / 60.0
        route_str = " -> ".join(str(n) for n in path)
        lines.append(f"Route {i + 1}: {route_str}")
        lines.append(f"  Estimated travel time: {minutes:.1f} minutes "
                     f"({cost:.0f} seconds), {len(path) - 1} links\n")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Quick sanity check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Parent folder path
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

    from travel_time import load_model, load_scalers, load_sites
    from graph_builder import build_graph

    MODEL_NAME = "saes" # edit to change model

    print(f"Loading model ({MODEL_NAME}) ...")
    model   = load_model(MODEL_NAME)
    scalers = load_scalers()
    sites   = load_sites()

    print("Building graph ...")
    graph = build_graph(
        model, MODEL_NAME, scalers, sites,
        origin=2000,
        destinations=[3002],
    )

    print("\nFinding top-5 routes ...")
    paths = k_shortest_paths(graph, k=5)

    print("\n" + format_paths(paths))
