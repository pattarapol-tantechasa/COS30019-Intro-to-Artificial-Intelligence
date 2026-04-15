import math
from graph import parse_file


def cus2(graph):
    """
    CUS2 — Iterative Deepening A* (IDA*).
    Informed method that finds the least-cost path to a goal.
    Uses Euclidean distance as the heuristic (straight-line to nearest goal).
    Returns: (goal_node, number_of_nodes_created, path_list)
    """
    start = graph.origin
    goals = set(graph.destinations)

    def heuristic(node):
        x1, y1 = graph.nodes[node]
        return min(
            math.sqrt((x1 - graph.nodes[g][0])**2 + (y1 - graph.nodes[g][1])**2)
            for g in goals
        )

    threshold = heuristic(start)
    nodes_created = 1

    while True:
        result, nodes_created, path = _dls(
            start, goals, graph, heuristic,
            [start], 0, threshold, nodes_created
        )

        if isinstance(result, int):
            return result, nodes_created, path

        if result == float('inf'):
            return None, nodes_created, []

        threshold = result


def _dls(node, goals, graph, heuristic, path, g_cost, threshold, nodes_created):
    """
    Recursive depth-limited search used inside IDA*.
    """
    f = g_cost + heuristic(node)

    if f > threshold:
        return f, nodes_created, []

    if node in goals:
        return node, nodes_created, path

    minimum = float('inf')

    for neighbour, cost in graph.get_neighbours(node):
        if neighbour in path:
            continue

        nodes_created += 1
        result, nodes_created, found_path = _dls(
            neighbour, goals, graph, heuristic,
            path + [neighbour], g_cost + cost, threshold, nodes_created
        )

        if isinstance(result, int):
            return result, nodes_created, found_path

        if result < minimum:
            minimum = result

    return minimum, nodes_created, []
