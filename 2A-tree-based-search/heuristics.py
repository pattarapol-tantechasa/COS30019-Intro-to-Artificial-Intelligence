import math

# Compute scaling once for each graph. Global scale can be passed to any algorithm that uses the heuristic 
SCALE = None # global variable for scaling

# Scale factor. Required to ensure heuristic is admissible
# Greedy, A* and any informed search can use this.
def compute_min_cost_per_unit(graph):
    """
    Compute the minimum cost per unit distance among all edges. Ensures heuristic is admissible
    """
    min_ratio = float('inf')
    for u in graph.edges:
        x1, y1 = graph.nodes[u]
        for v, cost in graph.edges[u]:
            x2, y2 = graph.nodes[v]
            dist = math.hypot(x1 - x2, y1 - y2)
            if dist > 0: # avoid division by 0
                ratio = cost / dist
                if ratio < min_ratio:
                    min_ratio = ratio
    return min_ratio if min_ratio != float('inf') else 1 # fallback to min_cost = 1


def heuristic(graph, node):
    '''Scaled straight-line distance to the nearest goal'''

    global SCALE # use global scale variable 
    if SCALE is None:
        SCALE = compute_min_cost_per_unit(graph)

    x1, y1 = graph.nodes[node]
    min_dist = min(
        math.hypot(x1 - graph.nodes[g][0], y1 - graph.nodes[g][1])
        for g in graph.destinations
    )
    return min_dist * SCALE
