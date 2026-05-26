import heapq  # Using heapq to efficiently always expand the node with the lowest f(n) (priority queue in O(log n) time)
import math

# Scale factor. Required to ensure heuristic is admissible
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


def heuristic(graph, node, scale):
    '''Scaled straight-line distance to the nearest goal'''

    x1, y1 = graph.nodes[node]

    min_dist = min(
        math.hypot(x1 - graph.nodes[g][0], y1 - graph.nodes[g][1])
        for g in graph.destinations
    )
    return min_dist * scale

# A-star search
def astar(graph):
    '''A* search. Uses scaled heuristic to ensure distance is always < path cost (admissible)
    Returns: goal_node, nodes_created, path_list'''
    start = graph.origin
    goals = set(graph.destinations)

    # Compute scale once per search
    scale = compute_min_cost_per_unit(graph)

    # Priority queue: (f, node, path, g)
    frontier = []
    # Push start node onto the heap
    start_h = heuristic(graph, start, scale) 
    heapq.heappush(frontier, (start_h, start, [start], 0))

    g_costs = {start: 0} # best known cost to each node
    explored = set()
    nodes_created = 1

    while frontier:
        f, node, path, g = heapq.heappop(frontier)

        if node in explored:
            continue
        explored.add(node)

        # Goal check after popping
        if node in goals:
            return node, nodes_created, path
        
        # Expand neighbours (sorted by Node ID for tie-breaking)
        for neighbour, cost in graph.get_neighbours(node):
            new_g = g + cost

            # Only proceed if this path is better
            if neighbour not in g_costs or new_g < g_costs[neighbour]:
                g_costs[neighbour] = new_g

                h = heuristic(graph, neighbour, scale)
                new_f = new_g + h

                nodes_created += 1
                heapq.heappush(frontier, (new_f, neighbour, path + [neighbour], new_g))

    return None, nodes_created, []