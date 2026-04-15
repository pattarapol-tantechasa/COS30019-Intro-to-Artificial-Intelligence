from collections import deque

def bfs(graph):
    """
    Breadth-First Search.
    Frontier is a FIFO queue — we always expand the OLDEST node first.
    This guarantees we find the shallowest (fewest hops) goal first.
    Returns: (goal_node, number_of_nodes_created, path_list)
    """
    start = graph.origin
    goals = set(graph.destinations)

    # Each item in the frontier is (current_node, path_so_far)
    frontier = deque()
    frontier.append((start, [start]))

    explored = set()
    nodes_created = 1  # We count the start node

    while frontier:
        # BFS: take from the LEFT (oldest = shallowest)
        node, path = frontier.popleft()

        if node in explored:
            continue
        explored.add(node)

        # Goal check AFTER popping (as per lecture slides)
        if node in goals:
            return node, nodes_created, path

        # Expand: get neighbours sorted by node number (tie-breaking rule)
        for neighbour, cost in graph.get_neighbours(node):
            if neighbour not in explored:
                nodes_created += 1
                frontier.append((neighbour, path + [neighbour]))

    return None, nodes_created, []


def dfs(graph):
    """
    Depth-First Search.
    Frontier is a LIFO stack — we always expand the NEWEST node first.
    This drives the search deep along one path before backtracking.
    Returns: (goal_node, number_of_nodes_created, path_list)
    """
    start = graph.origin
    goals = set(graph.destinations)

    # Stack: list used as LIFO
    frontier = [(start, [start])]

    explored = set()
    nodes_created = 1

    while frontier:
        # DFS: take from the RIGHT (newest = deepest)
        node, path = frontier.pop()

        if node in explored:
            continue
        explored.add(node)

        if node in goals:
            return node, nodes_created, path

        # Expand: neighbours sorted ascending — but we push to a stack
        # so to expand smaller-numbered nodes FIRST, we push in REVERSE order
        # (last pushed = first popped = smallest node number)
        for neighbour, cost in reversed(graph.get_neighbours(node)):
            if neighbour not in explored:
                nodes_created += 1
                frontier.append((neighbour, path + [neighbour]))

    return None, nodes_created, []


def cus2(graph):
    """
    CUS2 — Iterative Deepening A* (IDA*).
    Informed method that finds the least-cost path to a goal.
    Uses Euclidean distance as the heuristic (straight-line to nearest goal).
    Returns: (goal_node, number_of_nodes_created, path_list)
    """
    import math

    start = graph.origin
    goals = set(graph.destinations)

    # heuristic: straight-line distance from a node to the nearest goal
    def heuristic(node):
        x1, y1 = graph.nodes[node]
        return min(
            math.sqrt((x1 - graph.nodes[g][0])**2 + (y1 - graph.nodes[g][1])**2)
            for g in goals
        )

    # threshold starts as the heuristic from the start node
    threshold = heuristic(start)
    nodes_created = 1

    while True:
        # each iteration does a depth-limited search with the current threshold
        result, nodes_created, path = _dls(
            start, goals, graph, heuristic,
            [start], 0, threshold, nodes_created
        )

        if isinstance(result, int):
            # result is the goal node we found
            return result, nodes_created, path

        if result == float('inf'):
            # no path exists at all
            return None, nodes_created, []

        # raise the threshold to the next lowest f-value we saw
        threshold = result


def _dls(node, goals, graph, heuristic, path, g_cost, threshold, nodes_created):
    """
    Recursive depth-limited search used inside IDA*.
    g_cost is the cost to reach this node so far.
    Returns either (goal_node, nodes_created, path) or (new_threshold, nodes_created, [])
    """
    f = g_cost + heuristic(node)

    if f > threshold:
        # over budget — return this f value so IDA* can raise the threshold
        return f, nodes_created, []

    if node in goals:
        return node, nodes_created, path

    minimum = float('inf')

    # expand neighbours in ascending order (tie-breaking rule)
    for neighbour, cost in graph.get_neighbours(node):
        if neighbour in path:
            # skip nodes already in the current path to avoid cycles
            continue

        nodes_created += 1
        result, nodes_created, found_path = _dls(
            neighbour, goals, graph, heuristic,
            path + [neighbour], g_cost + cost, threshold, nodes_created
        )

        if isinstance(result, int):
            # found a goal — bubble it back up
            return result, nodes_created, found_path

        if result < minimum:
            minimum = result

    return minimum, nodes_created, []
