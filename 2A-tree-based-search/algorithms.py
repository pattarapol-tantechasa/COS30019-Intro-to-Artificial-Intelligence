from collections import deque
import heapq
import math

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

def gbfs(graph):


    start = graph.origin
    goals = set(graph.destinations)

    def heuristic(node):
        x1, y1 = graph.nodes[node]
        return min(
            math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
            for dest in goals
            for x2, y2 in [graph.nodes[dest]]
        )

    counter = 0
    frontier = [(heuristic(start), counter, start, [start])]
    explored = set()
    nodes_created = 1

    while frontier:
        h, _, node, path = heapq.heappop(frontier)

        if node in explored:
            continue
        explored.add(node)

        if node in goals:
            return node, nodes_created, path

        for neighbour, cost in graph.get_neighbours(node):
            if neighbour not in explored:
                counter += 1
                nodes_created += 1
                heapq.heappush(frontier, (heuristic(neighbour), counter, neighbour, path + [neighbour]))

    return None, nodes_created, []