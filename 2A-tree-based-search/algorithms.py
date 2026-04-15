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

    frontier = deque()
    frontier.append((start, [start]))

    explored = set()
    nodes_created = 1

    while frontier:
        node, path = frontier.popleft()

        if node in explored:
            continue
        explored.add(node)

        if node in goals:
            return node, nodes_created, path

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

    frontier = [(start, [start])]

    explored = set()
    nodes_created = 1

    while frontier:
        node, path = frontier.pop()

        if node in explored:
            continue
        explored.add(node)

        if node in goals:
            return node, nodes_created, path

        for neighbour, cost in reversed(graph.get_neighbours(node)):
            if neighbour not in explored:
                nodes_created += 1
                frontier.append((neighbour, path + [neighbour]))

    return None, nodes_created, []
