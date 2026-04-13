import heapq
import math


def heuristic(graph, node):
    x1, y1 = graph.nodes[node]
    return min(
        math.hypot(x1 - graph.nodes[g][0], y1 - graph.nodes[g][1])
        for g in graph.destinations
    )


def gbfs(graph):
    start = graph.origin
    goals = set(graph.destinations)

    counter = 0
    start_h = heuristic(graph, start)
    frontier = [(start_h, counter, start, [start])]
    heapq.heapify(frontier)

    visited = set()
    nodes_created = 1

    while frontier:
        h, _, current, path = heapq.heappop(frontier)

        if current in visited:
            continue
        visited.add(current)

        if current in goals:
            return current, nodes_created, path

        for neighbour, cost in graph.get_neighbours(current):
            if neighbour not in visited:
                counter += 1
                nodes_created += 1
                new_h = heuristic(graph, neighbour)
                heapq.heappush(frontier, (new_h, counter, neighbour, path + [neighbour]))

    return None, nodes_created, []