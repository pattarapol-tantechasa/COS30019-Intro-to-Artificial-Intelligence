from heuristics import heuristic, compute_min_cost_per_unit, SCALE
import heapq  # Using heapq to efficiently always expand the node with the lowest f(n) (priority queue in O(log n) time)


def astar(graph):
    '''A* search. Uses scaled heuristic to ensure distance is always < path cost (admissible)
    Returns: goal_node, nodes_created, path_list'''
    start = graph.origin
    goals = set(graph.destinations)

    # Priority queue: (f, node, path, g)
    frontier = []
    g_costs = {start: 0} # best known cost to each node
    nodes_created = 1

    # Push start node onto the heap
    start_h = heuristic(graph, start) # heuristic function computes scale automatically
    heapq.heappush(frontier, (start_h, start, [start], 0))

    explored = set()

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
                h = heuristic(graph, neighbour)
                new_f = new_g + h
                nodes_created += 1
                heapq.heappush(frontier, (new_f, neighbour, path + [neighbour], new_g))

    return None, nodes_created, []