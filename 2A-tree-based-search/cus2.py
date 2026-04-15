
# Custom uninformed Algorithm (CUS2): IDDFS (tree based with infinite state check)
def iddfs(graph):
    start = graph.origin
    goals = set(graph.destinations)

    nodes_created_total = 0

    # dfs helper function. search until a limit is reached
    def dls(node, path, depth):
        nonlocal nodes_created_total
        nodes_created_total += 1

        # goal test
        if node in goals:
            return path

        # depth limit, stops recursion
        if depth == 0:
            return None

        # expand children
        for neighbour, cost in graph.get_neighbours(node):

            # cycle prevention (path-based, not global)
            # branches cannot interfere with each other, prevents crash in the event of a no solution. 
            if neighbour in path:
                continue

            result = dls(
                neighbour,
                path + [neighbour],
                depth - 1
            )

            if result is not None:
                return result

        return None

    # safe maximum depth (prevents infinite looping in graphs)
    # if no solution exists by depth N, it does not exist
    max_depth = len(graph.nodes)

    # IDDFS (Iterative Deepening) Loop
    # search space is expanded gradually and finitely.
    # Run DFS multiple times and increase depth limit each time
    for depth in range(max_depth + 1):
        result = dls(start, [start], depth)

        if result is not None:
            return result[-1], nodes_created_total, result

    # no solution found
    return None, nodes_created_total, []