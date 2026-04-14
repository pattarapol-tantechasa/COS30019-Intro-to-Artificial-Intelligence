
# Custom uninformed Algorithm: IDDFS
def iddfs(graph):
    '''
    Iterative Deepening Depth-First Search
    Returns: goal_node, nodes_created, path_list'''

    start = graph.origin
    goals = set(graph.destinations)
    nodes_created_total = 0 # counts every node visited in DLS, accumlated across all depth iterations 

    # Depth-limited DFS helper
    def dls(node, path, depth, explored_set):
        nonlocal nodes_created_total # use outer variable 
        nodes_created_total += 1

        if node in goals:
            return path # found goal
        
        if depth == 0:
            return None # reached depth limit
        
        # Expend neighbours in reverse order to tie-break msaller node IDs first
        for neighbour, cost in reversed(graph.get_neighbours(node)):
            if neighbour not in explored_set:
                explored_set.add(neighbour)
            result = dls(neighbour, path + [neighbour], depth - 1, explored_set)
            if result:
                return result
            explored_set.remove(neighbour) # backtrack
        return None
    
    depth = 0
    while True:
        explored_set = set([start]) # start is always explored for this iteration
        result = dls(start, [start], depth, explored_set)
        if result:
            return result[-1], nodes_created_total, result # goal, nodes_created, path
        depth += 1
