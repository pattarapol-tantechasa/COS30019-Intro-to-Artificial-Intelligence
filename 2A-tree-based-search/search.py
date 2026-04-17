import sys
from graph import parse_file
from bfs import bfs
from dfs import dfs

from astar import astar
from cus1 import iddfs
from gbfs import gbfs
from cus2 import cus2


def format_output(filename, method, goal, nodes_created, path):
    path_str = ', '.join(str(n) for n in path) # fix output. join with comma, add brackets to path_str
    print(f"{filename} {method}")
    print(f"Goal: {goal}\nNode Visited: {nodes_created}\n")
    print(f"[{path_str}]")


def main():
    if len(sys.argv) != 3:
        print("Usage: python search.py <filename> <method>")
        sys.exit(1)

    filename = sys.argv[1]
    method = sys.argv[2].upper()

    graph = parse_file(filename)

    if method == 'BFS':
        goal, nodes_created, path = bfs(graph)
    elif method == 'DFS':
        goal, nodes_created, path = dfs(graph)
    elif method == 'AS':
        goal, nodes_created, path = astar(graph)
    elif method == 'GBFS':
        goal, nodes_created, path = gbfs(graph)
    elif method == 'CUS1':
        goal, nodes_created, path = iddfs(graph)
    elif method == 'CUS2':
        goal, nodes_created, path = cus2(graph)
    else:
        print(f"Unknown method: {method}")
        sys.exit(1)

    if goal is None:
        print(f"{filename} {method}")
        print("No solution found.")
    else:
        format_output(filename, method, goal, nodes_created, path)


if __name__ == '__main__':
    main()
