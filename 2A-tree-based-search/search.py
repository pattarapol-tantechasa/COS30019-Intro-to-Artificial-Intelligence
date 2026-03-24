import sys
from graph import parse_file
from algorithms import bfs, dfs


def format_output(filename, method, goal, nodes_created, path):
    path_str = ' -> '.join(str(n) for n in path)
    print(f"{filename} {method}")
    print(f"{goal} {nodes_created}")
    print(path_str)


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