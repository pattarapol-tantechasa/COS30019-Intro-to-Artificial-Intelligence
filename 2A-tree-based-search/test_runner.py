import sys
import os
import re
from pathlib import Path
from graph import parse_file
from algorithms import bfs, dfs
from astar import astar
from iddfs import iddfs
from gbfs import gbfs

ALGORITHMS = {
    'BFS':   bfs,
    'DFS':   dfs,
    'GBFS':  gbfs,
    'ASTAR': astar,
    'IDDFS': iddfs,
}


def _sort_key(path):
    """Extract numeric prefix from filename for natural sorting."""
    match = re.search(r'(\d+)', path.stem)
    return int(match.group(1)) if match else 0


def get_all_test_files():
    """Return all test files in tests/ directory, sorted in natural numeric order."""
    return sorted(Path('tests').glob('*.txt'), key=_sort_key)


def resolve_test_file(test_name):
    """
    Resolve a test name to a file path.
    - If test_name ends in .txt, use as-is
    - Otherwise, check tests/<test_name>.txt
    - If not found, scan tests/ for files starting with <test_name>
    Returns the path if found, else None
    """
    # If it ends in .txt, treat as direct path
    if test_name.endswith('.txt'):
        if os.path.isfile(test_name):
            return test_name
        return None

    # Try tests/<test_name>.txt
    direct_path = f"tests/{test_name}.txt"
    if os.path.isfile(direct_path):
        return direct_path

    # Scan tests/ for files starting with test_name
    tests_dir = Path('tests')
    if tests_dir.exists():
        for file in sorted(tests_dir.glob(f"{test_name}*.txt")):
            return str(file)

    return None


def run_all_tests(method):
    """Run a single algorithm on all test files."""
    algorithm = ALGORITHMS[method]
    test_files = get_all_test_files()

    if not test_files:
        print("No test files found in tests/ directory")
        sys.exit(1)

    print(f"Running {method} on {len(test_files)} test files\n")

    for filepath in test_files:
        graph = parse_file(str(filepath))
        goal, nodes_created, path = algorithm(graph)

        print(f"=== {filepath.name} ===")
        if goal is None:
            print("No solution found.")
        else:
            path_str = ', '.join(str(n) for n in path)
            print(f"{goal} {nodes_created}")
            print(f"[{path_str}]")
        print()


def print_usage():
    """Print usage instructions for both modes."""
    print("Usage:")
    print("  python test_runner.py <test_name>")
    print("    Run all algorithms on a single test file")
    print("  python test_runner.py --algo <METHOD>")
    print("    Run one algorithm on all test files")
    print()
    print("Examples:")
    print("  python test_runner.py PathFinder-test")
    print("  python test_runner.py test5")
    print("  python test_runner.py --algo BFS")
    print("  python test_runner.py --algo ASTAR")
    print()
    print("Available algorithms:", ', '.join(ALGORITHMS.keys()))


def main():
    if len(sys.argv) < 2:
        print_usage()
        sys.exit(1)

    # Check for --algo flag
    if sys.argv[1] == '--algo':
        if len(sys.argv) != 3:
            print("Error: --algo requires a method argument")
            print_usage()
            sys.exit(1)

        method = sys.argv[2].upper()
        if method not in ALGORITHMS:
            print(f"Error: Unknown algorithm '{method}'")
            print("Available algorithms:", ', '.join(ALGORITHMS.keys()))
            sys.exit(1)

        run_all_tests(method)
    else:
        # Test file mode
        test_name = sys.argv[1]
        filepath = resolve_test_file(test_name)

        if not filepath:
            print(f"Error: Could not find test file matching '{test_name}'")
            sys.exit(1)

        print(f"Running all algorithms on {filepath}\n")

        graph = parse_file(filepath)

        for method, algorithm in ALGORITHMS.items():
            goal, nodes_created, path = algorithm(graph)

            print(f"=== {method} ===")
            if goal is None:
                print("No solution found.")
            else:
                path_str = ', '.join(str(n) for n in path)
                print(f"{goal} {nodes_created}")
                print(f"[{path_str}]")
            print()


if __name__ == '__main__':
    main()
