"""
test_astar_2b.py

15 unit tests for routing/astar.py in the 2B Machine Learning project.

Run from the 2B-Machine-Learning folder:
    python test_astar_2b.py
"""

import sys
import math

sys.path.insert(0, 'routing')
from graph import Graph
from astar import astar



def make_graph(nodes, edges, origin, destinations):

    g = Graph()
    for node_id, x, y in nodes:
        g.add_node(node_id, x, y)
    for from_id, to_id, cost in edges:
        g.add_edge(from_id, to_id, cost)
    g.origin = origin
    g.destinations = destinations
    return g


passed = 0
failed = 0

def check(name, condition, detail=""):
    global passed, failed
    if condition:
        print(f"  PASS  {name}")
        passed += 1
    else:
        print(f"  FAIL  {name}" + (f"  ({detail})" if detail else ""))
        failed += 1


# TC01 Trivial: origin is the destination

print("\nTC01 Origin is the destination")
g = make_graph(
    nodes=[(1, 0, 0)],
    edges=[],
    origin=1,
    destinations=[1],
)
goal, nodes_created, path = astar(g)
check("goal == 1",    goal == 1,         f"got {goal}")
check("path == [1]",  path == [1],       f"got {path}")
check("nodes_created >= 1", nodes_created >= 1)


# TC02 Single direct edge, one hop

print("\nTC02 Single direct edge")
g = make_graph(
    nodes=[(1, 0, 0), (2, 3, 4)],
    edges=[(1, 2, 5)],
    origin=1,
    destinations=[2],
)
goal, nodes_created, path = astar(g)
check("goal == 2",       goal == 2,       f"got {goal}")
check("path == [1, 2]",  path == [1, 2],  f"got {path}")


# TC03 Linear chain of 5 nodes, no branching

print("\nTC03 Linear chain")
g = make_graph(
    nodes=[(i, float(i), 0.0) for i in range(1, 6)],
    edges=[(i, i+1, 1) for i in range(1, 5)],
    origin=1,
    destinations=[5],
)
goal, nodes_created, path = astar(g)
check("goal == 5",                  goal == 5,              f"got {goal}")
check("path == [1,2,3,4,5]",        path == [1,2,3,4,5],    f"got {path}")

# TC04 Two destinations: nearer one should be reached first

print("\nTC04 Two destinations, closer wins")
g = make_graph(
    nodes=[(1, 0, 0), (2, 2, 0), (3, 10, 0)],
    edges=[(1, 2, 2), (1, 3, 10)],
    origin=1,
    destinations=[2, 3],
)
goal, nodes_created, path = astar(g)
check("goal == 2",         goal == 2,       f"got {goal}")
check("path ends at 2",    path[-1] == 2,   f"got {path}")


# TC05 Expensive direct edge: cheap multi-hop should win

print("\nTC05 Cheap 3-hop beats expensive direct edge")
# Direct 1->2 costs 100. Via 1->3->4->2 costs 6.
g = make_graph(
    nodes=[(1, 0, 0), (2, 5, 0), (3, 2, 0), (4, 4, 0)],
    edges=[(1, 2, 100), (1, 3, 2), (3, 4, 2), (4, 2, 2)],
    origin=1,
    destinations=[2],
)
goal, nodes_created, path = astar(g)
check("goal == 2",                  goal == 2,               f"got {goal}")
check("path == [1,3,4,2]",          path == [1,3,4,2],       f"got {path}")

# TC06 No path: goal is a disconnected node

print("\nTC06 No solution (isolated goal)")
g = make_graph(
    nodes=[(1, 0, 0), (2, 1, 0), (3, 5, 5)],
    edges=[(1, 2, 1)],
    origin=1,
    destinations=[3],
)
goal, nodes_created, path = astar(g)
check("goal is None",  goal is None,  f"got {goal}")
check("path is []",    path == [],    f"got {path}")


# TC07 Directed edges only point AWAY from origin: no return path

print("\nTC07 Directed one-way: goal unreachable from origin")
g = make_graph(
    nodes=[(1, 0, 0), (2, 3, 0), (3, 6, 0)],
    edges=[(1, 2, 3), (2, 3, 3)],   # forward only
    origin=3,
    destinations=[1],
)
goal, nodes_created, path = astar(g)
check("goal is None",  goal is None,  f"got {goal}")


# TC08 Diamond graph: two paths, cheaper one selected

print("\nTC08 Diamond graph, cheaper path wins")
# 1->2->4 costs 6; 1->3->4 costs 13
g = make_graph(
    nodes=[(1, 0, 2), (2, 2, 4), (3, 2, 0), (4, 4, 2)],
    edges=[(1, 2, 3), (1, 3, 3), (2, 4, 3), (3, 4, 10)],
    origin=1,
    destinations=[4],
)
goal, nodes_created, path = astar(g)
check("goal == 4",            goal == 4,           f"got {goal}")
check("path == [1,2,4]",      path == [1,2,4],     f"got {path}")


# TC09 3x3 grid, top-left to bottom-right

print("\nTC09 3x3 grid")
#  1-2-3
#  4-5-6
#  7-8-9
g = make_graph(
    nodes=[
        (1,0,2),(2,1,2),(3,2,2),
        (4,0,1),(5,1,1),(6,2,1),
        (7,0,0),(8,1,0),(9,2,0),
    ],
    edges=[
        (1,2,1),(2,3,1),(4,5,1),(5,6,1),(7,8,1),(8,9,1),  # rows
        (1,4,1),(4,7,1),(2,5,1),(5,8,1),(3,6,1),(6,9,1),  # cols
    ],
    origin=1,
    destinations=[9],
)
goal, nodes_created, path = astar(g)
check("goal == 9",              goal == 9,                     f"got {goal}")
check("path length == 5",       len(path) == 5,                f"got {path}")
check("path starts at 1",       path[0] == 1,                  f"got {path}")
check("path ends at 9",         path[-1] == 9,                 f"got {path}")


# TC10 Graph with a cycle: A* must not loop or revisit expensively

print("\nTC10 Cycle: no infinite loop, optimal path found")
g = make_graph(
    nodes=[(1, 0, 0), (2, 2, 0), (3, 4, 0), (4, 2, 2)],
    edges=[(1,2,2),(2,3,2),(3,4,2),(4,2,5),(2,4,5),(1,4,10)],
    origin=1,
    destinations=[3],
)
goal, nodes_created, path = astar(g)
check("goal == 3",         goal == 3,       f"got {goal}")
check("path == [1,2,3]",   path == [1,2,3], f"got {path}")


# TC11 Two goals at equal cost: lower node ID wins (tie-breaking)

print("\nTC11 Equal-cost goals, lower ID wins")
g = make_graph(
    nodes=[(1, 0, 0), (2, 3, 0), (3, 0, 3)],
    edges=[(1, 2, 5), (1, 3, 5)],
    origin=1,
    destinations=[2, 3],
)
goal, nodes_created, path = astar(g)
check("goal == 2 (lower ID)",  goal == 2,  f"got {goal}")


# TC12 Extreme weight contrast: 50 direct vs. 3-hop at cost 3

print("\nTC12 Extreme weights: cheap multi-hop vs costly direct")
g = make_graph(
    nodes=[(1, 0, 0), (2, 4, 0), (3, 2, 0), (4, 3, 0)],
    edges=[(1, 2, 50), (1, 3, 1), (3, 4, 1), (4, 2, 1)],
    origin=1,
    destinations=[2],
)
goal, nodes_created, path = astar(g)
check("goal == 2",             goal == 2,           f"got {goal}")
check("path == [1,3,4,2]",     path == [1,3,4,2],   f"got {path}")


# TC13 Star topology: origin at center, 6 leaf destinations

print("\nTC13 Star topology: direct radial edge to goal")
g = make_graph(
    nodes=[(1,5,5),(2,5,9),(3,9,7),(4,9,3),(5,5,1),(6,1,3),(7,1,7)],
    edges=[(1,2,4),(1,3,5),(1,4,5),(1,5,4),(1,6,5),(1,7,5)],
    origin=1,
    destinations=[5],
)
goal, nodes_created, path = astar(g)
check("goal == 5",         goal == 5,       f"got {goal}")
check("path == [1,5]",     path == [1,5],   f"got {path}")


# TC14 Forced long route: 100-cost direct vs 5-hop at cost 5

print("\nTC14 Long route cheaper than expensive direct edge")
g = make_graph(
    nodes=[(i, float(i), 0.0) for i in range(1, 7)],
    edges=[(1,6,100),(1,2,1),(2,3,1),(3,4,1),(4,5,1),(5,6,1)],
    origin=1,
    destinations=[6],
)
goal, nodes_created, path = astar(g)
check("goal == 6",                  goal == 6,                  f"got {goal}")
check("path == [1,2,3,4,5,6]",      path == [1,2,3,4,5,6],     f"got {path}")


# TC15 Dead-end branch: A* must ignore it and find correct path

print("\nTC15 Dead-end branch, correct path found via backtracking")
# 1->2->3->5 (cost 6) is valid. 2->4 is a dead end (4 has no outgoing edges to goal)
g = make_graph(
    nodes=[(1,0,0),(2,2,0),(3,4,0),(4,2,3),(5,6,0)],
    edges=[(1,2,2),(2,3,2),(2,4,5),(3,5,2)],
    origin=1,
    destinations=[5],
)
goal, nodes_created, path = astar(g)
check("goal == 5",              goal == 5,           f"got {goal}")
check("path == [1,2,3,5]",      path == [1,2,3,5],   f"got {path}")


# Summary

total = passed + failed
print(f"\n")
print(f"Results: {passed}/{total} passed    {failed} failed")
sys.exit(0 if failed == 0 else 1)
