class Graph:
    def __init__(self):
        self.nodes = {}       # { node_id: (x, y) }
        self.edges = {}       # { node_id: [(neighbour_id, cost), ...] }
        self.origin = None
        self.destinations = []

    def add_node(self, node_id, x, y):
        self.nodes[node_id] = (x, y)
        if node_id not in self.edges:
            self.edges[node_id] = []

    def add_edge(self, from_node, to_node, cost):
        # Directed edge — only adds one direction
        self.edges[from_node].append((to_node, cost))

    def get_neighbours(self, node_id):
        # Returns list of (neighbour_id, cost), sorted by node number
        # The assignment requires tie-breaking by SMALLER node number first
        return sorted(self.edges.get(node_id, []), key=lambda x: x[0])


def parse_file(filepath):
    graph = Graph()
    section = None

    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            if line == 'Nodes:':
                section = 'nodes'
            elif line == 'Edges:':
                section = 'edges'
            elif line == 'Origin:':
                section = 'origin'
            elif line == 'Destinations:':
                section = 'destinations'

            elif section == 'nodes':
                # Format: "1: (4,1)"
                node_id, coords = line.split(':')
                node_id = int(node_id.strip())
                coords = coords.strip().strip('()')
                x, y = [int(c.strip()) for c in coords.split(',')]
                graph.add_node(node_id, x, y)

            elif section == 'edges':
                # Format: "(2,1): 4"
                parts = line.split(':')
                nodes_part = parts[0].strip().strip('()')
                cost = int(parts[1].strip())
                from_node, to_node = [int(n.strip()) for n in nodes_part.split(',')]
                graph.add_edge(from_node, to_node, cost)

            elif section == 'origin':
                graph.origin = int(line)

            elif section == 'destinations':
                # Format: "5; 4"
                graph.destinations = [int(d.strip()) for d in line.split(';')]

    return graph