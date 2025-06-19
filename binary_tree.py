import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from collections import deque, defaultdict

# All coordinates: First 27 = EO, rest = TW
coords = np.array([
    (211,63), (685,219), (421,269), (157,361), (178,371), (513,440), (754,463),
    (444,486), (673,525), (504,540), (396,567), (344,606), (117,486), (868,644),
    (916,652), (1047,631), (1184,781), (136,656), (677,702), (868,758), (881,816),
    (571,783), (450,875), (147,877), (196,931), (635,868), (698,939),
    (140,161), (954,353), (704,446), (1203,569), (752,665), (902,758),
    (296,606), (36,667), (533,893)
])

# Split coordinates
eo_coords = coords[:27]
tw_coords = coords[27:]

# Step 1: Build the full graph
G = nx.Graph()

# Add EO nodes with type
for i, (x, y) in enumerate(eo_coords):
    G.add_node(i, pos=(x, y), type='eo')

# Add TW nodes with type
offset = len(eo_coords)
for i, (x, y) in enumerate(tw_coords):
    G.add_node(offset + i, pos=(x, y), type='tw')

# Compute pairwise distances for EO nodes using NumPy
n = len(eo_coords)
x, y = eo_coords[:, 0], eo_coords[:, 1]
dx = x[:, np.newaxis] - x
dy = y[:, np.newaxis] - y
distances = np.hypot(dx, dy)

# Add edges between EO nodes only
for i in range(n):
    for j in range(i + 1, n):
        G.add_edge(i, j, weight=distances[i, j])

# Compute MST for EO nodes using Prim's algorithm
mst = nx.minimum_spanning_tree(G.subgraph
        (range(len(eo_coords))), 
        weight='weight', algorithm='prim')

# --- Converting MST to binary tree ---
# Choose node 0 as root
root = 0
binary_tree = nx.DiGraph()  # Directed graph to show direction (parent -> child)
visited = set()
child_count = defaultdict(int)

def dfs_binary_tree(curr, parent=None):
    visited.add(curr)
    if parent is not None:
        binary_tree.add_edge(parent, curr)

    children = [nbr for nbr in mst.neighbors(curr) if nbr not in visited]
    
    # At most 2 children (biner)
    for i, child in enumerate(children[:2]):
        dfs_binary_tree(child, curr)

dfs_binary_tree(root)

# --- Draw binary tree vertically ---
# Helper function for drawing binary tree
def hierarchy_pos(G, root=None, width=1.0, vert_gap=0.2, vert_loc=0, xcenter=0.5):
    '''
    If the graph is a tree, this will return the positions to plot it in a hierarchical layout.
    '''
    def _hierarchy_pos(G, root, leftmost, width, vert_gap, vert_loc, pos, parent=None):
        children = list(G.successors(root))
        if not children:
            pos[root] = (leftmost[0], vert_loc)
            leftmost[0] += width
        else:
            start = leftmost[0]
            for child in children:
                pos = _hierarchy_pos(G, child, leftmost, width, vert_gap, vert_loc - vert_gap, pos, root)
            mid = (start + leftmost[0] - width) / 2
            pos[root] = (mid, vert_loc)
        return pos

    if root is None:
        root = next(iter(nx.topological_sort(G)))  # Choose the first root
    return _hierarchy_pos(G, root, [0], width, vert_gap, vert_loc, {})

# Vertical layout position
pos_bt = hierarchy_pos(binary_tree, root)

plt.figure(figsize=(14, 10))
nx.draw(binary_tree, pos_bt, with_labels=True, node_color='skyblue', node_size=1200, font_size=18, arrows=True)
plt.title("Binary Tree dari MST (EO Nodes)")
plt.axis('off')
plt.tight_layout()
plt.show()