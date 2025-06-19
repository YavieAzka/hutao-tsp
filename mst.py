import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

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

# Plot all nodes, highlight EO vs TW
pos = nx.get_node_attributes(G, 'pos')
node_types = nx.get_node_attributes(G, 'type')

# Split EO and TW for color coding
eo_nodes = [n for n in G.nodes if node_types[n] == 'eo']
tw_nodes = [n for n in G.nodes if node_types[n] == 'tw']

plt.figure(figsize=(12, 12))

# Draw EO and TW nodes, and MST edges
nx.draw_networkx_nodes(G, pos, nodelist=eo_nodes, node_color='skyblue', label='Elemental Oculus')
nx.draw_networkx_nodes(G, pos, nodelist=tw_nodes, node_color='mediumpurple', label='Teleport Waypoint')
nx.draw_networkx_edges(mst, pos, edge_color='black')

# Draw edge weights
edge_labels = {(u, v): f"{d['weight']:.1f}" for u, v, d in mst.edges(data=True)}
nx.draw_networkx_edge_labels(mst, pos, edge_labels=edge_labels, font_size=12, font_color='red')

# Draw node labels
nx.draw_networkx_labels(G, pos, font_size=12, font_color='black')

plt.title("MST with Elemental Oculus and Teleport Waypoints")
plt.legend()
plt.gca().invert_yaxis()
plt.axis('equal')
plt.axis('off')
plt.tight_layout()
plt.show()