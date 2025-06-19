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
    (296,606), (36,667), (533,893), (879, 948)
])

# Split coordinates
eo_coords = coords[:27]
tw_coords = coords[27:]

# Step 1: Build the full graph
G = nx.Graph()
for i, (x, y) in enumerate(eo_coords):
    G.add_node(i, pos=(x, y), type='eo')
for i, (x, y) in enumerate(tw_coords):
    G.add_node(len(eo_coords) + i, pos=(x, y), type='tw')

# Create a subgraph with only EO nodes for MST calculation
eo_graph = G.subgraph(range(len(eo_coords))).copy()
for i in range(len(eo_coords)):
    for j in range(i + 1, len(eo_coords)):
        pos_i = eo_graph.nodes[i]['pos']
        pos_j = eo_graph.nodes[j]['pos']
        distance = np.hypot(pos_i[0] - pos_j[0], pos_i[1] - pos_j[1])
        eo_graph.add_edge(i, j, weight=distance)

# Compute MST
mst = nx.minimum_spanning_tree(eo_graph, weight='weight', algorithm='prim')

# FIND HEURISTIC JUMPS VIA ANCESTOR EXCLUSION ##

source_node = 0
T = nx.dfs_tree(mst, source=source_node)
leaves = [node for node, degree in mst.degree() if degree == 1]
heuristic_jumps = {}

print("Calculating Heuristic Jumps (Ancestor Exclusion Logic)...")

for leaf in leaves:
    if leaf == source_node:
        continue

    # --- LOGIC ---
    # The "branch" to exclude is simply all ancestors of the leaf.
    # This prevents any jump that backtracks along the path taken.
    nodes_to_exclude = nx.ancestors(T, leaf)
    # Also exclude the leaf itself from being a target.
    nodes_to_exclude.add(leaf)

    # Candidates for the jump are all other EO nodes
    candidate_nodes = set(range(len(eo_coords)))
    candidate_nodes -= nodes_to_exclude
    
    # Find the closest node among the valid candidates
    min_dist = float('inf')
    best_target = -1
    leaf_pos = G.nodes[leaf]['pos']

    for target in candidate_nodes:
        target_pos = G.nodes[target]['pos']
        dist = np.hypot(leaf_pos[0] - target_pos[0], leaf_pos[1] - target_pos[1])
        if dist < min_dist:
            min_dist = dist
            best_target = target
            
    if best_target != -1:
        heuristic_jumps[leaf] = (best_target, min_dist)

# Print the results
print("\n--- Heuristic Jump Targets ---")
for leaf, (target, dist) in heuristic_jumps.items():
    print(f"- From Leaf {leaf}: Jump to Node {target} (Distance: {dist:.1f})")


# PLOTTING
pos = nx.get_node_attributes(G, 'pos')
plt.figure(figsize=(16, 16))
# Nodes
nx.draw_networkx_nodes(G, pos, nodelist=range(len(eo_coords)), node_color='skyblue', label='Elemental Oculus')
nx.draw_networkx_nodes(G, pos, nodelist=range(len(eo_coords), len(coords)), node_color='mediumpurple', label='Teleport Waypoint')
# Edges
nx.draw_networkx_edges(mst, pos, edge_color='black', width=1.5, label='MST Path')
jump_edges = [(leaf, target) for leaf, (target, dist) in heuristic_jumps.items()]
nx.draw_networkx_edges(G, pos, edgelist=jump_edges, 
                       edge_color='green', 
                       style='dashed', 
                       width=2, 
                       label='Heuristic Jump')
# Labels
nx.draw_networkx_labels(G, pos, font_size=12, font_color='black')
plt.title("HUTAO-TSP: MST with Final Corrected Heuristic Jumps", fontsize=16)
plt.legend()
plt.gca().invert_yaxis()
plt.axis('equal')
plt.axis('off')
plt.tight_layout()
plt.show()  