import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

# --- STEP 0: DATA SETUP ---
# All coordinates: First 27 = EO, rest = TW
coords = np.array([
    (211,63), (685,219), (421,269), (157,361), (178,371), (513,440), (754,463),
    (444,486), (673,525), (504,540), (396,567), (344,606), (117,486), (868,644),
    (916,652), (1047,631), (1184,781), (136,656), (677,702), (868,758), (881,816),
    (571,783), (450,875), (147,877), (196,931), (635,868), (698,939),
    (140,161), (954,353), (704,446), (1203,569), (752,665), (902,758),
    (296,606), (36,667), (533,893), (879, 948)
])
eo_coords = coords[:27]
tw_coords = coords[27:]
tw_start_index = 27

# --- STEP 1: GRAPH AND MST CONSTRUCTION ---
# Build the complete graph G with all nodes and weighted edges between Oculi
G = nx.Graph()
for i, (x, y) in enumerate(eo_coords):
    G.add_node(i, pos=(x, y), type='eo')
for i, (x, y) in enumerate(tw_coords):
    G.add_node(tw_start_index + i, pos=(x, y), type='tw')

# Add weighted edges between all Oculus nodes
for i in range(len(eo_coords)):
    for j in range(i + 1, len(eo_coords)):
        pos_i, pos_j = G.nodes[i]['pos'], G.nodes[j]['pos']
        distance = np.hypot(pos_i[0] - pos_j[0], pos_i[1] - pos_j[1])
        G.add_edge(i, j, weight=distance)

# Compute the Minimum Spanning Tree (MST) to act as our "road map"
mst = nx.minimum_spanning_tree(G.subgraph(range(len(eo_coords))), weight='weight', algorithm='prim')


# --- STEP 2: HELPER FUNCTIONS FOR HEURISTICS ---
def precompute_nearest_tw_data(G, num_eo, num_tw, tw_start_idx):
    """Calculates the distance from each EO to its nearest TW and stores it."""
    nearest_tw_data = {}
    for i in range(num_eo):
        min_dist = float('inf')
        closest_tw = -1
        for j in range(num_tw):
            tw_node = tw_start_idx + j
            dist = np.hypot(G.nodes[i]['pos'][0] - G.nodes[tw_node]['pos'][0],
                            G.nodes[i]['pos'][1] - G.nodes[tw_node]['pos'][1])
            if dist < min_dist:
                min_dist = dist
                closest_tw = tw_node
        nearest_tw_data[i] = {'dist': min_dist, 'id': closest_tw}
    return nearest_tw_data

def get_branch_teleport_score(branch_root, T, nearest_tw_data):
    """
    Calculates the 'teleport inefficiency' of a branch.
    A higher score means the branch is further from the TW network.
    """
    subtree_nodes = nx.descendants(T, branch_root)
    subtree_nodes.add(branch_root)
    
    if not subtree_nodes:
        return 0

    total_dist = sum(nearest_tw_data[node]['dist'] for node in subtree_nodes if node in nearest_tw_data)
    return total_dist / len(subtree_nodes) if subtree_nodes else 0

# --- STEP 3: PHASE 1 - DETERMINE OCULUS VISIT SEQUENCE VIA DFS ---
print("--- Phase 1: Determining Optimal Oculus Sequence via Teleport-Aware DFS ---")

# Pre-computation for efficiency
nearest_tw_data = precompute_nearest_tw_data(G, len(eo_coords), len(tw_coords), tw_start_index)

# Data structures for the DFS traversal
source_node = 0
T = nx.dfs_tree(mst, source=source_node) # Directed tree for parent/child logic
stack = [source_node]
visited_oculi = set()
oculus_visit_sequence = [] # The ordered list of Oculi to visit

while stack:
    current_node = stack.pop()
    if current_node in visited_oculi:
        continue

    visited_oculi.add(current_node)
    oculus_visit_sequence.append(current_node)

    children = [child for child in T.neighbors(current_node) if child not in visited_oculi]
    if children:
        if len(children) > 1:
            scored_children = []
            for child in children:
                score = get_branch_teleport_score(child, T, nearest_tw_data)
                scored_children.append((child, score))
            scored_children.sort(key=lambda x: x[1], reverse=True)
            children = [child for child, score in scored_children]
        
        for child in children:
            stack.append(child)

print("Optimal Visit Sequence Found:", oculus_visit_sequence)


# --- STEP 4: PHASE 2 - BUILD FINAL PATH WITH TELEPORT DECISIONS ---
print("\n--- Phase 2: Building Final Path with Teleport-or-Walk Decisions ---")

final_path = [oculus_visit_sequence[0]] # Start the path with the first oculus
total_cost = 0.0
path_description = [f"Start at Node {final_path[0]}."]

# Iterate through the sequence to decide travel method between each pair
for i in range(len(oculus_visit_sequence) - 1):
    from_node = oculus_visit_sequence[i]
    to_node = oculus_visit_sequence[i+1]

    # Option A: Walk directly
    walk_cost = G[from_node][to_node]['weight']

    # Option B: Teleport from current location to the best TW for the destination.
    # The cost is only the walk from that destination TW to the target Oculus.
    tw_data_to = nearest_tw_data[to_node]
    teleport_cost = tw_data_to['dist']

    if walk_cost <= teleport_cost:
        # Decision: Walking is better or equal
        total_cost += walk_cost
        path_description.append(f"Travel from {from_node} -> {to_node}. Method: Walk (Cost: {walk_cost:.1f})")
        final_path.append(to_node)
    else:
        # Decision: Teleporting is better
        total_cost += teleport_cost
        tw_to = tw_data_to['id']
        path_description.append(f"Travel from {from_node} -> {to_node}. Method: Teleport to TW {tw_to} (Cost: {teleport_cost:.1f})")
        
        # Add the destination waypoint and the oculus to the path
        if final_path[-1] != tw_to:
            final_path.append(tw_to)
        final_path.append(to_node)


# --- STEP 5: RESULTS AND VISUALIZATION ---
print("\n--- Traversal Complete ---")
for step in path_description:
    print(step)

print("\n--- Final Path (including Teleport Waypoints) ---")
path_with_labels = [f"TW-{n}" if n >= tw_start_index else str(n) for n in final_path]
print(" -> ".join(path_with_labels))
print(f"\nTotal Estimated Cost: {total_cost:.2f}")

# Plotting
pos = nx.get_node_attributes(G, 'pos')
plt.figure(figsize=(16, 16))

# Nodes and Labels
nx.draw_networkx_nodes(G, pos, nodelist=range(len(eo_coords)), node_color='skyblue', label='Elemental Oculus')
nx.draw_networkx_nodes(G, pos, nodelist=range(tw_start_index, len(coords)), node_color='mediumpurple', label='Teleport Waypoint')
nx.draw_networkx_labels(G, pos, font_size=12, font_color='black')

# MST "Road Map"
nx.draw_networkx_edges(mst, pos, edge_color='gray', width=1.5, style='dashed', label='MST Base')

# --- MODIFIED VISUALIZATION FOR THE FINAL PATH ---
# Create a directed graph for the final path to draw arrows
path_graph = nx.DiGraph()
path_graph.add_edges_from(list(zip(final_path, final_path[1:])))

# Draw the final path with a red color and arrows
nx.draw_networkx_edges(path_graph, pos,
                       edge_color='red',          # Highlight color is now red
                       width=2.5,                 # Make the path line thicker
                       alpha=0.9,
                       arrows=True,               # Enable arrows to show direction
                       arrowstyle='-|>',          # Define the arrow head style
                       arrowsize=20,              # Define the arrow head size
                       label='Final HUTAO-TSP Path')

plt.title("HUTAO-TSP: Final Path with Corrected Teleport Model", fontsize=16)
plt.legend()
plt.gca().invert_yaxis()
plt.axis('equal')
plt.axis('off')
plt.tight_layout()
plt.show()
