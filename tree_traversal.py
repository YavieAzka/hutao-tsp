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

# Graph Construction
G = nx.Graph()
for i, (x, y) in enumerate(eo_coords):
    G.add_node(i, pos=(x, y), type='eo')
for i, (x, y) in enumerate(tw_coords):
    G.add_node(len(eo_coords) + i, pos=(x, y), type='tw')

for i in range(len(eo_coords)):
    for j in range(i + 1, len(eo_coords)):
        pos_i = G.nodes[i]['pos']
        pos_j = G.nodes[j]['pos']
        distance = np.hypot(pos_i[0] - pos_j[0], pos_i[1] - pos_j[1])
        G.add_edge(i, j, weight=distance)

# Compute MST
mst = nx.minimum_spanning_tree(G.subgraph(range(len(eo_coords))), weight='weight', algorithm='prim')

# Heuristic Jump Calculation
source_node = 0
T = nx.dfs_tree(mst, source=source_node)
leaves = [node for node, degree in mst.degree() if degree == 1]
heuristic_jumps = {}

for leaf in leaves:
    if leaf == source_node:
        continue
    nodes_to_exclude = nx.ancestors(T, leaf)
    nodes_to_exclude.add(leaf)
    candidate_nodes = set(range(len(eo_coords)))
    candidate_nodes -= nodes_to_exclude
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


# #####################################################################
# ## HUTAO-TSP TRAVERSAL SIMULATION (WITH FINAL FIX)                 ##
# #####################################################################
print("\n--- Starting HUTAO-TSP Traversal Simulation ---")
unvisited_nodes = set(range(len(eo_coords)))
final_path = []
total_distance = 0.0
path_description = []

def get_best_teleport_cost(current_node_idx, target_nodes_indices, G):
    if not target_nodes_indices:
        return float('inf'), -1
    current_pos = G.nodes[current_node_idx]['pos']
    tw_indices = range(len(eo_coords), len(coords))
    cost_to_nearest_tw = float('inf')
    for tw_idx in tw_indices:
        tw_pos = G.nodes[tw_idx]['pos']
        dist = np.hypot(current_pos[0] - tw_pos[0], current_pos[1] - tw_pos[1])
        if dist < cost_to_nearest_tw:
            cost_to_nearest_tw = dist
    best_overall_cost = float('inf')
    best_target_node = -1
    for target_idx in target_nodes_indices:
        target_pos = G.nodes[target_idx]['pos']
        cost_from_tw_to_target = float('inf')
        for tw_idx in tw_indices:
            tw_pos = G.nodes[tw_idx]['pos']
            dist = np.hypot(target_pos[0] - tw_pos[0], target_pos[1] - tw_pos[1])
            if dist < cost_from_tw_to_target:
                cost_from_tw_to_target = dist
        total_teleport_cost = cost_to_nearest_tw + cost_from_tw_to_target
        if total_teleport_cost < best_overall_cost:
            best_overall_cost = total_teleport_cost
            best_target_node = target_idx
    return best_overall_cost, best_target_node

current_node = source_node
final_path.append(current_node)
unvisited_nodes.remove(current_node)
path_description.append(f"Start at Node {current_node}.")

while unvisited_nodes:
    mst_neighbors = list(mst.neighbors(current_node))
    unvisited_mst_neighbors = [n for n in mst_neighbors if n in unvisited_nodes]
    if unvisited_mst_neighbors:
        next_node = unvisited_mst_neighbors[0]
        distance = G[current_node][next_node]['weight']
        path_description.append(f"Traverse MST from {current_node} to {next_node} (Cost: {distance:.1f}).")
        current_node = next_node
        total_distance += distance
        final_path.append(current_node)
        unvisited_nodes.remove(current_node)
    else:
        path_description.append(f"At dead end {current_node}. Evaluating options...")
        
        # Option A: Heuristic Jump
        jump_target, jump_cost = heuristic_jumps.get(current_node, (None, float('inf')))
        
        # --- FINAL FIX: Check if the jump target has already been visited ---
        if jump_target is not None and jump_target not in unvisited_nodes:
            jump_cost = float('inf')  # Invalidate this option
            path_description.append(f"  -> Note: Pre-calculated jump target {jump_target} already visited. Discarding jump option.")

        # Option B: Best Teleport Jump
        teleport_cost, teleport_target = get_best_teleport_cost(current_node, unvisited_nodes, G)
        
        # Check if there are any options left
        if jump_cost == float('inf') and teleport_cost == float('inf'):
             path_description.append("  -> No valid moves left. This should not happen if graph is connected.")
             break

        # Compare and decide
        if jump_cost <= teleport_cost:
            next_node = jump_target
            distance = jump_cost
            path_description.append(f"  -> Decision: Use Heuristic Jump to {next_node} (Cost: {distance:.1f}).")
        else:
            next_node = teleport_target
            distance = teleport_cost
            path_description.append(f"  -> Decision: Teleport to {next_node} (Cost: {distance:.1f}).")
        
        current_node = next_node
        total_distance += distance
        final_path.append(current_node)
        unvisited_nodes.remove(current_node)

print("\n--- Traversal Complete ---")
for step in path_description:
    print(step)
print("\n--- Final Path ---")
print(" -> ".join(map(str, final_path)))
print(f"\nTotal Estimated Distance: {total_distance:.2f}")

# Plotting
pos = nx.get_node_attributes(G, 'pos')
plt.figure(figsize=(16, 16))
nx.draw_networkx_nodes(G, pos, nodelist=range(len(eo_coords)), node_color='skyblue', label='Elemental Oculus')
nx.draw_networkx_nodes(G, pos, nodelist=range(len(eo_coords), len(coords)), node_color='mediumpurple', label='Teleport Waypoint')
nx.draw_networkx_edges(mst, pos, edge_color='black', width=1.5, label='MST Path')
final_path_edges = list(zip(final_path, final_path[1:]))
nx.draw_networkx_edges(G, pos, edgelist=final_path_edges, 
                       edge_color='blue', 
                       width=3,
                       alpha=0.7,
                       label='Final HUTAO-TSP Path')
nx.draw_networkx_labels(G, pos, font_size=12, font_color='black')
plt.title("HUTAO-TSP: Final Simulated Path", fontsize=16)
plt.legend()
plt.gca().invert_yaxis()
plt.axis('equal')
plt.axis('off')
plt.tight_layout()
plt.show()