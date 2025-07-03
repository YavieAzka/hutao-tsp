# HUTAO-TSP: A Teleport-Aware Heuristic for the Traveling Salesman Problem

![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)
![Libraries](https://img.shields.io/badge/Libraries-NumPy_|_NetworkX_|_Matplotlib-orange.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

**Author:** Yavie Azka Putra Araly (13524077)  
**Course:** IF1220 Discrete Mathematics  
**Institution:** Institut Teknologi Bandung (ITB)

---

## Overview

**HUTAO-TSP** (Hamiltonian Undirected Teleportation Aware Open Traveling Salesman Problem) is a novel heuristic algorithm designed to approximate an optimal route for a specialized variant of the Traveling Salesman Problem (TSP). The project is inspired by and applied to the practical challenge of collecting all "Elemental Oculi" in a given region of the open-world game *Genshin Impact*.

The core challenge is that the graph contains not only the nodes to be visited (Oculi) but also a network of "Teleport Waypoints" which allow for instantaneous, zero-cost travel between them. This fundamentally changes the cost-benefit analysis of any given path.

This repository contains the Python implementation of the HUTAO-TSP algorithm, which was developed as the final project for the IF1220 Discrete Mathematics course.


---

## The HUTAO-TSP Algorithm

Given that the TSP is NP-hard, HUTAO-TSP provides a computationally efficient, polynomial-time approximation. It operates in a two-phase framework to first determine a strategic visitation order and then construct a detailed travel path.

### Phase 1: Optimal Sequence Determination via Teleport-Aware DFS

The algorithm's primary innovation is how it decides the *order* in which to visit the Oculi.

1.  **MST Backbone:** A Minimum Spanning Tree (MST) is constructed from all Oculus nodes to create an efficient "road map" that connects all required destinations.
2.  **Branch Prioritization:** A Depth-First Search (DFS) traverses this MST. At any junction with multiple branches, the algorithm makes a strategic choice. It calculates a **"Teleport Inefficiency Score"** for each branch—the average distance from the nodes in that branch to the nearest Teleport Waypoint.
3.  **"Hardest First" Strategy:** The algorithm prioritizes exploring the branch with the *highest* score first. By tackling the most geographically isolated part of the map first, it saves teleport-friendly branches as efficient "escape routes" for later, minimizing costly cross-map travel.

The output of this phase is a strategically ordered list of Oculi to visit.

### Phase 2: Path Construction with Teleport-or-Walk Decisions

Using the sequence from Phase 1 as a blueprint, this phase builds the final path. For each segment (from Oculus A to Oculus B), it makes a clear decision:

1.  **`walk_cost`:** The direct Euclidean distance from A to B.
2.  **`teleport_cost`:** The Euclidean distance from the Teleport Waypoint nearest to B, to B itself. This accurately models the in-game mechanic where one can teleport instantly from anywhere to a waypoint, with the only travel cost being the final walk to the destination.
3.  **Decision:** The algorithm compares the two costs and chooses the cheaper option, constructing a final path that includes the specific Teleport Waypoints used.

---

## Getting Started

### Prerequisites

The project requires Python 3.9+ and the following libraries:
- `numpy`
- `networkx`
- `matplotlib`

You can install them via pip:
```
pip install numpy networkx matplotlib
```

Running the Algorithm
Clone the repository:
```
git clone [https://github.com/](https://github.com/)YavieAzka/HUTAO-TSP.git
cd HUTAO-TSP
```

Run the main script:
```
python hutao-tsp.py
```

The script will execute the algorithm and display the final path visualization using Matplotlib, along with a detailed step-by-step log of its decisions in the console.
### Complexity Analysis
The time complexity of the HUTAO-TSP algorithm is O(N² + N×M), where N is the number of Oculi and M is the number of Teleport Waypoints. This polynomial-time complexity makes it a highly efficient and scalable solution compared to the exponential time required for exact TSP solvers.
O(N²): Dominates during the MST construction on a dense graph of Oculi.
O(N×M): Arises from the pre-computation step that finds the nearest Teleport Waypoint for each Oculus.
### Future Work
- 3D Coordinate System: Incorporate the Z-axis (elevation) to account for verticality in the game world.
- *A Pathfinding:** Replace Euclidean distance with a more realistic cost function based on A* pathfinding that considers terrain and obstacles.
- Generalization: Adapt the core heuristic for other real-world logistical problems with fast-travel networks (e.g., public transit systems).
### Acknowledgements
This project was completed as part of the IF1220 Discrete Mathematics course at the Institut Teknologi Bandung. 
