# Author: Juan Ignacio Doval Roque
# Student ID: 10752534
#
# A* (A-Star) Path Planning Algorithm
# Based on: Hart, Nilsson & Raphael, "A Formal Basis for the Heuristic Determination
# of Minimum Cost Paths", IEEE Transactions on Systems Science and Cybernetics, 1968.
#
# THEORY:
# Path planning is the problem of finding a collision-free route from a start state
# to a goal state through an environment. Uninformed search algorithms such as
# Breadth-First Search (BFS) and Dijkstra's algorithm explore the search space
# without any knowledge of where the goal is, expanding nodes uniformly in all
# directions. This is computationally expensive as the number of nodes explored
# grows rapidly with the size of the environment.
#
# A* is an INFORMED search algorithm — it uses domain knowledge in the form of a
# heuristic function h(n) to estimate the remaining cost to the goal, biasing the
# search toward promising regions of the search space. At each step, A* selects
# the node that minimises the evaluation function:
#
#   f(n) = g(n) + h(n)
#     g(n) - actual cost accumulated from start to node n (path length so far)
#     h(n) - estimated cost from node n to goal (heuristic)
#     f(n) - total estimated cost of the cheapest solution path through node n
#
# This allows A* to find the optimal path while expanding significantly fewer nodes
# than uninformed methods, making it well suited for real-time robotic path planning.
#
# For A* to guarantee an optimal solution, h(n) must be ADMISSIBLE — it must never
# overestimate the true cost to the goal. Here, Euclidean (straight-line) distance
# is used as the heuristic. On a 4-directional grid, the true path length is always
# >= the straight-line distance, so admissibility is satisfied and optimality is
# guaranteed.
#
# The main path planning function. Additional functions, classes,
# variables, libraries, etc. can be added to the file, but this
# function must always be defined with these arguments and must
# return an array ('list') of coordinates (col,row).
#DO NOT EDIT THIS FUNCTION DECLARATION
def do_a_star(grid, start, end, display_message):
    #EDIT ANYTHING BELOW HERE

    # Grid indexed as grid[col][row]: 1 = free, 0 = obstacle
    COL = len(grid)
    ROW = len(grid[0])

    # Admissible heuristic: Euclidean distance to goal. Never overestimates
    # on a 4-directional grid, guaranteeing A* returns the optimal path.
    def heuristic(node):
        return ((node[0] - end[0]) ** 2 + (node[1] - end[1]) ** 2) ** 0.5

    # Open list: [f(n), g(n), node] — sorted by f(n), acts as priority queue
    open_set = [[heuristic(start), 0, start]]

    came_from = {}         # maps each node to its parent for path reconstruction
    g_score = {start: 0}  # best known actual cost g(n) from start to each node
    closed_set = set()     # fully expanded nodes — O(1) lookup, never re-expanded

    while open_set:
        # Always expand the node with lowest f(n) = g(n) + h(n)
        open_set.sort()
        _, g, current = open_set.pop(0)

        # Skip stale duplicates — node already expanded via a cheaper path
        if current in closed_set:
            continue
        closed_set.add(current)

        # Goal reached — reconstruct path by tracing parent pointers back to start
        if current == end:
            display_message("The destination cell is found")
            path = []
            node = end
            while node != start:
                path.append(node)
                node = came_from[node]
            path.append(start)
            path.reverse()  # built end->start, reverse to start->goal
            return path

        # Expand 4-directional neighbours — cardinal moves only, step cost = 1
        for d_col, d_row in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            neighbour = (current[0] + d_col, current[1] + d_row)

            # Skip out-of-bounds, obstacles, and already-expanded nodes
            if neighbour[0] < 0 or neighbour[0] >= COL or neighbour[1] < 0 or neighbour[1] >= ROW:
                continue
            if grid[neighbour[0]][neighbour[1]] == 0 or neighbour in closed_set:
                continue

            new_g = g + 1

            # Update if first discovery or cheaper path found (relaxation step)
            if neighbour not in g_score or new_g < g_score[neighbour]:
                g_score[neighbour] = new_g
                came_from[neighbour] = current
                open_set.append([new_g + heuristic(neighbour), new_g, neighbour])

    # Open list exhausted — no path exists
    display_message("No path found to destination")
    return []

#end of file
