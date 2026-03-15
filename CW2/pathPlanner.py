# Author: Juan Ignacio Doval Roque
# Student ID: 10752534
#
# A* (A-Star) Path Planning Algorithm
# Based on: Hart, Nilsson & Raphael, "A Formal Basis for the Heuristic Determination
# of Minimum Cost Paths", IEEE Transactions on Systems Science and Cybernetics, 1968.
#
# A* is an informed search algorithm that finds the optimal (shortest) path between
# a start node and a goal node on a weighted graph. It extends Dijkstra's algorithm
# by introducing a heuristic function h(n) to guide the search toward the goal,
# significantly reducing the number of nodes expanded.
#
# Core formula: f(n) = g(n) + h(n)
#   g(n) - actual cost from start node to current node n (path length so far)
#   h(n) - estimated cost from node n to goal (heuristic)
#   f(n) - total estimated cost of the cheapest path through node n
#
# The heuristic h(n) must be ADMISSIBLE - it must never overestimate the true cost
# to reach the goal. This guarantees A* finds the optimal path. Here we use Euclidean
# distance which is always <= the true path length on a 4-directional grid, satisfying
# admissibility.
#
# Motion model: 4-directional (up, down, left, right). Each move costs g(n) += 1.
# Heuristic: Euclidean distance h(n) = sqrt((dx)^2 + (dy)^2)
#
# The main path planning function. Additional functions, classes,
# variables, libraries, etc. can be added to the file, but this
# function must always be defined with these arguments and must
# return an array ('list') of coordinates (col,row).
#DO NOT EDIT THIS FUNCTION DECLARATION
def do_a_star(grid, start, end, display_message):
    #EDIT ANYTHING BELOW HERE

    # Grid dimensions: grid is indexed as grid[col][row]
    # grid[col][row] = 1 means free cell, 0 means obstacle/wall
    COL = len(grid)
    ROW = len(grid[0])

    # Admissible heuristic: Euclidean (straight-line) distance from node to goal.
    # Never overestimates true path cost on a 4-directional grid, guaranteeing
    # A* returns the optimal path.
    def heuristic(node):
        return ((node[0] - end[0]) ** 2 + (node[1] - end[1]) ** 2) ** 0.5

    # OPEN LIST: candidate nodes discovered but not yet expanded, stored as
    # [f(n), g(n), node]. Sorted by f(n) so the most promising node is always
    # expanded first. Equivalent to a priority queue in standard A* literature.
    open_set = [[heuristic(start), 0, start]]

    # CAME_FROM: maps each node to its parent node (the node it was reached from).
    # Used to reconstruct the optimal path by tracing back from goal to start
    # once the goal node is expanded.
    came_from = {}

    # G_SCORE: stores the best known actual cost g(n) from start to each discovered
    # node. Updated when a cheaper path to an already-discovered node is found.
    g_score = {start: 0}

    # CLOSED LIST: set of fully expanded nodes for which the optimal path cost is
    # known. Nodes in the closed list are never re-expanded. Implemented as a Python
    # set for O(1) average lookup time, critical for performance on large grids.
    closed_set = set()

    # Main A* loop - continues until the open list is empty (no path exists)
    # or the goal node is reached
    while open_set:
        # Select node with lowest f(n) = g(n) + h(n) for expansion.
        # This is the key step that makes A* optimal and efficient - always
        # expanding the most promising candidate first.
        open_set.sort()
        _, g, current = open_set.pop(0)

        # If node was already expanded via a cheaper path (duplicate in open list),
        # skip it. This handles the case where a node is added to the open list
        # multiple times when a better path to it is discovered.
        if current in closed_set:
            continue

        # Mark current node as fully expanded - optimal cost to reach it is known
        closed_set.add(current)

        # GOAL TEST: if the expanded node is the goal, the optimal path is found.
        # Reconstruct path by tracing came_from chain from goal back to start.
        if current == end:
            display_message("The destination cell is found")
            path = []
            node = end
            # Walk backwards through parent pointers from goal to start
            while node != start:
                path.append(node)
                node = came_from[node]
            path.append(start)
            path.reverse()  # Reverse to get path ordered start -> goal
            return path

        # NODE EXPANSION: generate successors using 4-directional motion model.
        # Only cardinal directions (up, down, left, right) are permitted per constraints.
        # Each move has a uniform step cost of 1, contributing to g(n).
        for d_col, d_row in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            neighbour = (current[0] + d_col, current[1] + d_row)

            # Boundary check: discard nodes outside the grid
            if neighbour[0] < 0 or neighbour[0] >= COL or neighbour[1] < 0 or neighbour[1] >= ROW:
                continue

            # Collision check: skip obstacle cells (value 0) and already-expanded nodes
            if grid[neighbour[0]][neighbour[1]] == 0 or neighbour in closed_set:
                continue

            # Tentative g(n): cost to reach neighbour via current node
            new_g = g + 1

            # Update neighbour if this is the first time discovering it, or if a
            # cheaper path has been found (relaxation step, core of A* optimality)
            if neighbour not in g_score or new_g < g_score[neighbour]:
                g_score[neighbour] = new_g
                came_from[neighbour] = current          # record optimal parent
                f_score = new_g + heuristic(neighbour)  # f(n) = g(n) + h(n)
                open_set.append([f_score, new_g, neighbour])

    # Open list exhausted with no path found - goal is unreachable from start
    display_message("No path found to destination")
    return []

#end of file
