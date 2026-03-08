# The main path planning function. Additional functions, classes,
# variables, libraries, etc. can be added to the file, but this
# function must always be defined with these arguments and must
# return an array ('list') of coordinates (col,row).
#DO NOT EDIT THIS FUNCTION DECLARATION
def do_a_star(grid, start, end, display_message):
    #EDIT ANYTHING BELOW HERE

    # Grid dimensions: grid is indexed as grid[col][row]
    COL = len(grid)
    ROW = len(grid[0])

    # Heuristic: Euclidean distance from node to goal (as required by spec)
    def heuristic(node):
        return ((node[0] - end[0]) ** 2 + (node[1] - end[1]) ** 2) ** 0.5

    # A* open set: list of [f_score, g_score, (col, row)]
    # f = g + h, where g = cost from start, h = heuristic to goal
    open_set = [[heuristic(start), 0, start]]

    # came_from maps each node to the node it was reached from (for path reconstruction)
    came_from = {}

    # g_score stores the best known cost from start to each node
    g_score = {start: 0}

    # closed_set holds nodes already fully evaluated (do not revisit)
    closed_set = set()

    while open_set:
        # Sort by f_score to always expand the most promising node first
        open_set.sort()
        _, g, current = open_set.pop(0)

        # Skip if already evaluated (a better path was found earlier)
        if current in closed_set:
            continue
        closed_set.add(current)

        # Goal reached - reconstruct and return the path
        if current == end:
            display_message("The destination cell is found")
            path = []
            node = end
            # Walk back through came_from to build the full path
            while node != start:
                path.append(node)
                node = came_from[node]
            path.append(start)
            path.reverse()  # Reverse so path goes from start to end
            return path

        # Expand neighbours in 4 directions: up, down, left, right
        for d_col, d_row in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            neighbour = (current[0] + d_col, current[1] + d_row)

            # Skip if outside grid bounds
            if neighbour[0] < 0 or neighbour[0] >= COL or neighbour[1] < 0 or neighbour[1] >= ROW:
                continue

            # Skip if obstacle (grid value 0) or already evaluated
            if grid[neighbour[0]][neighbour[1]] == 0 or neighbour in closed_set:
                continue

            # Each step costs 1
            new_g = g + 1

            # Only update if this path to neighbour is better than any previously found
            if neighbour not in g_score or new_g < g_score[neighbour]:
                g_score[neighbour] = new_g
                came_from[neighbour] = current
                f_score = new_g + heuristic(neighbour)
                open_set.append([f_score, new_g, neighbour])

    # No path exists between start and end
    display_message("No path found to destination")
    return []

#end of file
