# A* Path Planning - Complete Breakdown

## The Core Problem

You have a grid. You're at `start`. You want to reach `end`. There are walls. You need the **shortest path**.

A naive approach: try every possible route (Dijkstra's / BFS). Works, but slow — it expands in all directions like a flood.

A* is smarter: it **guides the search towards the goal** using a heuristic, so it explores far fewer nodes.

---

## The Key Formula

Every node gets a score:

```
f(n) = g(n) + h(n)
```

| Term | Meaning | In your code |
|------|---------|-------------|
| `g(n)` | **Actual cost** from start to node `n` (steps taken) | `new_g = g + 1` |
| `h(n)` | **Estimated cost** from `n` to goal (heuristic) | `((dx²+dy²)^0.5)` = Euclidean distance |
| `f(n)` | **Total estimated cost** of path through `n` | `f_score = new_g + heuristic(neighbour)` |

A* always expands the node with the **lowest f** first. That's the insight — you're not exploring blindly,
you're always chasing the most promising candidate.

---

## Data Structures

### `open_set` — the frontier
Nodes **discovered but not yet evaluated**. Think of it as a to-do list, sorted by `f` score.

```python
open_set = [[f, g, (col, row)], ...]
open_set.sort()       # cheapest f always at front
open_set.pop(0)       # grab the best candidate
```

### `closed_set` — already solved
Nodes **fully evaluated** — we know the optimal cost to reach them. Never revisit.

```python
closed_set = set()    # fast O(1) lookup
closed_set.add(current)
```

### `came_from` — breadcrumb trail
Maps every node to **which node it came from**. Used to reconstruct the path at the end.

```python
came_from = {(3,2): (2,2), (2,2): (1,2), ...}
```

### `g_score` — best known cost
Dictionary of the best `g` value found so far for each node.
If we find a cheaper route to a node we've already seen, we update it.

```python
if neighbour not in g_score or new_g < g_score[neighbour]:
    g_score[neighbour] = new_g  # found a cheaper path
```

---

## The Algorithm Step-by-Step

```
1. Put START in open_set with f = h(start)
2. Loop:
   a. Pick node with lowest f from open_set  <- the "best guess"
   b. If it's the GOAL -> reconstruct and return path
   c. Add it to closed_set (done with it)
   d. For each of its 4 neighbours:
      - Skip if wall, out of bounds, or in closed_set
      - Calculate new_g = current_g + 1
      - If new_g is better than what we knew -> update and add to open_set
3. If open_set empties -> no path exists
```

---

## Why Euclidean Heuristic?

The spec requires it. Euclidean = straight-line distance:

```python
h = sqrt((col_diff^2 + row_diff^2))
```

For a grid where you can only move 4 directions, **Manhattan distance** (`|dx| + |dy|`) would actually
be a tighter heuristic (never overestimates), but the spec explicitly requires Euclidean.

**Key engineering concept:** The heuristic must be **admissible** — it must never *overestimate* the
true cost. If it overestimates, A* can miss the optimal path. Euclidean distance is always <= actual
path length on a grid (since you can't cut diagonals), so it's admissible.

---

## Path Reconstruction

At the end, `came_from` is a chain of "who led me here":

```
end <- node_N <- ... <- node_1 <- start
```

You walk it backwards, then reverse:

```python
node = end
while node != start:
    path.append(node)
    node = came_from[node]
path.append(start)
path.reverse()
```

---

## Why It's Fast

On your maze test — A* was near-instant because:

1. The heuristic **directs** the search toward the goal, avoiding dead-ends until forced
2. `closed_set` as a Python `set` gives **O(1) lookup** — checking "have I been here?" is instant
3. Once a node is in `closed_set`, it's never re-processed

Compare to BFS which expands 360 degrees outward from start — on a 20x10 grid that's potentially
200 nodes. A* might solve the same maze expanding only 30-50.

---

## Engineering Context — Why This Matters for Robotics

| Concept | Real-world equivalent |
|---------|----------------------|
| Grid cell | Discretised map tile (SLAM occupancy grid) |
| `g(n)` | Actual distance/energy travelled |
| `h(n)` | Estimated remaining distance to waypoint |
| Obstacle | Detected wall, no-fly zone, hazard zone |
| Open set | ROS `nav_stack` global planner candidate list |
| Replanning | Re-running A* when sensor detects new obstacle |

In real robots (ROS 2 Nav2, PX4, ArduPilot), A* or its derivatives (Theta*, D* Lite, RRT*) are used
for **global path planning** — the high-level "how do I get from A to B". A separate **local planner**
(e.g. DWA - Dynamic Window Approach) then handles real-time obstacle avoidance along that path.

---

## Variants Worth Knowing (for exams / interviews)

| Algorithm | Difference from A* |
|-----------|-------------------|
| **Dijkstra** | A* with h=0. Explores everything equally. Guaranteed optimal but slow. |
| **BFS** | Dijkstra with uniform cost. Good for unweighted grids. |
| **Greedy Best-First** | A* with g=0. Fast but not guaranteed optimal. |
| **D* Lite** | Dynamic A*. Re-plans efficiently when new obstacles appear (used in Mars rovers). |
| **RRT*** | Sampling-based. Works in continuous space, not just grids. Used in robot arms, drones. |
| **Theta*** | A* variant that allows any-angle paths, not just grid-aligned. |

---

## One-Line Summary

> A* finds the shortest path by always expanding the node that looks most promising —
> balancing how far you've come (g) against how far you still need to go (h).
