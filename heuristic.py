import numpy as np
from collections import deque
from util import generate_moves


# -------------------------------------------------------
# BFS Distance Map
# -------------------------------------------------------

def bfs_distance_map(bit_map, start_pos):
    """
    Computes shortest distance from start_pos to all reachable cells.
    Walls in bit_map are treated as blocked.
    Wrap-around world is preserved.
    Returns a distance array filled with np.inf for unreachable.
    """
    H, W = bit_map.shape
    dist = np.full((H, W), np.inf, dtype=float)

    queue = deque([start_pos])
    x0, y0 = start_pos
    dist[y0, x0] = 0

    while queue:
        x, y = queue.popleft()
        d_next = dist[y, x] + 1

        for dx, dy in ((0,1),(0,-1),(1,0),(-1,0)):
            nx, ny = (x + dx) % W, (y + dy) % H

            # Skip walls or already-visited cells
            if bit_map[ny, nx] or dist[ny, nx] != np.inf:
                continue

            dist[ny, nx] = d_next
            queue.append((nx, ny))

    return dist


# -------------------------------------------------------
# Voronoi Control Score
# -------------------------------------------------------

def voronoi_control(bit_map, our_dist, enemy_dist):
    """
    Computes Voronoi area control.
    Returns:
        us_win: number of tiles we can reach first
        enemy_win: number of tiles enemy can reach first
    """
    free_mask = (bit_map == 0)
    us_win = np.sum((our_dist < enemy_dist) & free_mask)
    enemy_win = np.sum((enemy_dist < our_dist) & free_mask)
    return us_win, enemy_win


# -------------------------------------------------------
# Mobility Score
# -------------------------------------------------------

def mobility_score(bit_map, our_pos, enemy_pos):
    """
    Difference in number of legal moves.
    """
    our_moves = len(generate_moves(bit_map, our_pos))
    enemy_moves = len(generate_moves(bit_map, enemy_pos))
    return our_moves - enemy_moves


# -------------------------------------------------------
# Combined Evaluation
# -------------------------------------------------------

def evaluate_position(bit_map, our_pos, enemy_pos, our_boosts, enemy_boosts, turn_count):
    """
    Combines Voronoi territory + mobility into a unified score.

    Formula:
        score = 2 * (us_win - enemy_win) + mobility
    """
    # Compute BFS distance maps
    dist_us = bfs_distance_map(bit_map, our_pos)
    dist_enemy = bfs_distance_map(bit_map, enemy_pos)

    # Voronoi split
    us_win, enemy_win = voronoi_control(bit_map, dist_us, dist_enemy)

    # Mobility difference
    mob = mobility_score(bit_map, our_pos, enemy_pos)

    # Final heuristic score
    return int(2 * (us_win - enemy_win) + mob)

def light_evaluate(bit_map, our_pos, enemy_pos):
    return abs(our_pos[0] - enemy_pos[0]) + abs(our_pos[1] - enemy_pos[1])