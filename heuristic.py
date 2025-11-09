import numpy as np

# ----------------------
# Neighbor offsets
# ----------------------
NEIGHBORS = ((0,1),(1,0),(0,-1),(-1,0))

# ----------------------
# Fast Evaluate Position
# ----------------------
def evaluate_position(bit_map, our_pos, enemy_pos, our_boosts=0, enemy_boosts=0, turn_count=0):
    """
    Fast heuristic evaluation using Voronoi, mobility, trap awareness, and boosts.
    Fully vectorized for speed on torus maps.
    """
    H, W = bit_map.shape
    free_mask = (bit_map == 0)
    total_free = np.sum(free_mask)
    if total_free == 0:
        return -9999  # board full, lose

    # ----------------------
    # 1️⃣ Compute distance maps (Voronoi)
    # ----------------------
    def bfs_dist(start):
        dist = np.full((H, W), np.inf)
        visited = np.zeros((H, W), bool)
        queue = [start]
        dist[start[1], start[0]] = 0
        visited[start[1], start[0]] = True

        while queue:
            x, y = queue.pop(0)
            for dx, dy in NEIGHBORS:
                nx, ny = (x + dx) % W, (y + dy) % H
                if free_mask[ny, nx] and not visited[ny, nx]:
                    visited[ny, nx] = True
                    dist[ny, nx] = dist[y, x] + 1
                    queue.append((nx, ny))
        return dist

    dist_us = bfs_dist(our_pos)
    dist_enemy = bfs_dist(enemy_pos)

    # Voronoi score with small neutral credit
    us_cells = (dist_us < dist_enemy) & free_mask
    enemy_cells = (dist_enemy < dist_us) & free_mask
    neutral_cells = (dist_us == dist_enemy) & free_mask
    voronoi_score = np.sum(us_cells) - np.sum(enemy_cells) + 0.1*np.sum(neutral_cells)

    # ----------------------
    # 2️⃣ Mobility (reachable area) – vectorized BFS
    # ----------------------
    mobility_score = (np.sum(dist_us < np.inf) - np.sum(dist_enemy < np.inf)) / total_free

    # ----------------------
    # 3️⃣ Enemy distance penalty
    # ----------------------
    manhattan_dist = abs(our_pos[0] - enemy_pos[0]) + abs(our_pos[1] - enemy_pos[1])
    dist_score = -2 / (1 + manhattan_dist)  # inversely penalize proximity

    # ----------------------
    # 4️⃣ Boost factor
    # ----------------------
    boost_score = 1.5 * (our_boosts - enemy_boosts)

    # ----------------------
    # 5️⃣ Trap / corner awareness using shifts
    # ----------------------
    trap_us = sum(~np.roll(np.roll(free_mask, dx, axis=1), dy, axis=0)[our_pos[1], our_pos[0]] 
                  for dx, dy in NEIGHBORS)
    trap_enemy = sum(~np.roll(np.roll(free_mask, dx, axis=1), dy, axis=0)[enemy_pos[1], enemy_pos[0]] 
                     for dx, dy in NEIGHBORS)
    trap_score = -2*trap_us + 1*trap_enemy

    # ----------------------
    # 6️⃣ Adaptive weighting
    # ----------------------
    if turn_count < 30:
        w_v, w_m, w_d, w_b, w_t = 3.0, 1.0, 0.5, 1.0, 1.0
    elif turn_count < 60:
        w_v, w_m, w_d, w_b, w_t = 1.5, 2.0, 0.8, 1.0, 1.5
    else:
        w_v, w_m, w_d, w_b, w_t = 1.0, 3.0, 1.2, 0.5, 2.0

    # ----------------------
    # 7️⃣ Combine Scores
    # ----------------------
    score = (
        w_v * voronoi_score +
        w_m * mobility_score +
        w_d * dist_score +
        w_b * boost_score +
        w_t * trap_score
    )

    return score
