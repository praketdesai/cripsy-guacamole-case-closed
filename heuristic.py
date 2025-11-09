import numpy as np
from collections import deque

NEIGHBORS = ((0,1),(1,0),(0,-1),(-1,0))

# ----------------------
# Enhanced Evaluate Position
# ----------------------
def evaluate_position(bit_map, our_pos, enemy_pos, our_boosts=0, enemy_boosts=0, turn_count=0):
    """
    Improved heuristic for evaluating the game position:
    Combines Voronoi control, mobility, enemy distance, and boost awareness.
    Adapts weighting depending on turn number for early/mid/late game strategy.
    Works on torus maps.
    """
    H, W = bit_map.shape
    free_mask = (bit_map == 0)

    # ----------------------
    # 1️⃣ Voronoi / Distance Control
    # ----------------------
    def bfs(start):
        dist = np.full((H, W), np.inf)
        q = deque([start])
        x, y = start
        dist[y, x] = 0
        while q:
            cx, cy = q.popleft()
            for dx, dy in NEIGHBORS:
                nx, ny = (cx + dx) % W, (cy + dy) % H
                if free_mask[ny, nx] and dist[ny, nx] == np.inf:
                    dist[ny, nx] = dist[cy, cx] + 1
                    q.append((nx, ny))
        return dist

    dist_us = bfs(our_pos)
    dist_enemy = bfs(enemy_pos)

    us_voronoi = np.sum((dist_us < dist_enemy) & free_mask)
    enemy_voronoi = np.sum((dist_enemy < dist_us) & free_mask)
    voronoi_score = us_voronoi - enemy_voronoi

    # ----------------------
    # 2️⃣ Mobility / Reachable Area
    # ----------------------
    def reachable_area(pos):
        visited = np.zeros_like(bit_map, dtype=bool)
        q = deque([pos])
        visited[pos[1], pos[0]] = True
        area = 1
        while q:
            cx, cy = q.popleft()
            for dx, dy in NEIGHBORS:
                nx, ny = (cx + dx) % W, (cy + dy) % H
                if free_mask[ny, nx] and not visited[ny, nx]:
                    visited[ny, nx] = True
                    q.append((nx, ny))
                    area += 1
        return area

    mobility_score = reachable_area(our_pos) - reachable_area(enemy_pos)

    # ----------------------
    # 3️⃣ Enemy Distance Factor
    # ----------------------
    manhattan_dist = abs(our_pos[0] - enemy_pos[0]) + abs(our_pos[1] - enemy_pos[1])

    # ----------------------
    # 4️⃣ Boost Factor
    # ----------------------
    boost_factor = 0.5 * (our_boosts - enemy_boosts)

    # ----------------------
    # 5️⃣ Adaptive Weighting by Turn
    # ----------------------
    if turn_count < 30:  # Early game: favor expansion
        w_v, w_m, w_d, w_b = 3.0, 1.0, 0.2, 0.5
    elif turn_count < 60:  # Mid game: favor mobility & positioning
        w_v, w_m, w_d, w_b = 1.5, 2.0, 0.5, 0.5
    else:  # Late game: survival, avoid traps
        w_v, w_m, w_d, w_b = 1.0, 3.0, 1.0, 0.2

    # ----------------------
    # 6️⃣ Combine Scores
    # ----------------------
    score = (w_v * voronoi_score +
             w_m * mobility_score +
             w_d * manhattan_dist +
             w_b * boost_factor)

    return score
