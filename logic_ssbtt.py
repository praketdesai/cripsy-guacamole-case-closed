import time
import numpy as np
import random
from case_closed_game import Game
from collections import deque, defaultdict



BOARD_HEIGHT = 18
BOARD_WIDTH = 20
MOVES = ["UP", "DOWN", "LEFT", "RIGHT"]
DELTAS = {"UP": (0, -1), "DOWN": (0, 1), "LEFT": (-1, 0), "RIGHT": (1, 0)}

# ----------------------
# Board Conversion
# ----------------------
def game_to_bit_map(game: Game):
    bit_map = np.zeros((BOARD_HEIGHT, BOARD_WIDTH), dtype=np.uint8)
    for tx, ty in game.agent1.get_trail_positions() + game.agent2.get_trail_positions():
        if 0 <= tx < BOARD_WIDTH and 0 <= ty < BOARD_HEIGHT:
            bit_map[ty, tx] = 1
    return bit_map

# ----------------------
# Move Generation / Application
# ----------------------
def generate_moves(bit_map, pos, boosts=0):
    """Return only valid moves (including checking boosted collisions)."""
    H, W = bit_map.shape
    x, y = pos
    moves = []
    for name, (dx, dy) in DELTAS.items():
        # Normal move
        nx, ny = (x + dx) % W, (y + dy) % H
        if bit_map[ny, nx] == 0:
            moves.append(name)
        # Boosted move (if available)
        if boosts > 0:
            nx2, ny2 = (x + 2*dx) % W, (y + 2*dy) % H
            if bit_map[ny, nx] == 0 and bit_map[ny2, nx2] == 0:
                moves.append(f"{name}:BOOST")
    return moves

def apply_move(bit_map, pos, move):
    """Apply a move safely, return (new_map, new_pos, dead)."""
    H, W = bit_map.shape
    new_map = bit_map.copy()
    boost = ":BOOST" in move
    base_move = move.replace(":BOOST", "")
    dx, dy = DELTAS[base_move]
    steps = 2 if boost else 1
    x, y = pos
    for _ in range(steps):
        nx, ny = (x + dx) % W, (y + dy) % H
        if new_map[ny, nx]:
            return new_map, (x, y), True  # dead
        x, y = nx, ny
        new_map[y, x] = 1
    return new_map, (x, y), False

def heuristic(bit_map, our_pos, enemy_pos, our_boosts=0, enemy_boosts=0, turn_count=0):
    """
    Final unified heuristic:
      - Voronoi territory
      - Mobility / reachable area
      - Checkerboard parity dominance
      - Dead-end / trap detection
      - Chamber decomposition (Tarjan articulation points) INLINE
      - Chamber graph reachability and scoring
      - Flood-fill territory control
      - Boost advantage
      - Adaptive weighting by turns + map openness
    """

    H, W = bit_map.shape
    free_mask = (bit_map == 0)

    # =============== 1. Voronoi distance maps ===============
    dist_us = np.full((H, W), np.inf)
    dist_enemy = np.full((H, W), np.inf)

    def bfs(start, dist):
        q = deque([start])
        sx, sy = start
        dist[sy, sx] = 0
        while q:
            x, y = q.popleft()
            d = dist[y, x] + 1
            for dx, dy in ((0,1),(0,-1),(1,0),(-1,0)):
                nx, ny = (x+dx) % W, (y+dy) % H
                if free_mask[ny, nx] and dist[ny, nx] == np.inf:
                    dist[ny, nx] = d
                    q.append((nx, ny))

    bfs(our_pos, dist_us)
    bfs(enemy_pos, dist_enemy)

    us_voronoi = np.sum((dist_us < dist_enemy) & free_mask)
    enemy_voronoi = np.sum((dist_enemy < dist_us) & free_mask)
    voronoi_score = us_voronoi - enemy_voronoi


    # =============== 2. Mobility / reachable area ===============
    def reachable_area(pos):
        visited = np.zeros_like(bit_map, dtype=bool)
        q = deque([pos])
        visited[pos[1], pos[0]] = True
        area = 1
        while q:
            x, y = q.popleft()
            for dx, dy in DELTAS.values():
                nx, ny = (x+dx)%W, (y+dy)%H
                if free_mask[ny, nx] and not visited[ny, nx]:
                    visited[ny, nx] = True
                    q.append((nx, ny))
                    area += 1
        return area

    mobility_score = reachable_area(our_pos) - reachable_area(enemy_pos)


    # =============== 3. Checkerboard parity dominance ===============
    def accessible_colors(pos):
        visited = np.zeros_like(bit_map, dtype=bool)
        q = deque([pos])
        visited[pos[1], pos[0]] = True
        r = b = 0
        while q:
            x, y = q.popleft()
            if free_mask[y, x]:
                if (x+y) & 1:
                    b += 1
                else:
                    r += 1
            for dx, dy in DELTAS.values():
                nx, ny = (x+dx)%W, (y+dy)%H
                if free_mask[ny, nx] and not visited[ny, nx]:
                    visited[ny, nx] = True
                    q.append((nx, ny))
        return r, b

    our_r, our_b = accessible_colors(our_pos)
    enemy_r, enemy_b = accessible_colors(enemy_pos)
    color_score = (our_r + our_b) - (enemy_r + enemy_b)


    # =============== 4. Trap / dead-end penalties ===============
    def dead_end_penalty(pos):
        moves = generate_moves(bit_map, pos)
        if not moves:
            return -9999
        return -5 if len(moves) == 1 else 0

    penalty_score = dead_end_penalty(our_pos) - dead_end_penalty(enemy_pos)


    # =============== 5. Manhattan distance ===============
    manhattan_dist = abs(our_pos[0] - enemy_pos[0]) + abs(our_pos[1] - enemy_pos[1])


    # =============== 6. Boost advantage ===============
    boost_score = 2 * (our_boosts - enemy_boosts)


    # ==========================================================
    # =================== 7. Chambers ==========================
    # ==========================================================

    # ---- Tarjan articulation detection ----
    index = np.full((H, W), -1, dtype=int)
    lowlink = np.zeros((H, W), dtype=int)
    is_articulation = np.zeros((H, W), dtype=bool)
    idx = [0]

    def dfs_articulation(x, y, parent=None):
        index[y, x] = idx[0]
        lowlink[y, x] = idx[0]
        idx[0] += 1
        children = 0

        for dx, dy in ((0,1),(1,0),(-1,0),(0,-1)):
            nx, ny = (x+dx) % W, (y+dy) % H
            if not free_mask[ny, nx]:
                continue

            if index[ny, nx] == -1:
                children += 1
                dfs_articulation(nx, ny, (x, y))
                lowlink[y, x] = min(lowlink[y, x], lowlink[ny, nx])

                if parent and lowlink[ny, nx] >= index[y, x]:
                    is_articulation[y, x] = True
            elif parent and (nx, ny) != parent:
                lowlink[y, x] = min(lowlink[y, x], index[ny, nx])

        if not parent and children > 1:
            is_articulation[y, x] = True

    for y in range(H):
        for x in range(W):
            if free_mask[y, x] and index[y, x] == -1:
                dfs_articulation(x, y)


    # ---- Chamber labeling (components excluding articulation pts) ----
    chamber_id = np.full((H, W), -1, dtype=int)
    chamber_size = []
    cid = 0

    for y in range(H):
        for x in range(W):
            if free_mask[y, x] and not is_articulation[y, x] and chamber_id[y, x] == -1:
                q = deque([(x, y)])
                chamber_id[y, x] = cid
                size = 1
                while q:
                    qx, qy = q.popleft()
                    for dx, dy in ((0,1),(1,0),(-1,0),(0,-1)):
                        nx, ny = (qx+dx)%W, (qy+dy)%H
                        if free_mask[ny, nx] and not is_articulation[ny, nx] and chamber_id[ny, nx] == -1:
                            chamber_id[ny, nx] = cid
                            size += 1
                            q.append((nx, ny))
                chamber_size.append(size)
                cid += 1

    

    # ---- Chamber adjacency graph ----
    chamber_graph = defaultdict(set)

    for y in range(H):
        for x in range(W):
            if is_articulation[y, x]:
                adjacent = set()
                for dx, dy in ((0,1),(1,0),(-1,0),(0,-1)):
                    nx, ny = (x+dx)%W, (y+dy)%H
                    if chamber_id[ny, nx] != -1:
                        adjacent.add(chamber_id[ny, nx])
                # connect all pairs of chambers via this articulation point
                for a in adjacent:
                    for b in adjacent:
                        if a != b:
                            chamber_graph[a].add(b)


    # ---- Which chamber players are in ----
    def get_chamber(pos):
        x, y = pos
        return chamber_id[y, x] if chamber_id[y, x] != -1 else -1

    our_chamber = get_chamber(our_pos)
    enemy_chamber = get_chamber(enemy_pos)

    # ---- BFS chamber reachability ----
    def reachable_chambers(start):
        if start == -1:
            return {}
        dist = {start: 0}
        q = deque([start])
        while q:
            c = q.popleft()
            for n in chamber_graph[c]:
                if n not in dist:
                    dist[n] = dist[c] + 1
                    q.append(n)
        return dist

    our_reach = reachable_chambers(our_chamber)
    enemy_reach = reachable_chambers(enemy_chamber)

    # ---- scoring the chambers ----
    chamber_score = 0
    for i, size in enumerate(chamber_size):
        d1 = our_reach.get(i, np.inf)
        d2 = enemy_reach.get(i, np.inf)
        if d1 < d2:
            chamber_score += size * 2
        elif d2 < d1:
            chamber_score -= size * 2
        else:
            chamber_score += size * 0.2

    # small chamber penalties
    if our_chamber != -1 and chamber_size[our_chamber] < 5:
        chamber_score -= 10
    if enemy_chamber != -1 and chamber_size[enemy_chamber] < 5:
        chamber_score += 10


    # ==========================================================
    # ================= Flood-fill control =====================
    # ==========================================================
    our_control = np.sum((dist_us < dist_enemy) & free_mask)
    enemy_control = np.sum((dist_enemy < dist_us) & free_mask)
    neutral = np.sum((dist_us == dist_enemy) & free_mask)
    flood_score = (our_control - enemy_control) + 0.2 * neutral


    # ==========================================================
    # ================= Combining everything ===================
    # ==========================================================

    # Map openness → flood-fill weight
    open_fraction = np.mean(free_mask)
    chamber_weight = 0.5 * (1 - open_fraction)
    flood_weight = min(1.0, open_fraction * 2.5)
    global_weight = 1 - chamber_weight

    # Turn-based adaptive global weights
    if turn_count < 30:
        w_voro = 2.0
        w_mobi = 1.2
        w_color = 0.8
        w_dist = 0.4
    else:
        w_voro = 1.0
        w_mobi = 2.5
        w_color = 0.5
        w_dist = 1.0

    global_score = (
        w_voro * voronoi_score +
        w_mobi * mobility_score +
        w_color * color_score +
        penalty_score +
        w_dist * manhattan_dist +
        boost_score +
        flood_weight * flood_score
    )

    total = global_weight * global_score + chamber_weight * chamber_score
    return int(total)


# ----------------------
# Modified Alphabeta
# ----------------------
def alphabeta(bit_map, our_pos, enemy_pos, our_boosts, enemy_boosts,
              depth, alpha=-float('inf'), beta=float('inf'),
              maximizing=True, start_time=None, time_limit=3.8, turn_count=0):
    if start_time and time.time() - start_time > time_limit:
        raise TimeoutError()

    if depth == 0:
        return heuristic(bit_map, our_pos, enemy_pos, our_boosts, enemy_boosts, turn_count), None

    pos = our_pos if maximizing else enemy_pos
    boosts = our_boosts if maximizing else enemy_boosts
    moves = generate_moves(bit_map, pos, boosts=boosts)

    if not moves:
        return (-99999 if maximizing else 99999), None

    # Move ordering using heuristic
    def move_score(mv):
        nm, np_pos, dead = apply_move(bit_map, pos, mv)
        if dead:
            return -9999
        return -heuristic(nm, np_pos if maximizing else our_pos,
                                   enemy_pos if maximizing else np_pos,
                                   our_boosts, enemy_boosts, turn_count)

    moves.sort(key=move_score, reverse=maximizing)

    best_moves = []
    best_value = -float('inf') if maximizing else float('inf')

    for move in moves:
        nm, np_pos, dead = apply_move(bit_map, pos, move)
        if dead:
            val = -99999 if maximizing else 99999
        else:
            boost_used = ":BOOST" in move
            val, _ = alphabeta(
                nm,
                np_pos if maximizing else our_pos,
                enemy_pos if maximizing else np_pos,
                our_boosts - (1 if maximizing and boost_used else 0),
                enemy_boosts - (1 if not maximizing and boost_used else 0),
                depth-1,
                alpha, beta,
                not maximizing,
                start_time, time_limit,
                turn_count+1
            )

        if maximizing:
            if val > best_value:
                best_value = val
                best_moves = [move]
            elif val == best_value:
                best_moves.append(move)
            alpha = max(alpha, val)
            if alpha >= beta:
                break
        else:
            if val < best_value:
                best_value = val
                best_moves = [move]
            elif val == best_value:
                best_moves.append(move)
            beta = min(beta, val)
            if beta <= alpha:
                break

    # Randomly pick among tied best moves
    best_move = random.choice(best_moves) if best_moves else None
    return (best_value, best_move)

# ----------------------
# Choose Next Move
# ----------------------
def choose_next_move(game: Game, player_number=1, max_depth=4, time_limit=3.8):
    our_agent = game.agent1 if player_number == 1 else game.agent2
    enemy_agent = game.agent2 if player_number == 1 else game.agent1
    bit_map = game_to_bit_map(game)
    our_pos = tuple(our_agent.trail[-1])
    enemy_pos = tuple(enemy_agent.trail[-1])
    our_boosts = getattr(our_agent, "boosts_remaining", 0)
    enemy_boosts = getattr(enemy_agent, "boosts_remaining", 0)

    best_move = None
    start_time = time.time()
    try:
        _, best_move = alphabeta(bit_map, our_pos, enemy_pos,
                                 our_boosts, enemy_boosts,
                                 max_depth, start_time=start_time, time_limit=time_limit,
                                 turn_count=game.turns)
    except TimeoutError:
        pass

    if not best_move:
        moves = generate_moves(bit_map, our_pos)
        best_move = moves[0] if moves else "UP"

    return best_move
