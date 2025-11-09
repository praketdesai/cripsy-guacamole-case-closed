import time
import numpy as np
from collections import defaultdict, deque

import random
from collections import deque
from case_closed_game import Game

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

# ----------------------
# Chamber Tree Heuristic
# ----------------------

def chamber_heuristic(bit_map, our_pos, enemy_pos):
    """
    Chamber + flood-fill hybrid heuristic.
    Evaluates space control via chamber decomposition and open-space reachability.
    """
    H, W = bit_map.shape
    free_mask = (bit_map == 0)

    # 1️⃣ Tarjan’s algorithm: find articulation points
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
            nx, ny = (x + dx) % W, (y + dy) % H
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

    # 2️⃣ Label chambers (connected components excluding articulation points)
    chamber_id = np.full((H, W), -1, dtype=int)
    chamber_size = []
    cid = 0
    for y in range(H):
        for x in range(W):
            if free_mask[y, x] and not is_articulation[y, x] and chamber_id[y, x] == -1:
                queue = deque([(x, y)])
                chamber_id[y, x] = cid
                size = 1
                while queue:
                    qx, qy = queue.popleft()
                    for dx, dy in ((0,1),(1,0),(-1,0),(0,-1)):
                        nx, ny = (qx + dx) % W, (qy + dy) % H
                        if free_mask[ny, nx] and not is_articulation[ny, nx] and chamber_id[ny, nx] == -1:
                            chamber_id[ny, nx] = cid
                            size += 1
                            queue.append((nx, ny))
                chamber_size.append(size)
                cid += 1

    # 3️⃣ Build chamber adjacency graph through articulation points
    chamber_graph = defaultdict(set)
    for y in range(H):
        for x in range(W):
            if is_articulation[y, x]:
                adj = set()
                for dx, dy in ((0,1),(1,0),(-1,0),(0,-1)):
                    nx, ny = (x + dx) % W, (y + dy) % H
                    if 0 <= nx < W and 0 <= ny < H and chamber_id[ny, nx] != -1:
                        adj.add(chamber_id[ny, nx])
                for a in adj:
                    for b in adj:
                        if a != b:
                            chamber_graph[a].add(b)

    def find_chamber(pos):
        x, y = pos
        return chamber_id[y, x] if 0 <= y < H and 0 <= x < W else -1

    our_chamber = find_chamber(our_pos)
    enemy_chamber = find_chamber(enemy_pos)

    def reachable_chambers(start_cid):
        if start_cid == -1:
            return {}
        dist = {start_cid: 0}
        queue = deque([start_cid])
        while queue:
            c = queue.popleft()
            for n in chamber_graph[c]:
                if n not in dist:
                    dist[n] = dist[c] + 1
                    queue.append(n)
        return dist

    our_reach = reachable_chambers(our_chamber)
    enemy_reach = reachable_chambers(enemy_chamber)

    # 4️⃣ Chamber-based score
    chamber_score = 0
    for cid, size in enumerate(chamber_size):
        d1 = our_reach.get(cid, np.inf)
        d2 = enemy_reach.get(cid, np.inf)
        if d1 < d2:
            chamber_score += size * 2
        elif d2 < d1:
            chamber_score -= size * 2
        else:
            chamber_score += size * 0.2

    if our_chamber != -1 and chamber_size[our_chamber] < 5:
        chamber_score -= 10
    if enemy_chamber != -1 and chamber_size[enemy_chamber] < 5:
        chamber_score += 10

    # 5️⃣ Open-space (flood-fill) control
    q1, q2 = deque([our_pos]), deque([enemy_pos])
    dist1 = np.full((H, W), np.inf)
    dist2 = np.full((H, W), np.inf)
    dist1[our_pos[1], our_pos[0]] = 0
    dist2[enemy_pos[1], enemy_pos[0]] = 0

    while q1:
        x, y = q1.popleft()
        for dx, dy in ((0,1),(1,0),(-1,0),(0,-1)):
            nx, ny = (x+dx)%W, (y+dy)%H
            if free_mask[ny, nx] and dist1[ny, nx] == np.inf:
                dist1[ny, nx] = dist1[y, x] + 1
                q1.append((nx, ny))

    while q2:
        x, y = q2.popleft()
        for dx, dy in ((0,1),(1,0),(-1,0),(0,-1)):
            nx, ny = (x+dx)%W, (y+dy)%H
            if free_mask[ny, nx] and dist2[ny, nx] == np.inf:
                dist2[ny, nx] = dist2[y, x] + 1
                q2.append((nx, ny))

    our_control = np.sum((dist1 < dist2) & free_mask)
    enemy_control = np.sum((dist2 < dist1) & free_mask)
    neutral = np.sum((dist1 == dist2) & free_mask)
    flood_score = (our_control - enemy_control) + 0.2 * neutral

    # 6️⃣ Blend chamber and flood-fill heuristics adaptively
    open_fraction = np.mean(free_mask)
    flood_weight = min(1.0, open_fraction * 3.0)
    chamber_weight = 1.0 - flood_weight

    total_score = chamber_score * chamber_weight + flood_score * flood_weight

    return int(total_score)


# ----------------------
# Alpha-Beta Minimax with Random Tie-Breaking
# ----------------------
def alphabeta(bit_map, our_pos, enemy_pos, our_boosts, enemy_boosts,
              depth, alpha=-float('inf'), beta=float('inf'),
              maximizing=True, start_time=None, time_limit=3.8):
    if start_time and time.time() - start_time > time_limit:
        raise TimeoutError()

    if depth == 0:
        return chamber_heuristic(bit_map, our_pos, enemy_pos), None

    pos = our_pos if maximizing else enemy_pos
    boosts = our_boosts if maximizing else enemy_boosts
    moves = generate_moves(bit_map, pos, boosts=boosts)

    if not moves:
        return (-99999 if maximizing else 99999), None

    # Move ordering by heuristic
    def move_score(mv):
        nm, np_pos, dead = apply_move(bit_map, pos, mv)
        return -chamber_heuristic(bit_map, np_pos if maximizing else our_pos,
                          enemy_pos if maximizing else np_pos)

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
                start_time, time_limit
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

    best_move = random.choice(best_moves) if best_moves else None
    return (best_value, best_move)


# ----------------------
# Choose Next Move
# ----------------------
def choose_next_move(game: Game, player_number=1, max_depth=4, time_limit=3.8):
    our_agent = game.agent1 if player_number==1 else game.agent2
    enemy_agent = game.agent2 if player_number==1 else game.agent1
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
                                 max_depth, start_time=start_time, time_limit=time_limit)
    except TimeoutError:
        pass

    if not best_move:
        moves = generate_moves(bit_map, our_pos)
        best_move = moves[0] if moves else "UP"

    return best_move
