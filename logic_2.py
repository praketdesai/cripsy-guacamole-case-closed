import time
import numpy as np
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
    H, W = bit_map.shape
    x, y = pos
    moves = []

    for name, (dx, dy) in DELTAS.items():
        # Normal move
        nx, ny = (x + dx) % W, (y + dy) % H
        if bit_map[ny, nx] == 0:
            moves.append(name)

        # Boosted move
        if boosts > 0:
            nx1, ny1 = (x + dx) % W, (y + dy) % H
            nx2, ny2 = (nx1 + dx) % W, (ny1 + dy) % H
            if bit_map[ny1, nx1] == 0 and bit_map[ny2, nx2] == 0:
                moves.append(f"{name}:BOOST")

    return moves

def apply_move(bit_map, pos, move):
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
# Adaptive Heuristic
# ----------------------
def adaptive_heuristic(bit_map, our_pos, enemy_pos, our_boosts=0, enemy_boosts=0, turn_count=0):
    H, W = bit_map.shape
    
    # 1. Voronoi control (distance-based territory)
    dist_us = np.full((H, W), np.inf)
    dist_enemy = np.full((H, W), np.inf)

    def bfs(start, dist):
        queue = deque([start])
        sx, sy = start
        dist[sy, sx] = 0
        while queue:
            x, y = queue.popleft()
            d = dist[y, x] + 1
            for dx, dy in ((0,1),(0,-1),(1,0),(-1,0)):
                nx, ny = (x + dx) % W, (y + dy) % H
                if bit_map[ny, nx] or dist[ny, nx] != np.inf:
                    continue
                dist[ny, nx] = d
                queue.append((nx, ny))

    bfs(our_pos, dist_us)
    bfs(enemy_pos, dist_enemy)

    free_mask = (bit_map == 0)
    us_win = np.sum((dist_us < dist_enemy) & free_mask)
    enemy_win = np.sum((dist_enemy < dist_us) & free_mask)
    voronoi_score = us_win - enemy_win

    # 2. Mobility / reachable area
    def reachable_area(pos):
        visited = np.zeros_like(bit_map, dtype=bool)
        queue = deque([pos])
        visited[pos[1], pos[0]] = True
        area = 1
        while queue:
            x, y = queue.popleft()
            for dx, dy in DELTAS.values():
                nx, ny = (x + dx) % W, (y + dy) % H
                if bit_map[ny, nx] == 0 and not visited[ny, nx]:
                    visited[ny, nx] = True
                    queue.append((nx, ny))
                    area += 1
        return area

    our_area = reachable_area(our_pos)
    enemy_area = reachable_area(enemy_pos)
    mobility_score = our_area - enemy_area

    # 3. Distance to opponent (Manhattan)
    manhattan_dist = abs(our_pos[0] - enemy_pos[0]) + abs(our_pos[1] - enemy_pos[1])

    # 4. Boost factor
    boost_factor = 0.5 * (our_boosts - enemy_boosts)

    # 5. Adaptive weighting
    if turn_count < 30:  # early game favors area control
        weight_voronoi = 2.5
        weight_mobility = 1.0
        weight_distance = 0.5
    else:  # late game favors mobility and spacing
        weight_voronoi = 1.0
        weight_mobility = 2.0
        weight_distance = 1.0

    score = (weight_voronoi * voronoi_score +
             weight_mobility * mobility_score +
             weight_distance * manhattan_dist +
             boost_factor)

    return score


# ----------------------
# Modified Alphabeta
# ----------------------
def alphabeta(bit_map, our_pos, enemy_pos, our_boosts, enemy_boosts,
              depth, alpha=-float('inf'), beta=float('inf'),
              maximizing=True, start_time=None, time_limit=3.8, turn_count=0):
    if start_time and time.time() - start_time > time_limit:
        raise TimeoutError()

    if depth == 0:
        return adaptive_heuristic(bit_map, our_pos, enemy_pos, our_boosts, enemy_boosts, turn_count), None

    pos = our_pos if maximizing else enemy_pos
    boosts = our_boosts if maximizing else enemy_boosts
    moves = generate_moves(bit_map, pos, boosts=boosts)

    if not moves:
        return (-99999 if maximizing else 99999), None

    # Move ordering using heuristic
    def move_score(mv):
        nm, np_pos, dead = apply_move(bit_map, pos, mv)
        if dead:
            return -99999
        return -adaptive_heuristic(nm, np_pos if maximizing else our_pos,
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
