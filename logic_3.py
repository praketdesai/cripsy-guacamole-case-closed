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
        nx, ny = (x + dx) % W, (y + dy) % H
        if bit_map[ny, nx] == 0:
            moves.append(name)
        if boosts > 0:
            nx2, ny2 = (x + 2*dx) % W, (y + 2*dy) % H
            if bit_map[ny, nx] == 0 and bit_map[ny2, nx2] == 0:
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
            return new_map, (x, y), True
        x, y = nx, ny
        new_map[y, x] = 1
    return new_map, (x, y), False

# ----------------------
# Advanced Heuristic: Dead-end + Red/Black + Voronoi/Mobility
# ----------------------
def advanced_heuristic(bit_map, our_pos, enemy_pos, our_boosts=0, enemy_boosts=0):
    H, W = bit_map.shape
    free_mask = (bit_map == 0)

    # 1. Red/Black checkerboard
    red_squares = np.zeros_like(bit_map, dtype=int)
    black_squares = np.zeros_like(bit_map, dtype=int)
    for y in range(H):
        for x in range(W):
            if free_mask[y, x]:
                if (x + y) % 2 == 0:
                    red_squares[y, x] = 1
                else:
                    black_squares[y, x] = 1

    def accessible_colors(pos):
        visited = np.zeros_like(bit_map, dtype=bool)
        queue = deque([pos])
        visited[pos[1], pos[0]] = True
        reds = blacks = 0
        while queue:
            x, y = queue.popleft()
            if red_squares[y, x]:
                reds += 1
            elif black_squares[y, x]:
                blacks += 1
            for dx, dy in DELTAS.values():
                nx, ny = (x + dx) % W, (y + dy) % H
                if free_mask[ny, nx] and not visited[ny, nx]:
                    visited[ny, nx] = True
                    queue.append((nx, ny))
        return reds, blacks

    our_red, our_black = accessible_colors(our_pos)
    enemy_red, enemy_black = accessible_colors(enemy_pos)

    # 2. Dead-end penalty
    def dead_end_penalty(pos):
        moves = generate_moves(bit_map, pos)
        if not moves:
            return -9999
        return -5 if len(moves) == 1 else 0

    penalty = dead_end_penalty(our_pos) - dead_end_penalty(enemy_pos)

    # 3. Voronoi + mobility
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

    us_win = np.sum((dist_us < dist_enemy) & free_mask)
    enemy_win = np.sum((dist_enemy < dist_us) & free_mask)
    mobility = len(generate_moves(bit_map, our_pos)) - len(generate_moves(bit_map, enemy_pos))

    # 4. Boost factor
    boost_factor = (our_boosts - enemy_boosts) * 2

    # Combine everything
    score = (our_red + our_black) - (enemy_red + enemy_black)
    score += penalty + (us_win - enemy_win) * 2 + mobility + boost_factor

    return score

# ----------------------
# Alpha-Beta Minimax with Random Tie-Breaking
# ----------------------
def alphabeta(bit_map, our_pos, enemy_pos, our_boosts, enemy_boosts,
              depth, alpha=-float('inf'), beta=float('inf'),
              maximizing=True, start_time=None, time_limit=3.8):
    if start_time and time.time() - start_time > time_limit:
        raise TimeoutError()

    if depth == 0:
        return advanced_heuristic(bit_map, our_pos, enemy_pos, our_boosts, enemy_boosts), None

    pos = our_pos if maximizing else enemy_pos
    boosts = our_boosts if maximizing else enemy_boosts
    moves = generate_moves(bit_map, pos, boosts=boosts)

    if not moves:
        return (-99999 if maximizing else 99999), None

    # Move ordering by advanced heuristic
    def move_score(mv):
        nm, np_pos, dead = apply_move(bit_map, pos, mv)
        return -advanced_heuristic(nm, np_pos if maximizing else our_pos,
                                   enemy_pos if maximizing else np_pos,
                                   our_boosts, enemy_boosts)

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
    return best_value, best_move

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
