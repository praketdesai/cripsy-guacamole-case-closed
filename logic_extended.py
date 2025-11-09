import time
import numpy as np
from collections import deque
from case_closed_game import Game

BOARD_HEIGHT = 18
BOARD_WIDTH = 20
MOVES = ["UP", "DOWN", "LEFT", "RIGHT"]
DELTAS = {"UP": (0, -1), "DOWN": (0, 1), "LEFT": (-1, 0), "RIGHT": (1, 0)}

# ----------------------
# Board Conversion
# ----------------------

def wrap(pos: tuple[int, int]) -> tuple[int, int]:
    x, y = pos
    if BOARD_WIDTH == 0 or BOARD_HEIGHT == 0:
        return (0, 0)
    return (x % BOARD_WIDTH, y % BOARD_HEIGHT)


def neighbors(pos: tuple[int, int]):
    x, y = pos
    for dx, dy in ((0, -1), (0, 1), (1, 0), (-1, 0)):
        yield wrap((x + dx, y + dy))

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
# Heuristic (Voronoi + Mobility)
# ----------------------
def heuristic(bit_map, move_number, our_pos, enemy_pos):
    H, W = bit_map.shape
    dist_us = np.full((H, W), np.inf)
    dist_enemy = np.full((H, W), np.inf)

    def bfs(start, dist, opp_loc) -> bool:
        found_opp = False
        queue = deque([start])
        sx, sy = start
        dist[sy, sx] = 0
        while queue:
            x, y = queue.popleft()
            if opp_loc == (x, y):
                found_opp = True
            d = dist[y, x] + 1
            for dx, dy in ((0,1),(0,-1),(1,0),(-1,0)):
                nx, ny = (x + dx) % W, (y + dy) % H
                if bit_map[ny, nx] or dist[ny, nx] != np.inf:
                    continue
                dist[ny, nx] = d
                queue.append((nx, ny))
        return found_opp

    is_endgame = bfs(our_pos, dist_us, enemy_pos)
    bfs(enemy_pos, dist_enemy, our_pos)

    free_mask = (bit_map == 0)
    us_win = np.sum((dist_us < dist_enemy) & free_mask)
    enemy_win = np.sum((dist_enemy < dist_us) & free_mask)

    mobility_factor = 1

    if move_number > 50:
        mobility_factor = 0

    mobility = len(generate_moves(bit_map, our_pos)) - len(generate_moves(bit_map, enemy_pos))

    
    if is_endgame:
        return greedy_safe_wall_hug_heuristic(bit_map, our_pos)
    return (us_win - enemy_win) * 2 + mobility * mobility_factor

def dead_end_fill(bit_map):
    """
    Modify a copy of bit_map in-place:
    Repeatedly mark every open cell with <= 1 open neighbor as blocked (1).
    """
    H, W = bit_map.shape
    filled = bit_map.copy()
    changed = True
    while changed:
        changed = False
        for y in range(H):
            for x in range(W):
                if filled[y, x] == 0:
                    open_neighbors = 0
                    for dx, dy in ((0,1),(0,-1),(1,0),(-1,0)):
                        nx, ny = (x+dx)%W, (y+dy)%H
                        if filled[ny, nx] == 0:
                            open_neighbors += 1
                    if open_neighbors <= 1:
                        filled[y, x] = 1  # Mark dead end as wall
                        changed = True
    return filled


def greedy_safe_wall_hug_heuristic(bit_map, our_pos):
    """
    High score for wall hugging, but safer: fills dead ends before scoring.
    Penalizes small surviving regions and corridors created by dead ends.
    """

    pruned_map = dead_end_fill(bit_map)
    H, W = pruned_map.shape

    dist_us = np.full((H, W), np.inf)

    def bfs(start, dist) -> int:
        count = 0
        queue = deque([start])
        sx, sy = start
        dist[sy, sx] = 0
        while queue:
            x, y = queue.popleft()
            count += 1
            for dx, dy in ((0,1),(0,-1),(1,0),(-1,0)):
                nx, ny = (x + dx) % W, (y + dy) % H
                if pruned_map[ny, nx] or dist[ny, nx] != np.inf:
                    continue
                dist[ny, nx] = 1 # marked as visited
                queue.append((nx, ny))
        return count
    
    cells = bfs(our_pos, dist_us)

    # Wall hugging part (use pruned map)
    wall_count = 0
    for dx, dy in ((0,1),(0,-1),(1,0),(-1,0)):
        nx, ny = (our_pos[0]+dx)%W, (our_pos[1]+dy)%H
        if pruned_map[ny, nx] != 0:
            wall_count += 1

    # Mobility part
    # Recompute area after pruning dead ends (recommended)
    area = cells
    small_area_penalty = -100 if area <= 3 else 0

    return wall_count + 0.01*area + small_area_penalty

# ----------------------
# Alpha-Beta Minimax
# ----------------------
def alphabeta(bit_map, move_number, our_pos, enemy_pos, our_boosts, enemy_boosts,
              depth, alpha=-float('inf'), beta=float('inf'),
              maximizing=True, start_time=None, time_limit=2.0):
    if start_time and time.time() - start_time > time_limit:
        raise TimeoutError()

    if depth == 0:
        return heuristic(bit_map, move_number, our_pos, enemy_pos), None

    pos = our_pos if maximizing else enemy_pos
    boosts = our_boosts if maximizing else enemy_boosts
    best_move = None
    moves = generate_moves(bit_map, pos, boosts=boosts)

    if not moves:
        return (-99999 if maximizing else 99999), None

    # Move ordering by heuristic
    def move_score(mv):
        nm, np_pos, dead = apply_move(bit_map, pos, mv)
        return -heuristic(bit_map, move_number, np_pos if maximizing else our_pos,
                          enemy_pos if maximizing else np_pos)

    moves.sort(key=move_score, reverse=maximizing)

    for move in moves:
        nm, np_pos, dead = apply_move(bit_map, pos, move)
        if dead:
            val = -99999 if maximizing else 99999
        else:
            # Reduce boosts only if move uses a boost
            boost_used = ":BOOST" in move
            val, _ = alphabeta(
                nm,
                move_number+1 if maximizing else move_number,
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
            if val > alpha:
                alpha = val
                best_move = move
            if alpha >= beta:
                break
        else:
            if val < beta:
                beta = val
                best_move = move
            if beta <= alpha:
                break

    return (alpha if maximizing else beta), best_move

# ----------------------
# Choose Next Move
# ----------------------
def choose_next_move(game: Game, player_number=1, max_depth=4, time_limit=3.2):
    our_agent = game.agent1 if player_number==1 else game.agent2
    enemy_agent = game.agent2 if player_number==1 else game.agent1
    bit_map = game_to_bit_map(game)
    our_pos = tuple(our_agent.trail[-1])
    enemy_pos = tuple(enemy_agent.trail[-1])
    our_boosts = getattr(our_agent, "boosts_remaining", 0)
    enemy_boosts = getattr(enemy_agent, "boosts_remaining", 0)

    move_number = game.turns

    best_move = None
    start_time = time.time()
    try:
        # iterative deepening
        best_move_score = float("-inf")
        best_move = None
        fall_back_score = float("-inf")
        fall_back = None
        depth = 4
        while depth < 16:
            fall_back = best_move
            fall_back_score = best_move_score
            best_move_score, best_move = alphabeta(bit_map, move_number, our_pos, enemy_pos,
                                    our_boosts, enemy_boosts,
                                    depth, start_time=start_time, time_limit=time_limit)
            depth += 2
    except TimeoutError:
        pass

    if (fall_back_score > best_move_score):
        best_move = fall_back

    if not best_move:
        moves = generate_moves(bit_map, our_pos)
        best_move = moves[0] if moves else "UP"

    return best_move
