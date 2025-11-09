import time
import random
import numpy as np
from collections import deque, defaultdict
from util import generate_moves, apply_move, game_to_bit_map
from heuristic import evaluate_position  # your improved heuristic

# ----------------------
# Alpha-Beta Minimax with caching
# ----------------------
def alphabeta(bit_map, our_pos, enemy_pos, our_boosts, enemy_boosts,
              depth, alpha=-float('inf'), beta=float('inf'),
              maximizing=True, start_time=None, time_limit=3.2,
              cache=None):

    # Timeout check
    if start_time and time.time() - start_time > time_limit:
        raise TimeoutError()

    # Initialize cache
    if cache is None:
        cache = {}

    # Board hash key for caching
    key = (bit_map.tobytes(), our_pos, enemy_pos, our_boosts, enemy_boosts, depth, maximizing)
    if key in cache:
        return cache[key], None

    # Leaf node: evaluate
    if depth == 0:
        val = evaluate_position(bit_map, our_pos, enemy_pos, our_boosts, enemy_boosts)
        cache[key] = val
        return val, None

    # Determine whose turn
    pos = our_pos if maximizing else enemy_pos
    boosts = our_boosts if maximizing else enemy_boosts

    # Generate all legal moves
    moves = generate_moves(bit_map, pos, boosts)
    if not moves:
        val = -99999 if maximizing else 99999
        cache[key] = val
        return val, None

    # Precompute move scores for ordering
    move_scores = []
    for move in moves:
        nm, np_pos, dead = apply_move(bit_map, pos, move)
        if dead:
            score = -99999 if maximizing else 99999
        else:
            score = -evaluate_position(
                nm,
                np_pos if maximizing else our_pos,
                enemy_pos if maximizing else np_pos,
                our_boosts,
                enemy_boosts
            )
        move_scores.append((score, move))

    # Sort moves descending for maximizing, ascending for minimizing
    move_scores.sort(key=lambda x: x[0], reverse=maximizing)

    best_value = -float('inf') if maximizing else float('inf')
    best_moves = []

    for _, move in move_scores:
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
                depth - 1,
                alpha, beta,
                not maximizing,
                start_time, time_limit,
                cache
            )

        # Update best value and alpha/beta
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
    cache[key] = best_value
    return best_value, best_move


# ----------------------
# Choose Next Move
# ----------------------
def choose_next_move(game, player_number=1, max_depth=4, time_limit=3.2):
    our_agent = game.agent1 if player_number == 1 else game.agent2
    enemy_agent = game.agent2 if player_number == 1 else game.agent1

    bit_map = game_to_bit_map(game)
    our_pos = tuple(our_agent.trail[-1])
    enemy_pos = tuple(enemy_agent.trail[-1])
    our_boosts = getattr(our_agent, "boosts_remaining", 0)
    enemy_boosts = getattr(enemy_agent, "boosts_remaining", 0)

    
    free_cells = np.sum(bit_map == 0)
    if free_cells > 290:
        max_depth = 4  # early game, large board -> shallow
    elif free_cells > 200:
        max_depth = 6  # mid game
    elif free_cells > 100:
        max_depth = 8  # late game, fewer options -> deeper search
    else:
        max_depth = 12  # late game, fewer options -> deeper search

    start_time = time.time()
    best_move = None
    try:
        _, best_move = alphabeta(
            bit_map,
            our_pos,
            enemy_pos,
            our_boosts,
            enemy_boosts,
            max_depth,
            start_time=start_time,
            time_limit=time_limit,
            cache={}
        )
    except TimeoutError:
        pass

    if not best_move:
        moves = generate_moves(bit_map, our_pos)
        best_move = moves[0] if moves else "UP"

    return best_move
