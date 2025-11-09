import time
import random
from collections import deque
from case_closed_game import Game
from util import BOARD_HEIGHT, BOARD_WIDTH, DELTAS, game_to_bit_map, generate_moves, apply_move_inplace, undo_move_inplace
from heuristic import evaluate_position, light_evaluate


# ----------------------
# Alphabeta Search
# ----------------------
def alphabeta(bit_map, our_pos, enemy_pos, our_boosts, enemy_boosts,
              depth, alpha=-float('inf'), beta=float('inf'),
              maximizing=True, start_time=None, time_limit=3.2, turn_count=0):
    
    # Time cutoff
    if start_time and time.time() - start_time > time_limit:
        raise TimeoutError()

    # Depth limit
    if depth == 0:
        return evaluate_position(bit_map, our_pos, enemy_pos, our_boosts, enemy_boosts, turn_count), None

    pos = our_pos if maximizing else enemy_pos
    boosts = our_boosts if maximizing else enemy_boosts
    moves = generate_moves(bit_map, pos, boosts)

    if not moves:
        return (-99999 if maximizing else 99999), None

    # Move ordering
    def move_score(move):
        new_pos, dead, changed = apply_move_inplace(bit_map, pos, move)
        if dead:
            return -9999
        score = -light_evaluate(
            bit_map,
            new_pos if maximizing else our_pos,
            enemy_pos if maximizing else new_pos,
        )

        undo_move_inplace(bit_map, changed)
        return score

    moves.sort(key=move_score, reverse=maximizing)

    best_value = -float('inf') if maximizing else float('inf')
    best_moves = []

    for move in moves:
        new_pos, dead, changed = apply_move_inplace(bit_map, pos, move)

        if dead:
            val = -99999 if maximizing else 99999
        else:
            boost_used = ":BOOST" in move
            val, _ = alphabeta(
                bit_map,
                new_pos if maximizing else our_pos,
                enemy_pos if maximizing else new_pos,
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
            
        undo_move_inplace(bit_map, changed)

    # Choose randomly among tied best moves
    best_move = random.choice(best_moves) if best_moves else None
    return best_value, best_move


# ----------------------
# Choose Next Move
# ----------------------
def choose_next_move(game: Game, player_number=1, max_depth=4, time_limit=3.8):
    our_agent = game.agent1 if player_number == 1 else game.agent2
    enemy_agent = game.agent2 if player_number == 1 else game.agent1

    bit_map = game_to_bit_map(game, player_number)
    our_pos = tuple(our_agent.trail[-1])
    enemy_pos = tuple(enemy_agent.trail[-1])
    our_boosts = getattr(our_agent, "boosts_remaining", 0)
    enemy_boosts = getattr(enemy_agent, "boosts_remaining", 0)
    move_number = game.turns

    start_time = time.time()
    best_move = None
    best_score = float("-inf")
    fallback_move = None
    fallback_score = float("-inf")

    try:
        depth = 4
        while depth <= max_depth:
            fallback_move = best_move
            fallback_score = best_score
            best_score, best_move = alphabeta(
                bit_map, our_pos, enemy_pos,
                our_boosts, enemy_boosts,
                depth,
                start_time=start_time,
                time_limit=time_limit,
                turn_count=move_number
            )
            depth += 2
    except TimeoutError:
        pass

    if fallback_score > best_score:
        best_move = fallback_move

    if not best_move:
        print("No best")
        moves = generate_moves(bit_map, our_pos, our_boosts)
        best_move = random.choice(moves) if moves else "UP"

    return best_move
