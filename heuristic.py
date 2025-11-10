import random
from util import generate_moves, apply_move

def random_heuristic(bit_map, our_pos, enemy_pos, boosts):
    possible_moves = generate_moves(bit_map, our_pos, boosts)

    # Filter out moves that would immediately collide
    safe_moves = []
    for move in possible_moves:
        _, _, collision = apply_move(bit_map, our_pos, move)
        if not collision:
            safe_moves.append(move)

    if not safe_moves:
        # No safe moves left; just pick a random normal move (will collide)
        return random.choice(possible_moves) if possible_moves else None

    # Return one safe move randomly
    return random.choice(safe_moves)