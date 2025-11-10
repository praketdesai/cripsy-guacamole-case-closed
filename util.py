import numpy as np
import random
from case_closed_game import Direction, Game, GameBoard

# ------------------
# Board Constants
# ------------------
_default_board = GameBoard()
BOARD_HEIGHT = _default_board.height
BOARD_WIDTH = _default_board.width
MOVES = list(Direction)
DELTAS = {d.name: d.value for d in Direction}

# Opposite directions (cannot reverse)
OPPOSITE = {
    'UP': 'DOWN',
    'DOWN': 'UP',
    'LEFT': 'RIGHT',
    'RIGHT': 'LEFT'
}

# ------------------
# Board Conversion
# ------------------
def game_to_bit_map(game: Game, player_number: int = 1) -> np.ndarray:
    """
    Convert the current game state to a bit map.
    1 = occupied by a trail, 0 = free space.
    """
    bit_map = np.zeros((game.board.height, game.board.width), dtype=np.uint8)

    trails = game.agent1.get_trail_positions() + game.agent2.get_trail_positions()
    for x, y in trails:
        if 0 <= x < game.board.width and 0 <= y < game.board.height:
            bit_map[y, x] = 1

    return bit_map

# ----------------------
# Move Generation
# ----------------------
def generate_moves(bit_map, pos, current_dir=None, boosts=0):
    """
    Returns a list of valid moves from pos (x,y), avoiding collisions and reversals.
    """
    H, W = bit_map.shape
    x, y = pos
    moves = []

    for name, (dx, dy) in DELTAS.items():
        # Skip reversing
        if current_dir and name == OPPOSITE.get(current_dir):
            continue

        nx1, ny1 = (x + dx) % W, (y + dy) % H
        if bit_map[ny1, nx1] == 0:
            moves.append(name)
            if boosts > 0:
                nx2, ny2 = (x + 2*dx) % W, (y + 2*dy) % H
                if bit_map[ny2, nx2] == 0:
                    moves.append(f"{name}:BOOST")

    return moves

# ----------------------
# Move Application
# ----------------------
def apply_move(bit_map, pos, move, current_dir=None):
    """
    Apply a move to the bit map.
    Returns (new_map, new_position, collision_flag)
    """
    H, W = bit_map.shape
    new_map = bit_map.copy()
    boost = ":BOOST" in move
    base_move = move.replace(":BOOST", "")

    # Reject invalid or reverse moves
    if base_move not in DELTAS:
        return new_map, pos, True
    if current_dir and base_move == OPPOSITE.get(current_dir):
        return new_map, pos, True

    dx, dy = DELTAS[base_move]
    steps = 2 if boost else 1
    x, y = pos

    for _ in range(steps):
        nx, ny = (x + dx) % W, (y + dy) % H
        if new_map[ny, nx]:  # Collision
            return new_map, (x, y), True
        x, y = nx, ny
        new_map[y, x] = 1  # Mark trail

    return new_map, (x, y), False

# ----------------------
# Random Safe Move Heuristic
# ----------------------
def random_move(bit_map, pos, current_dir=None, boosts=0):
    """
    Chooses a random valid move that avoids collisions and reversals.
    """
    moves = generate_moves(bit_map, pos, current_dir, boosts)
    if not moves:
        return None  # No valid moves
    return random.choice(moves)
