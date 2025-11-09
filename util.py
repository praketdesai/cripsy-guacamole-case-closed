import numpy as np
from case_closed_game import Game

# ----------------------
# Board Constants
# ----------------------
BOARD_HEIGHT = 18
BOARD_WIDTH = 20
MOVES = ["UP", "DOWN", "LEFT", "RIGHT"]
DELTAS = {"UP": (0, -1), "DOWN": (0, 1), "LEFT": (-1, 0), "RIGHT": (1, 0)}

# ----------------------
# Board Conversion
# ----------------------
def game_to_bit_map(game: Game, player_number: int = 1) -> np.ndarray:
    """
    Convert the current game state to a bit map.
    1 = occupied by a trail, 0 = free space.

    Args:
        game (Game): The current game instance.
        player_number (int): 1 or 2, determines which agent is 'us'.

    Returns:
        np.ndarray: 2D array representing occupied/free cells.
    """
    bit_map = np.zeros((BOARD_HEIGHT, BOARD_WIDTH), dtype=np.uint8)

    # Get all trail positions
    trails = game.agent1.get_trail_positions() + game.agent2.get_trail_positions()
    for x, y in trails:
        if 0 <= x < BOARD_WIDTH and 0 <= y < BOARD_HEIGHT:
            bit_map[y, x] = 1

    return bit_map

# ----------------------
# Move Generation / Application
# ----------------------
def generate_moves(bit_map, pos, boosts=0):
    H, W = bit_map.shape
    x, y = pos
    moves = []

    for name, (dx, dy) in DELTAS.items():
        nx1, ny1 = (x + dx) % W, (y + dy) % H
        if bit_map[ny1, nx1] == 0:
            moves.append(name)
            if boosts > 0:
                nx2, ny2 = (x + 2*dx) % W, (y + 2*dy) % H
                if bit_map[ny2, nx2] == 0:
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
        if new_map[ny, nx]:  # Collision
            return new_map, (x, y), True
        x, y = nx, ny
        new_map[y, x] = 1  # Mark trail

    return new_map, (x, y), False
