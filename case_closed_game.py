import random
from collections import deque
from enum import Enum
from typing import Optional

EMPTY = 0
AGENT = 1

"""
GameBoard class manages the game board.

Handles the 2D grid, state of each cell, and provides torus (wraparound)
functionality for all coordinate-based operations.
"""
class GameBoard:
    def __init__(self, height: int = 50, width: int = 50):
        self.height = height
        self.width = width
        self.grid = [[EMPTY for _ in range(width)] for _ in range(height)]

    def _torus_check(self, position: tuple[int, int]) -> tuple[int, int]:
        x, y = position
        normalized_x = x % self.width
        normalized_y = y % self.height
        return (normalized_x, normalized_y)
    
    def get_cell_state(self, position: tuple[int, int]) -> int:
        x, y = self._torus_check(position)
        return self.grid[y][x]

    def set_cell_state(self, position: tuple[int, int], state: int):
        x, y = self._torus_check(position)
        self.grid[y][x] = state

    def get_random_empty_cell(self) -> tuple[int, int] | None:
        empty_cells = []
        for y in range(self.height):
            for x in range(self.width):
                if self.grid[y][x] == EMPTY:
                    empty_cells.append((x, y))
        
        if not empty_cells:
            return None
        
        return random.choice(empty_cells)

    def __str__(self) -> str:
        chars = {EMPTY: '.', AGENT: 'A'}
        board_str = ""
        for y in range(self.height):
            for x in range(self.width):
                board_str += chars.get(self.grid[y][x], '?') + ' '
            board_str += '\n'
        return board_str

class Direction(Enum):
    UP = (0, -1)
    DOWN = (0, 1)
    RIGHT = (1, 0)
    LEFT = (-1, 0)

class GameResult(Enum):
    AGENT1_WIN = 1
    AGENT2_WIN = 2
    DRAW = 3

class Agent:
    '''This class represents an agent in the game. It manages the agent's trail using a deque.'''
    def __init__(self, agent_id: str, start_pos: tuple[int, int], start_dir: Direction, board: GameBoard):
        self.agent_id = agent_id
        second = (start_pos[0] + start_dir.value[0], start_pos[1] + start_dir.value[1])
        self.trail = deque([start_pos, second])  # Trail of positions
        self.direction = start_dir
        self.board = board
        self.alive = True
        self.length = 2  # Initial length of the trail
        self.boosts_remaining = 3  # Each agent gets 3 speed boosts

        self.board.set_cell_state(start_pos, AGENT)
        self.board.set_cell_state(second, AGENT)
    
    def is_head(self, position: tuple[int, int]) -> bool:
        return position == self.trail[-1]
    
    def calculate_new_positions(self, direction: Direction, use_boost: bool = False) -> list[tuple[int, int]]:
        """
        Calculate where the agent would move without actually moving.
        Returns list of new positions (1 if normal, 2 if boosted).
        """
        if not self.alive:
            return []
        
        if use_boost and self.boosts_remaining <= 0:
            use_boost = False
        
        num_moves = 2 if use_boost else 1
        new_positions = []
        current_head = self.trail[-1]
        current_direction = self.direction
        
        for move_num in range(num_moves):
            # Check if direction is opposite to current direction
            cur_dx, cur_dy = current_direction.value
            req_dx, req_dy = direction.value
            if (req_dx, req_dy) == (-cur_dx, -cur_dy):
                # Invalid move, skip
                continue
            
            dx, dy = direction.value
            new_head = (current_head[0] + dx, current_head[1] + dy)
            new_head = self.board._torus_check(new_head)
            new_positions.append(new_head)
            
            # Update for next iteration
            current_head = new_head
            current_direction = direction
        
        return new_positions
    
    def execute_move(self, new_positions: list[tuple[int, int]], direction: Direction, use_boost: bool = False):
        """
        Execute a pre-calculated move by adding positions to trail.
        Does NOT check collisions - that's done separately.
        """
        if not self.alive or not new_positions:
            return
        
        # Update direction
        self.direction = direction
        
        # Use boost
        if use_boost and self.boosts_remaining > 0:
            self.boosts_remaining -= 1
        
        # Add new positions to trail and board
        for pos in new_positions:
            self.trail.append(pos)
            self.length += 1
            self.board.set_cell_state(pos, AGENT)

    def get_trail_positions(self) -> list[tuple[int, int]]:
        return list(self.trail)
    

class Game:
    def __init__(self):
        self.board = GameBoard()
        
        # Random starting positions
        self.start_pos_agent1 = self.board.get_random_empty_cell()
        self.start_pos_agent2 = self.board.get_random_empty_cell()
        
        # Ensure they are not the same
        while self.start_pos_agent2 == self.start_pos_agent1:
            self.start_pos_agent2 = self.board.get_random_empty_cell()
        
        # Random starting directions
        self.start_dir_agent1 = random.choice(list(Direction))
        self.start_dir_agent2 = random.choice(list(Direction))
        
        # Create agents with saved positions
        self.agent1 = Agent(agent_id=1, start_pos=self.start_pos_agent1, start_dir=self.start_dir_agent1, board=self.board)
        self.agent2 = Agent(agent_id=2, start_pos=self.start_pos_agent2, start_dir=self.start_dir_agent2, board=self.board)
        
        self.turns = 0

    def reset(self):
        """Resets the game to the initial state with same starting positions."""
        self.board = GameBoard()
        
        # Reuse saved start positions and directions
        self.agent1 = Agent(agent_id=1, start_pos=self.start_pos_agent1, start_dir=self.start_dir_agent1, board=self.board)
        self.agent2 = Agent(agent_id=2, start_pos=self.start_pos_agent2, start_dir=self.start_dir_agent2, board=self.board)
        
        self.turns = 0

    
    def check_collisions(self, agent1_new_positions: list[tuple[int, int]], 
                        agent2_new_positions: list[tuple[int, int]]) -> tuple[bool, bool]:
        """
        Check if either agent collides after their moves.
        Returns (agent1_alive, agent2_alive)
        """
        agent1_alive = True
        agent2_alive = True
        
        # Get final positions after all moves
        agent1_final = agent1_new_positions[-1] if agent1_new_positions else None
        agent2_final = agent2_new_positions[-1] if agent2_new_positions else None
        
        # Check head-on collision (both agents move to same position)
        if agent1_final and agent2_final and agent1_final == agent2_final:
            print("Head-on collision!")
            return False, False
        
        # Check if agent1 hits anything
        if agent1_final:
            # Check collision with own trail (before the move)
            if agent1_final in self.agent1.trail:
                agent1_alive = False
            
            # Check collision with agent2's trail (before the move)
            if agent1_final in self.agent2.trail:
                agent1_alive = False
            
            # Check intermediate positions if boosting
            for pos in agent1_new_positions[:-1]:
                if pos in self.agent1.trail or pos in self.agent2.trail:
                    agent1_alive = False
                    break
        
        # Check if agent2 hits anything
        if agent2_final:
            # Check collision with own trail (before the move)
            if agent2_final in self.agent2.trail:
                agent2_alive = False
            
            # Check collision with agent1's trail (before the move)
            if agent2_final in self.agent1.trail:
                agent2_alive = False
            
            # Check intermediate positions if boosting
            for pos in agent2_new_positions[:-1]:
                if pos in self.agent2.trail or pos in self.agent1.trail:
                    agent2_alive = False
                    break
        
        return agent1_alive, agent2_alive
    
    def step(self, dir1: Direction, dir2: Direction, boost1: bool = False, boost2: bool = False):
        """Advances the game by one step, moving both agents simultaneously."""
        if self.turns >= 2000:
            print("Max turns reached. Checking trail lengths...")
            if self.agent1.length > self.agent2.length:
                print(f"Agent 1 wins with trail length {self.agent1.length} vs {self.agent2.length}")
                return GameResult.AGENT1_WIN
            elif self.agent2.length > self.agent1.length:
                print(f"Agent 2 wins with trail length {self.agent2.length} vs {self.agent1.length}")
                return GameResult.AGENT2_WIN
            else:
                print(f"Draw - both agents have trail length {self.agent1.length}")
                return GameResult.DRAW
        
        # Phase 1: Calculate new positions for both agents (without moving)
        agent1_new_positions = self.agent1.calculate_new_positions(dir1, boost1)
        agent2_new_positions = self.agent2.calculate_new_positions(dir2, boost2)
        
        # Phase 2: Check collisions based on calculated positions
        agent1_alive, agent2_alive = self.check_collisions(agent1_new_positions, agent2_new_positions)
        
        # Phase 3: Execute moves only if agent survives
        if agent1_alive and agent1_new_positions:
            self.agent1.execute_move(agent1_new_positions, dir1, boost1)
        else:
            self.agent1.alive = False
        
        if agent2_alive and agent2_new_positions:
            self.agent2.execute_move(agent2_new_positions, dir2, boost2)
        else:
            self.agent2.alive = False
        
        # Phase 4: Determine game result
        if not agent1_alive and not agent2_alive:
            print("Both agents have crashed.")
            return GameResult.DRAW
        elif not agent1_alive:
            print("Agent 1 has crashed.")
            return GameResult.AGENT2_WIN
        elif not agent2_alive:
            print("Agent 2 has crashed.")
            return GameResult.AGENT1_WIN

        self.turns += 1
        return None