import requests
import sys
import time
import os
from case_closed_game import Game, Direction, GameResult
import random

import visualizer


import threading
import subprocess
import time

# List to track agent processes
agent_processes = []

def run_agent(script_path):
    """Start an agent subprocess"""
    with open(os.devnull, "w") as devnull:
        proc = subprocess.Popen(["python", script_path])
        agent_processes.append(proc)

class RandomPlayer:
    def __init__(self, player_id=1):
        self.player_id = player_id
    
    def get_possible_moves(self):
        """Returns list of all possible directions for agent."""
        return [Direction.UP, Direction.DOWN, Direction.LEFT, Direction.RIGHT]
        
    def get_best_move(self):
        """Returns a random valid direction."""
        possible_moves = self.get_possible_moves()
        return random.choice(possible_moves)

TIMEOUT = 4  # time for each move

class PlayerAgent:
    def __init__(self, participant, agent_name):
        self.participant = participant
        self.agent_name = agent_name
        self.latency = None

class Judge:
    def __init__(self, p1_url, p2_url):
        self.p1_url = p1_url
        self.p2_url = p2_url
        self.game = Game()
        self.p1_agent = None
        self.p2_agent = None
        self.game_str = ""  # Track game moves as string

    def check_latency(self):
        """Check latency for both players and create their agents"""
        # Check P1
        try:
            start_time = time.time()
            response = requests.get(self.p1_url, timeout=TIMEOUT)
            end_time = time.time()
            
            if response.status_code == 200:
                data = response.json()
                self.p1_agent = PlayerAgent(data.get("participant", "Participant1"), 
                                     data.get("agent_name", "Agent1"))
                self.p1_agent.latency = (end_time - start_time)
            else:
                return False
                
        except (requests.RequestException, requests.Timeout):
            return False

        # Check P2
        try:
            start_time = time.time()
            response = requests.get(self.p2_url, timeout=TIMEOUT)
            end_time = time.time()
            
            if response.status_code == 200:
                data = response.json()
                self.p2_agent = PlayerAgent(data.get("participant", "Participant2"), 
                                     data.get("agent_name", "Agent2"))
                self.p2_agent.latency = (end_time - start_time)
            else:
                return False
                
        except (requests.RequestException, requests.Timeout):
            return False

        return True

    def send_state(self, player_num):
        """Send current game state to a player via POST"""
        url = self.p1_url if player_num == 1 else self.p2_url
        
        state_data = {
            "board": self.game.board.grid,
            "agent1_trail": self.game.agent1.get_trail_positions(),
            "agent2_trail": self.game.agent2.get_trail_positions(),
            "agent1_length": self.game.agent1.length,
            "agent2_length": self.game.agent2.length,
            "agent1_alive": self.game.agent1.alive,
            "agent2_alive": self.game.agent2.alive,
            "agent1_boosts": self.game.agent1.boosts_remaining,
            "agent2_boosts": self.game.agent2.boosts_remaining,
            "turn_count": self.game.turns,
            "player_number": player_num,
        }
        
        try:
            response = requests.post(f"{url}/send-state", json=state_data, timeout=TIMEOUT)
            return response.status_code == 200
        except (requests.RequestException, requests.Timeout):
            return False

    def get_move(self, player_num, attempt_number, random_moves_left):
        """Request a move from a player via GET with query parameters"""
        url = self.p1_url if player_num == 1 else self.p2_url
        
        # Build query parameters for GET request
        params = {
            "player_number": player_num,
            "attempt_number": attempt_number,
            "random_moves_left": random_moves_left,
            "turn_count": self.game.turns,
        }
        
        try:
            start_time = time.time()
            response = requests.get(f"{url}/send-move", params=params, timeout=TIMEOUT)
            end_time = time.time()
            
            if player_num == 1:
                self.p1_agent.latency = (end_time - start_time)
            else:
                self.p2_agent.latency = (end_time - start_time)
            
            if response.status_code == 200:
                move = response.json()
                return move.get('move')
            else:
                return None
                
        except (requests.RequestException, requests.Timeout):
            return None

    def end_game(self, result):
        """End the game and notify both players"""
        end_data = {
            "board": self.game.board.grid,
            "agent1_trail": self.game.agent1.get_trail_positions(),
            "agent2_trail": self.game.agent2.get_trail_positions(),
            "agent1_length": self.game.agent1.length,
            "agent2_length": self.game.agent2.length,
            "agent1_alive": self.game.agent1.alive,
            "agent2_alive": self.game.agent2.alive,
            "agent1_boosts": self.game.agent1.boosts_remaining,
            "agent2_boosts": self.game.agent2.boosts_remaining,
            "turn_count": self.game.turns,
            "result": result.name if isinstance(result, GameResult) else str(result),
        }
        
        try:
            requests.post(f"{self.p1_url}/end", json=end_data, timeout=TIMEOUT)
            requests.post(f"{self.p2_url}/end", json=end_data, timeout=TIMEOUT)
            
            if isinstance(result, GameResult):
                if result == GameResult.AGENT1_WIN:
                    print(f"Winner: Agent 1 ({self.p1_agent.agent_name})")
                elif result == GameResult.AGENT2_WIN:
                    print(f"Winner: Agent 2 ({self.p2_agent.agent_name})")
                else:
                    print("Game ended in a draw")
            else:
                print(f"Game ended: {result}")
        except (requests.RequestException, requests.Timeout):
            return False

    def handle_move(self, move, player_num, is_random=False):
        """Validate and execute a move. Returns 'forfeit' or tuple (valid, boost_flag, direction)"""
        
        # Validate move format
        if not isinstance(move, str):
            print(f"Invalid move format by Player {player_num}: move must be a string")
            return "forfeit"
        
        # Parse move - can be "DIRECTION" or "DIRECTION:BOOST"
        move_parts = move.upper().split(':')
        direction_str = move_parts[0]
        use_boost = len(move_parts) > 1 and move_parts[1] == 'BOOST'
        
        # Convert move string to Direction
        direction_map = {
            'UP': Direction.UP,
            'DOWN': Direction.DOWN,
            'LEFT': Direction.LEFT,
            'RIGHT': Direction.RIGHT,
        }
        
        if direction_str not in direction_map:
            print(f"Invalid direction by Player {player_num}: {direction_str}")
            return "forfeit"
        
        direction = direction_map[direction_str]
        
        # Check if move is opposite to current direction (invalid move)
        agent = self.game.agent1 if player_num == 1 else self.game.agent2
        current_dir = agent.direction
        
        # Check if requested direction is opposite to current
        cur_dx, cur_dy = current_dir.value
        req_dx, req_dy = direction.value
        if (req_dx, req_dy) == (-cur_dx, -cur_dy):
            print(f"Player {player_num} attempted invalid move (opposite direction). Using current direction instead.")
            direction = current_dir
            direction_str = {Direction.UP: 'UP', Direction.DOWN: 'DOWN', 
                           Direction.LEFT: 'LEFT', Direction.RIGHT: 'RIGHT'}[direction]
        
        print(f"Player {player_num}'s move: {direction_str}{' (BOOST)' if use_boost else ''}{' (RANDOM)' if is_random else ''}")
        
        # Record move in game string with improved format
        move_abbrev = {'UP': 'U', 'DOWN': 'D', 'LEFT': 'L', 'RIGHT': 'R'}
        boost_marker = 'B' if use_boost else ''
        random_marker = 'R' if is_random else ''
        self.game_str += f"{player_num}{move_abbrev[direction_str]}{boost_marker}{random_marker}-"
        
        return (True, use_boost, direction)  # Return tuple: (valid, boost_flag, direction)


from collections import deque

def get_voronoi_board(board, agent1_trail, agent2_trail):
    height = len(board)
    width = len(board[0])

    dist1 = [[None]*width for _ in range(height)]
    dist2 = [[None]*width for _ in range(height)]

    # mark trails as blocked
    blocked = [[0]*width for _ in range(height)]
    for x, y in agent1_trail + agent2_trail:
        blocked[y][x] = 1

    def bfs(trail, dist_map):
        queue = deque()
        for x, y in trail:
            dist_map[y][x] = 0  # distance starts at 0
            queue.append((x, y))
        while queue:
            x, y = queue.popleft()
            d = dist_map[y][x] + 1
            for dx, dy in [(-1,0),(1,0),(0,-1),(0,1)]:
                nx = (x + dx) % width
                ny = (y + dy) % height
                if blocked[ny][nx]:
                    continue
                if dist_map[ny][nx] is None:
                    dist_map[ny][nx] = d
                    queue.append((nx, ny))

    bfs([agent1_trail[-1]], dist1)  # only BFS from head, not entire trail
    bfs([agent2_trail[-1]], dist2)

    # build Voronoi
    voronoi = [[0]*width for _ in range(height)]
    for y in range(height):
        for x in range(width):
            if blocked[y][x]:
                voronoi[y][x] = 1  # trail
            else:
                d1 = dist1[y][x]
                d2 = dist2[y][x]
                if d1 is None and d2 is None:
                    voronoi[y][x] = 0
                elif d1 is None:
                    voronoi[y][x] = 4
                elif d2 is None:
                    voronoi[y][x] = 3
                elif d1 < d2:
                    voronoi[y][x] = 3
                elif d2 < d1:
                    voronoi[y][x] = 4
                else:
                    voronoi[y][x] = 5
    return voronoi
   


def main():
    print("Judge engine starting up, waiting for agents...")
    time.sleep(5)

    # Get agent URLs from environment variables
    # PLAYER1_URL = os.getenv("PLAYER1_URL", "http://localhost:5008")
    # PLAYER1_URL = os.getenv("PLAYER2_URL", "https://algebra-captured-helpful-suited.trycloudflare.com/")
    PLAYER1_URL = os.getenv("PLAYER1_URL", "https://grateful-layout-dietary-oven.trycloudflare.com/")
    PLAYER2_URL = os.getenv("PLAYER2_URL", "http://localhost:5009")
    # PLAYER2_URL = os.getenv("PLAYER2_URL", "https://algebra-captured-helpful-suited.trycloudflare.com/")
    # PLAYER2_URL = os.getenv("PLAYER2_URL", "https://grateful-layout-dietary-oven.trycloudflare.com/")

    # Creating judge
    print(f"Creating judge for {PLAYER1_URL} and {PLAYER2_URL}...")
    judge = Judge(PLAYER1_URL, PLAYER2_URL)

    # Check connectivity and latency
    if not judge.check_latency():
        print("Failed to connect to one or both players")
        return
        
    print(f"Player 1: {judge.p1_agent.agent_name} ({judge.p1_agent.participant})")
    print(f"Player 2: {judge.p2_agent.agent_name} ({judge.p2_agent.participant})")
    print(f"Initial latencies - P1: {judge.p1_agent.latency:.3f}s, P2: {judge.p2_agent.latency:.3f}s")
    
    # Send initial state to both players
    print("Sending initial game state...")
    if not judge.send_state(1) or not judge.send_state(2):
        print("Failed to send initial state")
        return

    # Random moves left for p1 and p2
    p1_random = 5
    p2_random = 5

    # Game loop
    while True:
        print(f"\n=== Turn {judge.game.turns + 1} ===")
        # Get moves from both players
        p1_move = None
        p2_move = None
        p1_boost = False
        p2_boost = False
        
        # Player 1 move
        print("Requesting move from Player 1...")
        for attempt in range(1, 3):  # 2 attempts
            p1_move = judge.get_move(1, attempt, p1_random)
            if p1_move:
                validation = judge.handle_move(p1_move, 1, is_random=False)
                if validation == "forfeit":
                    print("Player 1 forfeited")
                    judge.end_game(GameResult.AGENT2_WIN)
                    print("Game String:", judge.game_str)
                    return
                elif validation:
                    p1_boost = validation[1]  # Extract boost flag
                    p1_direction = validation[2]  # Extract direction
                    break
            print(f"  Attempt {attempt} failed")
        
        # If both attempts failed, use random move or forfeit
        if not p1_move or not validation:
            if p1_random > 0:
                print(f"Using random move for Player 1 ({p1_random} random moves left)")
                random_agent = RandomPlayer(1)
                p1_direction = random_agent.get_best_move()
                p1_random -= 1
                # Convert Direction to string for handle_move
                dir_to_str = {Direction.UP: 'UP', Direction.DOWN: 'DOWN', Direction.LEFT: 'LEFT', Direction.RIGHT: 'RIGHT'}
                validation = judge.handle_move(dir_to_str[p1_direction], 1, is_random=True)
                p1_boost = False  # Random moves don't use boost
            else:
                print("Player 1 has no random moves left. Forfeiting.")
                judge.end_game(GameResult.AGENT2_WIN)
                print("Game String:", judge.game_str)
                return
        else:
            # Direction already extracted from validation
            pass
        
        # Player 2 move
        print("Requesting move from Player 2...")
        for attempt in range(1, 3):  # 2 attempts
            p2_move = judge.get_move(2, attempt, p2_random)
            if p2_move:
                validation = judge.handle_move(p2_move, 2, is_random=False)
                if validation == "forfeit":
                    print("Player 2 forfeited")
                    judge.end_game(GameResult.AGENT1_WIN)
                    print("Game String:", judge.game_str)
                    return
                elif validation:
                    p2_boost = validation[1]  # Extract boost flag
                    p2_direction = validation[2]  # Extract direction
                    break
            print(f"  Attempt {attempt} failed")
        
        # If both attempts failed, use random move or forfeit
        if not p2_move or not validation:
            if p2_random > 0:
                print(f"Using random move for Player 2 ({p2_random} random moves left)")
                random_agent = RandomPlayer(2)
                p2_direction = random_agent.get_best_move()
                p2_random -= 1
                # Convert Direction to string for handle_move
                dir_to_str = {Direction.UP: 'UP', Direction.DOWN: 'DOWN', Direction.LEFT: 'LEFT', Direction.RIGHT: 'RIGHT'}
                validation = judge.handle_move(dir_to_str[p2_direction], 2, is_random=True)
                p2_boost = False  # Random moves don't use boost
            else:
                print("Player 2 has no random moves left. Forfeiting.")
                judge.end_game(GameResult.AGENT1_WIN)
                print("Game String:", judge.game_str)
                return
        else:
            # Direction already extracted from validation
            pass
        
        # Execute both moves simultaneously
        result = judge.game.step(p1_direction, p2_direction, p1_boost, p2_boost)
        
        # Send updated state to both players
        judge.send_state(1)
        judge.send_state(2)
        
        # Display current board state
        # print(judge.game.board)
        print(f"Agent 1: Trail Length={judge.game.agent1.length}, Alive={judge.game.agent1.alive}, Boosts={judge.game.agent1.boosts_remaining}")
        print(f"Agent 2: Trail Length={judge.game.agent2.length}, Alive={judge.game.agent2.alive}, Boosts={judge.game.agent2.boosts_remaining}")
        
        # Update visualizer state
        with visualizer.state_lock:
            visualizer.shared_state['agent1_trail'] = judge.game.agent1.get_trail_positions()
            visualizer.shared_state['agent2_trail'] = judge.game.agent2.get_trail_positions()
            visualizer.shared_state['turn'] = judge.game.turns
            visualizer.shared_state['width'] = judge.game.board.width
            visualizer.shared_state['height'] = judge.game.board.height
            visualizer.shared_state['board'] = get_voronoi_board(
                judge.game.board.grid,
                judge.game.agent1.get_trail_positions(),
                judge.game.agent2.get_trail_positions()
            )


        # Check for game end
        if result is not None:
            judge.end_game(result)
            print("Game String:", judge.game_str)
            break
        
        # Check for max turns (safety)
        if judge.game.turns >= 500:
            print("Maximum turns reached")
            judge.end_game(GameResult.DRAW)
            print("Game String:", judge.game_str)
            break

if __name__ == "__main__":
    visualizer.start_visualizer_thread(host="0.0.0.0", port=5000)
    print("Visualizer running at http://localhost:5000")

    # Threads for agentsos.environ["PORT"] = "5008"
    # run_agent("agents/r_agent.py")
    time.sleep(0.5)  # let it start
    os.environ["PORT"] = "5009"
    run_agent("agents/r_agent.py")
    time.sleep(0.5)


    try:
        main()
    except KeyboardInterrupt:
        print("Quit Game")
    finally:
        for p in agent_processes:
            if p.poll() is None:  # still running
                print(f"Terminating agent PID={p.pid}")
                p.terminate()
                try:
                    p.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    print(f"Force killing agent PID={p.pid}")
                    p.kill()
        print("Exit")

    sys.exit(0)