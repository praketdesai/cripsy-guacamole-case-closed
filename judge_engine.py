import requests
import sys
import time
import os
import random
import subprocess
import multiprocessing

from case_closed_game import Game, Direction, GameResult
from display import start_display_process, shared_state, state_lock

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

TIMEOUT = 5 # time for each move

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
        if not isinstance(move, str):
            print(f"Invalid move format by Player {player_num}: move must be a string")
            return "forfeit"
        move_parts = move.upper().split(':')
        direction_str = move_parts[0]
        use_boost = len(move_parts) > 1 and move_parts[1] == 'BOOST'
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
        agent = self.game.agent1 if player_num == 1 else self.game.agent2
        cur_dx, cur_dy = agent.direction.value
        req_dx, req_dy = direction.value
        if (req_dx, req_dy) == (-cur_dx, -cur_dy):
            print(f"Player {player_num} attempted invalid move (opposite direction). Using current direction instead.")
            direction = agent.direction
            direction_str = {Direction.UP: 'UP', Direction.DOWN: 'DOWN', 
                             Direction.LEFT: 'LEFT', Direction.RIGHT: 'RIGHT'}[direction]
        print(f"Player {player_num}'s move: {direction_str}{' (BOOST)' if use_boost else ''}{' (RANDOM)' if is_random else ''}")
        move_abbrev = {'UP': 'U', 'DOWN': 'D', 'LEFT': 'L', 'RIGHT': 'R'}
        boost_marker = 'B' if use_boost else ''
        random_marker = 'R' if is_random else ''
        self.game_str += f"{player_num}{move_abbrev[direction_str]}{boost_marker}{random_marker}-"
        return (True, use_boost, direction)


def main():
    print("Judge engine starting up, waiting for agents...")
    time.sleep(2)

    PLAYER1_URL = os.getenv("PLAYER1_URL", "http://localhost:5008")
    PLAYER2_URL = os.getenv("PLAYER2_URL", "http://localhost:5009")
    judge = Judge(PLAYER1_URL, PLAYER2_URL)

    if not judge.check_latency():
        print("Failed to connect to one or both players")
        return

    print(f"Player 1: {judge.p1_agent.agent_name} ({judge.p1_agent.participant})")
    print(f"Player 2: {judge.p2_agent.agent_name} ({judge.p2_agent.participant})")
    print(f"Initial latencies - P1: {judge.p1_agent.latency:.3f}s, P2: {judge.p2_agent.latency:.3f}s")

    # --- Prompt once at the start ---
    start_choice = input("Start new game or provide game string (leave empty for new game): ").strip()
    if start_choice:  # Game string provided
        if len(start_choice) % 2 != 0:
            print("Invalid game string: must be even number of moves")
            return
        judge.game.reset()
        judge.game_str = start_choice
        print("Starting game from provided game string...")
    else:  # New game
        judge.game.reset()
        judge.game_str = ""
        print("Starting new game...")

    # Send initial state
    judge.send_state(1)
    judge.send_state(2)

    p1_random = 5
    p2_random = 5

    # --- Game loop ---
    while True:
        print(f"\n=== Turn {judge.game.turns + 1} ===")
        # --- Player 1 move ---
        p1_move = judge.get_move(1, 1, p1_random)
        if not p1_move:
            if p1_random > 0:
                random_agent = RandomPlayer(1)
                p1_direction = random_agent.get_best_move()
                p1_random -= 1
                validation = judge.handle_move({Direction.UP:'UP', Direction.DOWN:'DOWN', Direction.LEFT:'LEFT', Direction.RIGHT:'RIGHT'}[p1_direction], 1, True)
                p1_boost = False
            else:
                judge.end_game(GameResult.AGENT2_WIN)
                print("Game String:", judge.game_str)
                return
        else:
            validation = judge.handle_move(p1_move, 1)
            p1_boost = validation[1]
            p1_direction = validation[2]

        # --- Player 2 move ---
        p2_move = judge.get_move(2, 1, p2_random)
        if not p2_move:
            if p2_random > 0:
                random_agent = RandomPlayer(2)
                p2_direction = random_agent.get_best_move()
                p2_random -= 1
                validation = judge.handle_move({Direction.UP:'UP', Direction.DOWN:'DOWN', Direction.LEFT:'LEFT', Direction.RIGHT:'RIGHT'}[p2_direction], 2, True)
                p2_boost = False
            else:
                judge.end_game(GameResult.AGENT1_WIN)
                print("Game String:", judge.game_str)
                return
        else:
            validation = judge.handle_move(p2_move, 2)
            p2_boost = validation[1]
            p2_direction = validation[2]

        result = judge.game.step(p1_direction, p2_direction, p1_boost, p2_boost)
        judge.send_state(1)
        judge.send_state(2)

        # Update visualizer shared state
        with lock:
            shared["width"] = judge.game.board.width
            shared["height"] = judge.game.board.height
            shared["board"] = [row[:] for row in judge.game.board.grid]  # copy of 2D grid
            shared["agent1_trail"] = judge.game.agent1.get_trail_positions()
            shared["agent2_trail"] = judge.game.agent2.get_trail_positions()
            shared["turn"] = judge.game.turns

        print(f"Agent 1: Trail={judge.game.agent1.length}, Alive={judge.game.agent1.alive}, Boosts={judge.game.agent1.boosts_remaining}")
        print(f"Agent 2: Trail={judge.game.agent2.length}, Alive={judge.game.agent2.alive}, Boosts={judge.game.agent2.boosts_remaining}")

        if result is not None:
            judge.end_game(result)
            print("Game String:", judge.game_str)
            break

        if judge.game.turns >= 2000:
            judge.end_game(GameResult.DRAW)
            print("Game String:", judge.game_str)
            break


if __name__ == "__main__":
    manager = multiprocessing.Manager()
    shared = manager.dict()
    shared["width"] = 50
    shared["height"] = 50
    shared["agent1_trail"] = manager.list()
    shared["agent2_trail"] = manager.list()
    shared["board"] = manager.list()
    shared["turn"] = 0

    lock = manager.Lock()

    viz_proc = start_display_process(shared, lock, host="0.0.0.0", port=5000)
    print("Visualizer started in PID:", viz_proc.pid)

    os.environ["PORT"] = "5008"
    run_agent("agents/random_agent.py")
    time.sleep(0.5)
    os.environ["PORT"] = "5009"
    run_agent("agents/random_agent.py")
    time.sleep(0.5)

    try:
        main()
    except KeyboardInterrupt:
        print("Quit Game")
    finally:
        for p in agent_processes:
            if p.poll() is None:
                print(f"Terminating agent PID={p.pid}")
                p.terminate()
                try:
                    p.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    print(f"Force killing agent PID={p.pid}")
                    p.kill()
        print("Exit")
