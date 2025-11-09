import requests
import sys
import time
import os
from case_closed_game import Game, Direction, GameResult
import random
from tqdm import tqdm


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

def kill_agents():
    for p in agent_processes:
        if p.poll() is None:  # still running
            print(f"Terminating agent PID={p.pid}")
            p.terminate()
            try:
                p.wait(timeout=5)
            except subprocess.TimeoutExpired:
                print(f"Force killing agent PID={p.pid}")
                p.kill()

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
            return "forfeit"
        
        direction = direction_map[direction_str]
        
        # Check if move is opposite to current direction (invalid move)
        agent = self.game.agent1 if player_num == 1 else self.game.agent2
        current_dir = agent.direction
        
        # Check if requested direction is opposite to current
        cur_dx, cur_dy = current_dir.value
        req_dx, req_dy = direction.value
        if (req_dx, req_dy) == (-cur_dx, -cur_dy):
            direction = current_dir
            direction_str = {Direction.UP: 'UP', Direction.DOWN: 'DOWN', 
                           Direction.LEFT: 'LEFT', Direction.RIGHT: 'RIGHT'}[direction]
        
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
    time.sleep(2)

    # Get agent URLs from environment variables
    PLAYER1_URL = os.getenv("PLAYER1_URL", "http://localhost:5008")
    PLAYER2_URL = os.getenv("PLAYER2_URL", "http://localhost:5009")

    # Creating judge
    judge = Judge(PLAYER1_URL, PLAYER2_URL)

    # Check connectivity and latency
    if not judge.check_latency():
        return
    
    # Send initial state to both players
    if not judge.send_state(1) or not judge.send_state(2):
        return

    # Random moves left for p1 and p2
    p1_random = 5
    p2_random = 5

    # Game loop
    while True:
        # Get moves from both players
        p1_move = None
        p2_move = None
        p1_boost = False
        p2_boost = False
        
        # Player 1 move
        for attempt in range(1, 3):  # 2 attempts
            p1_move = judge.get_move(1, attempt, p1_random)
            if p1_move:
                validation = judge.handle_move(p1_move, 1, is_random=False)
                if validation == "forfeit":
                    judge.end_game(GameResult.AGENT2_WIN)
                    return
                elif validation:
                    p1_boost = validation[1]  # Extract boost flag
                    p1_direction = validation[2]  # Extract direction
                    break
        
        # If both attempts failed, use random move or forfeit
        if not p1_move or not validation:
            if p1_random > 0:
                random_agent = RandomPlayer(1)
                p1_direction = random_agent.get_best_move()
                p1_random -= 1
                # Convert Direction to string for handle_move
                dir_to_str = {Direction.UP: 'UP', Direction.DOWN: 'DOWN', Direction.LEFT: 'LEFT', Direction.RIGHT: 'RIGHT'}
                validation = judge.handle_move(dir_to_str[p1_direction], 1, is_random=True)
                p1_boost = False  # Random moves don't use boost
            else:
                judge.end_game(GameResult.AGENT2_WIN)
                return
        else:
            # Direction already extracted from validation
            pass
        
        # Player 2 move
        for attempt in range(1, 3):  # 2 attempts
            p2_move = judge.get_move(2, attempt, p2_random)
            if p2_move:
                validation = judge.handle_move(p2_move, 2, is_random=False)
                if validation == "forfeit":
                    judge.end_game(GameResult.AGENT1_WIN)
                    return
                elif validation:
                    p2_boost = validation[1]  # Extract boost flag
                    p2_direction = validation[2]  # Extract direction
                    break
        
        # If both attempts failed, use random move or forfeit
        if not p2_move or not validation:
            if p2_random > 0:
                random_agent = RandomPlayer(2)
                p2_direction = random_agent.get_best_move()
                p2_random -= 1
                # Convert Direction to string for handle_move
                dir_to_str = {Direction.UP: 'UP', Direction.DOWN: 'DOWN', Direction.LEFT: 'LEFT', Direction.RIGHT: 'RIGHT'}
                validation = judge.handle_move(dir_to_str[p2_direction], 2, is_random=True)
                p2_boost = False  # Random moves don't use boost
            else:
                judge.end_game(GameResult.AGENT1_WIN)
                return
        else:
            # Direction already extracted from validation
            pass
        
        # Execute both moves simultaneously
        result = judge.game.step(p1_direction, p2_direction, p1_boost, p2_boost)
        
        # Send updated state to both players
        judge.send_state(1)
        judge.send_state(2)

        # Check for game end
        if result is not None:
            judge.end_game(result)
            return result
        
        # Check for max turns (safety)
        if judge.game.turns >= 500:
            judge.end_game(GameResult.DRAW)
            return result

agents = ["defense_king.py", "agents/v_agent.py", "agents/m_agent.py", "agents/random_agent.py"]
test_agent = agents[2]
opp_agent = agents[1]
NUM_TESTS = 10

def run_all_games():
    wins = 0
    total_games = NUM_TESTS * 2

    with tqdm(total=total_games, desc="Running games") as pbar:
        # As Agent 1
        os.environ["PORT"] = "5008"
        run_agent(test_agent)
        time.sleep(0.5)
        os.environ["PORT"] = "5009"
        run_agent(opp_agent)
        time.sleep(0.5)
        for _ in range(NUM_TESTS):

            result = main()
            if result == GameResult.AGENT1_WIN:
                wins += 1

            pbar.update(1)
        kill_agents()
        
        print(f"{test_agent} won {wins}/{NUM_TESTS} games ({wins/NUM_TESTS:.2%}) against {opp_agent}")
        wins = 0
        os.environ["PORT"] = "5008"
        run_agent(opp_agent)
        time.sleep(0.5)
        os.environ["PORT"] = "5009"
        run_agent(test_agent)
        time.sleep(0.5)
        # As Agent 2
        for _ in range(NUM_TESTS):

            result = main()
            if result == GameResult.AGENT2_WIN:
                wins += 1

            pbar.update(1)
        kill_agents()

    print(f"{test_agent} won {wins}/{NUM_TESTS} games ({wins/NUM_TESTS:.2%}) against {opp_agent}")

if __name__ == "__main__":
    try:
        run_all_games()
    except KeyboardInterrupt:
        print("Quit Game")
    finally:
        kill_agents()
        print("Exit")
        sys.exit(0)