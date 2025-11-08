# visual_judge.py
import requests
import sys
import time
import os
import random
import threading
from threading import Lock
from flask import Flask, jsonify, render_template_string
from case_closed_game import Game, Direction, GameResult

TIMEOUT = 4  # time for each move

# ---------- your existing RandomPlayer, PlayerAgent, Judge classes ----------
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


# ---------- Flask visualizer ----------

app = Flask(__name__)
state_lock = Lock()
shared_state = {
    "width": 20,
    "height": 20,
    "agent1_trail": [],
    "agent2_trail": [],
    "turn": 0,
}

# Simple HTML page with canvas and JS to poll /state
HTML_PAGE = """
<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>Case Closed - Visualizer</title>
  <style>
    body { font-family: sans-serif; margin: 10px; }
    canvas { background: #ffffff; border:1px solid #ddd; }
    #info { margin-top: 8px; }
  </style>
</head>
<body>
  <h3>Case Closed - Visual Visualizer</h3>
  <canvas id="board" width="600" height="600"></canvas>
  <div id="info">Turn: <span id="turn">0</span></div>

<script>
const canvas = document.getElementById('board');
const ctx = canvas.getContext('2d');

let gridW = 20;
let gridH = 20;

function draw(state) {
  gridW = state.width;
  gridH = state.height;
  const cellW = canvas.width / gridW;
  const cellH = canvas.height / gridH;

  // clear
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  // draw grid background
  ctx.fillStyle = "#ffffff";
  ctx.fillRect(0, 0, canvas.width, canvas.height);

  // optional grid lines
  ctx.strokeStyle = '#eee';
  for (let x=0; x<=gridW; x++) {
    ctx.beginPath();
    ctx.moveTo(x * cellW, 0);
    ctx.lineTo(x * cellW, canvas.height);
    ctx.stroke();
  }
  for (let y=0; y<=gridH; y++) {
    ctx.beginPath();
    ctx.moveTo(0, y * cellH);
    ctx.lineTo(canvas.width, y * cellH);
    ctx.stroke();
  }

  // draw trails first (so heads on top)
  // agent1 trail: reddish
  ctx.globalAlpha = 0.9;
  (state.agent1_trail || []).forEach((pos,i) => {
    const x = pos[0], y = pos[1];
    ctx.fillStyle = 'rgba(255,100,100,' + (0.35 + 0.65 * (i/(state.agent1_trail.length||1))) + ')';
    ctx.fillRect(x*cellW+1, y*cellH+1, cellW-2, cellH-2);
  });

  // agent2 trail: bluish
  (state.agent2_trail || []).forEach((pos,i) => {
    const x = pos[0], y = pos[1];
    ctx.fillStyle = 'rgba(100,120,255,' + (0.35 + 0.65 * (i/(state.agent2_trail.length||1))) + ')';
    ctx.fillRect(x*cellW+1, y*cellH+1, cellW-2, cellH-2);
  });

  // draw heads
  const a1 = state.agent1_trail && state.agent1_trail.length ? state.agent1_trail[state.agent1_trail.length - 1] : null;
  const a2 = state.agent2_trail && state.agent2_trail.length ? state.agent2_trail[state.agent2_trail.length - 1] : null;

  if (a1) {
    ctx.fillStyle = 'red';
    ctx.fillRect(a1[0]*cellW+1, a1[1]*cellH+1, cellW-2, cellH-2);
  }

  if (a2) {
    ctx.fillStyle = 'blue';
    ctx.fillRect(a2[0]*cellW+1, a2[1]*cellH+1, cellW-2, cellH-2);
  }

  document.getElementById('turn').innerText = state.turn;
  ctx.globalAlpha = 1.0;
}

async function fetchAndDraw() {
  try {
    const res = await fetch('/state');
    if (!res.ok) return;
    const state = await res.json();
    draw(state);
  } catch (e) {
    // ignore transient errors
  }
}

// poll every 250 ms
setInterval(fetchAndDraw, 250);
fetchAndDraw();
</script>
</body>
</html>
"""

@app.route('/')
def index():
    return render_template_string(HTML_PAGE)

@app.route('/state')
def get_state():
    with state_lock:
        return jsonify(shared_state)


# ---------- main: run judge and update shared_state ----------
def run_flask():
    # run without reloader inside a thread
    app.run(host='0.0.0.0', port=5000, debug=False, use_reloader=False)


def main():
    print("Judge engine starting up, waiting for agents...")
    time.sleep(1)

    PLAYER1_URL = os.getenv("PLAYER1_URL", "http://localhost:5008")
    PLAYER2_URL = os.getenv("PLAYER2_URL", "http://localhost:5009")

    judge = Judge(PLAYER1_URL, PLAYER2_URL)

    if not judge.check_latency():
        print("Failed to connect to one or both players")
        return

    print(f"Player 1: {judge.p1_agent.agent_name} ({judge.p1_agent.participant})")
    print(f"Player 2: {judge.p2_agent.agent_name} ({judge.p2_agent.participant})")
    print(f"Initial latencies - P1: {judge.p1_agent.latency:.3f}s, P2: {judge.p2_agent.latency:.3f}s")

    # Send initial state to both players
    print("Sending initial game state...")
    judge.send_state(1)
    judge.send_state(2)

    # initialize shared_state from judge.game
    with state_lock:
        # try to read width/height from board, fall back to 20
        try:
            w = judge.game.board.width
            h = judge.game.board.height
        except Exception:
            # attempt to infer from grid
            try:
                grid = judge.game.board.grid
                h = len(grid)
                w = len(grid[0]) if h>0 else 20
            except Exception:
                w, h = 20, 20

        shared_state['width'] = w
        shared_state['height'] = h
        shared_state['agent1_trail'] = judge.game.agent1.get_trail_positions()
        shared_state['agent2_trail'] = judge.game.agent2.get_trail_positions()
        shared_state['turn'] = judge.game.turns

    # start flask in background thread
    flask_thread = threading.Thread(target=run_flask, daemon=True)
    flask_thread.start()
    print("Flask visualizer started on http://localhost:5000/")

    # Random moves left for p1 and p2
    p1_random = 5
    p2_random = 5

    # Game loop (same logic as your original script)
    while True:
        time.sleep(0.5)
        print(f"\n=== Turn {judge.game.turns + 1} ===")

        # Get moves from both players
        p1_move = None
        p2_move = None
        p1_boost = False
        p2_boost = False
        validation = None

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
                    p1_boost = validation[1]
                    p1_direction = validation[2]
                    break
            print(f"  Attempt {attempt} failed")

        if not p1_move or not validation:
            if p1_random > 0:
                print(f"Using random move for Player 1 ({p1_random} random moves left)")
                random_agent = RandomPlayer(1)
                p1_direction = random_agent.get_best_move()
                p1_random -= 1
                dir_to_str = {Direction.UP: 'UP', Direction.DOWN: 'DOWN', Direction.LEFT: 'LEFT', Direction.RIGHT: 'RIGHT'}
                validation = judge.handle_move(dir_to_str[p1_direction], 1, is_random=True)
                p1_boost = False
            else:
                print("Player 1 has no random moves left. Forfeiting.")
                judge.end_game(GameResult.AGENT2_WIN)
                print("Game String:", judge.game_str)
                return

        # Player 2 move
        print("Requesting move from Player 2...")
        validation = None
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
                    p2_boost = validation[1]
                    p2_direction = validation[2]
                    break
            print(f"  Attempt {attempt} failed")

        if not p2_move or not validation:
            if p2_random > 0:
                print(f"Using random move for Player 2 ({p2_random} random moves left)")
                random_agent = RandomPlayer(2)
                p2_direction = random_agent.get_best_move()
                p2_random -= 1
                dir_to_str = {Direction.UP: 'UP', Direction.DOWN: 'DOWN', Direction.LEFT: 'LEFT', Direction.RIGHT: 'RIGHT'}
                validation = judge.handle_move(dir_to_str[p2_direction], 2, is_random=True)
                p2_boost = False
            else:
                print("Player 2 has no random moves left. Forfeiting.")
                judge.end_game(GameResult.AGENT1_WIN)
                print("Game String:", judge.game_str)
                return

        # Execute both moves simultaneously
        result = judge.game.step(p1_direction, p2_direction, p1_boost, p2_boost)

        # Send updated state to both players
        judge.send_state(1)
        judge.send_state(2)

        # Update shared_state for visualizer
        with state_lock:
            shared_state['agent1_trail'] = judge.game.agent1.get_trail_positions()
            shared_state['agent2_trail'] = judge.game.agent2.get_trail_positions()
            shared_state['turn'] = judge.game.turns
            # try updating width/height if changed
            try:
                shared_state['width'] = judge.game.board.width
                shared_state['height'] = judge.game.board.height
            except Exception:
                pass

        # keep the original prints too (optional)
        print(f"Agent 1: Trail Length={judge.game.agent1.length}, Alive={judge.game.agent1.alive}, Boosts={judge.game.agent1.boosts_remaining}")
        print(f"Agent 2: Trail Length={judge.game.agent2.length}, Alive={judge.game.agent2.alive}, Boosts={judge.game.agent2.boosts_remaining}")

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

    # keep server alive until program exit
    print("Game loop finished. Visualizer still running until process exit.")


if __name__ == "__main__":
    main()
    sys.exit(0)
