import os
import uuid
from flask import Flask, request, jsonify
from threading import Lock
from collections import deque

from case_closed_game import Game, Direction, GameResult
import copy

# Flask API server setup
app = Flask(__name__)

GLOBAL_GAME = Game()
LAST_POSTED_STATE = {}

game_lock = Lock()
 
PARTICIPANT = "ParticipantX"
AGENT_NAME = "MiniMaxX"
WIDTH = GLOBAL_GAME.board.width
HEIGHT = GLOBAL_GAME.board.height
EMPTY = 0

@app.route("/", methods=["GET"])
def info():
    """Basic health/info endpoint used by the judge to check connectivity.H

    Returns participant and agent_name (so Judge.check_latency can create Agent objects).
    """
    return jsonify({"participant": PARTICIPANT, "agent_name": AGENT_NAME}), 200


def _update_local_game_from_post(data: dict):
    """Update the local GLOBAL_GAME using the JSON posted by the judge.

    The judge posts a dictionary with keys matching the Judge.send_state payload
    (board, agent1_trail, agent2_trail, agent1_length, agent2_length, agent1_alive,
    agent2_alive, agent1_boosts, agent2_boosts, turn_count).
    """
    with game_lock:
        LAST_POSTED_STATE.clear()
        LAST_POSTED_STATE.update(data)

        if "board" in data:
            try:
                GLOBAL_GAME.board.grid = data["board"]
            except Exception:
                pass

        if "agent1_trail" in data:
            GLOBAL_GAME.agent1.trail = deque(tuple(p) for p in data["agent1_trail"]) 
        if "agent2_trail" in data:
            GLOBAL_GAME.agent2.trail = deque(tuple(p) for p in data["agent2_trail"]) 
        if "agent1_length" in data:
            GLOBAL_GAME.agent1.length = int(data["agent1_length"])
        if "agent2_length" in data:
            GLOBAL_GAME.agent2.length = int(data["agent2_length"])
        if "agent1_alive" in data:
            GLOBAL_GAME.agent1.alive = bool(data["agent1_alive"])
        if "agent2_alive" in data:
            GLOBAL_GAME.agent2.alive = bool(data["agent2_alive"])
        if "agent1_boosts" in data:
            GLOBAL_GAME.agent1.boosts_remaining = int(data["agent1_boosts"])
        if "agent2_boosts" in data:
            GLOBAL_GAME.agent2.boosts_remaining = int(data["agent2_boosts"])
        if "turn_count" in data:
            GLOBAL_GAME.turns = int(data["turn_count"])

    

@app.route("/send-state", methods=["POST"])
def receive_state():
    """Judge calls this to push the current game state to the agent server.

    The agent should update its local representation and return 200.
    """
    data = request.get_json()
    if not data:
        return jsonify({"error": "no json body"}), 400
    _update_local_game_from_post(data)
    return jsonify({"status": "state received"}), 200

def getHead(agent) -> tuple[int, int]:
    return agent.trail[-1]

def isBlocked(pos : tuple[int, int], board) -> bool:
    if board.grid[pos[1]][pos[0]] == EMPTY:
        return False
    return True

def getCurrentDirection(agent) -> tuple[int, int]:
    return agent.direction.value

def imminentDeath(agent, direction, boost=False):
    """
    Returns True if the specified move (with or without boost) causes a collision.
    - Checks both the intermediate and final cell for boost moves.
    - Handles toroidal (wraparound) boards of any size.
    - Safe for [row][col] (i.e., [y][x]) grids.
    """
    board = agent.board
    width = board.width
    height = board.height
    EMPTY = 0  # Or pass as an argument if needed.
    x, y = agent.trail[-1]
    dx, dy = direction.value

    # First step
    x1 = (x + dx) % width
    y1 = (y + dy) % height
    if board.grid[y1][x1] != EMPTY:
        return True

def evaluate(game_obj: Game, player_number: int) -> float:
    """Dont: "keep summer safe".

    For the specified player this computes the sum of open distances in the
    four cardinal directions from the player's head to the nearest occupied
    cell (agent trail). Larger sums mean more space (further from "walls").

    We return player's_sum - 0.5 * opponent_sum so the agent prefers more
    open space than the opponent. Strong terminal scores are applied when a
    player is dead.
    """
    board = game_obj.board
    width = board.width
    height = board.height

    me = game_obj.agent1 if player_number == 1 else game_obj.agent2
    opp = game_obj.agent2 if player_number == 1 else game_obj.agent1

    if not me.alive and not opp.alive:
        return 0.0
    if not me.alive:
        return -10000.0
    if not opp.alive:
        return 10000.0
    
    territory_scores = territory_score(me, opp)
    flood_score = flood_fill(me)
    
    return territory_scores * .4 + flood_score * .6
    
def wrap(pos: tuple[int, int]) -> tuple[int, int]:
    x, y = pos
    if WIDTH == 0 or HEIGHT == 0:
        return (0, 0)
    return (x % WIDTH, y % HEIGHT)

def neighbors(pos: tuple[int, int]):
        x, y = pos
        for dx, dy in ((0, -1), (0, 1), (1, 0), (-1, 0)):
            yield wrap((x + dx, y + dy))

def territory_score(my_agent, opp_agent) -> int:
    """
    Voronoi-style territory: multi-source BFS from both heads.
    Score = (#cells closer to me) - (#cells closer to opp).
    """

    my_start = getHead(my_agent)
    opp_start = getHead(opp_agent)

    owner = {}
    dist = {}
    q = deque()

    my_start = wrap(my_start)
    opp_start = wrap(opp_start)

    owner[my_start] = 1
    dist[my_start] = 0
    q.append(my_start)

    owner[opp_start] = 2
    dist[opp_start] = 0
    q.append(opp_start)

    while q:
        x, y = q.popleft()
        cur_owner = owner[(x, y)]
        cur_d = dist[(x, y)]
        for nx, ny in neighbors((x, y)):
            if isBlocked((nx, ny), my_agent.board):
                continue
            nd = cur_d + 1
            if (nx, ny) not in dist:
                dist[(nx, ny)] = nd
                owner[(nx, ny)] = cur_owner
                q.append((nx, ny))
            else:
                # Same distance from me & opponent -> contested cell
                if nd == dist[(nx, ny)] and owner[(nx, ny)] != cur_owner:
                    owner[(nx, ny)] = 0

    my_cells = sum(1 for o in owner.values() if o == 1)
    opp_cells = sum(1 for o in owner.values() if o == 2)

    # print(my_cells)
    # print(opp_cells)

    return my_cells - opp_cells

def flood_fill(my_agent) -> int:
    """Return size of region reachable from start without crossing blocked."""

    board = my_agent.board
    start = getHead(my_agent)
    start = wrap(start)
    visited = set([start])
    q = deque([start])
    count = 0
    while q:
        x, y = q.popleft()
        count += 1
        for nx, ny in neighbors((x, y)):
            if (nx, ny) in visited or isBlocked((nx, ny), board):
                continue
            visited.add((nx, ny))
            q.append((nx, ny))
    return count


@app.route("/send-move", methods=["GET"])
def send_move():
    """Judge calls this (GET) to request the agent's move for the current tick.

    Query params the judge sends (optional): player_number, attempt_number,
    random_moves_left, turn_count. Agents can use this to decide.
    
    Return format: {"move": "DIRECTION"} or {"move": "DIRECTION:BOOST"}
    where DIRECTION is UP, DOWN, LEFT, or RIGHT
    and :BOOST is optional to use a speed boost (move twice)
    """
    player_number = request.args.get("player_number", default=1, type=int)

    with game_lock:
        state = dict(LAST_POSTED_STATE)   
        my_agent = GLOBAL_GAME.agent1 if player_number == 1 else GLOBAL_GAME.agent2
        boosts_remaining = my_agent.boosts_remaining
   
    # -----------------your code here-------------------
    # Simple minimax search over a small depth. This will call an
    # external `evaluate(game, player_number)` function which you can
    # implement later. If `evaluate` is not present yet, we fall back
    # to a neutral score of 0 so the agent still runs.

    # Configuration
    MAX_DEPTH = 2

    board_grid = my_agent.board.grid
    width = my_agent.board.width
    height = my_agent.board.height
    
    def safe_evaluate(game_obj, player_num: int) -> float:
        try:
            # User will implement this function. We call it here.
            return evaluate(game_obj, player_num)
        except NameError:
            # evaluate not implemented yet by user
            print("evaluate failed")
            return 0.0

    def possible_moves_for(agent) -> list[tuple[Direction, bool]]:
        moves = []

        cur_dx, cur_dy = getCurrentDirection(agent)
        print(getCurrentDirection(agent))
        opposite = (-cur_dx, -cur_dy)
        for d in (Direction.UP, Direction.DOWN, Direction.LEFT, Direction.RIGHT):
            if (d.value == opposite or imminentDeath(agent, d)):
                # print("ARE EQUAL: ", d.value, opposite)
                continue
            else:
                # print("ARE NOT EQUAL: ", d.value,)
                moves.append((d, False))
                if agent.boosts_remaining > 0 and imminentDeath(agent, d, True):
                    moves.append((d, True))

        print(moves)
        return moves

    def is_terminal(game_obj) -> bool:
        # Terminal if either agent is dead or max turns reached
        if not game_obj.agent1.alive or not game_obj.agent2.alive:
            return True
        if game_obj.turns >= 200:
            return True
        return False

    def minimax(
        game_obj: Game,
        depth: int,
        alpha: float,
        beta: float,
        maximizing: bool,
        player_num: int
    ) -> float:
        """Alpha-beta pruning minimax.

        Args:
            game_obj: simulated game state
            depth: remaining depth
            alpha: alpha value
            beta: beta value
            maximizing: whether this node is maximizing for `player_num`
            player_num: which player (1 or 2) we are evaluating for
        """
        if depth == 0 or is_terminal(game_obj):
            return safe_evaluate(game_obj, player_num)

        my_agent = game_obj.agent1 if player_num == 1 else game_obj.agent2
        opp_agent = game_obj.agent2 if player_num == 1 else game_obj.agent1

        my_moves = possible_moves_for(my_agent)
        opp_moves = possible_moves_for(opp_agent)

        if maximizing:
            value = float("-inf")
            # loop over our moves then opponent's moves (full-turn simulation)
            for my_d, my_boost in my_moves:
                worst_value = float("inf")
                for opp_d, opp_boost in opp_moves:
                    sim = copy.deepcopy(game_obj)
                    if player_num == 1:
                        sim.step(my_d, opp_d, my_boost, opp_boost)
                    else:
                        sim.step(opp_d, my_d, opp_boost, my_boost)
                    score = minimax(sim, depth - 1, alpha, beta, False, player_num)

                    if score < worst_value:
                        worst_value = score
                
                if worst_value > value:
                    value = worst_value
                if value > alpha:
                    alpha = value
                if alpha >= beta:
                    # beta cutoff
                    return value
            return value if value != float("-inf") else safe_evaluate(game_obj, player_num)
        else:
            value = float("inf")
            for my_d, my_boost in my_moves:
                best_value = float("-inf")
                for opp_d, opp_boost in opp_moves:
                    sim = copy.deepcopy(game_obj)
                    if player_num == 1:
                        sim.step(my_d, opp_d, my_boost, opp_boost)
                    else:
                        sim.step(opp_d, my_d, opp_boost, my_boost)
                    score = minimax(sim, depth - 1, alpha, beta, True, player_num)
                    if score > best_value:
                        best_value = score


                if best_value < value:
                    value = best_value
                if value < beta:
                    beta = value
                if alpha >= beta:
                    # alpha cutoff
                    return value
            return value if value != float("inf") else safe_evaluate(game_obj, player_num)

    # Top-level search: pick best move for our player
    root_game = None
    with game_lock:
        # construct a local copy of the last posted state via GLOBAL_GAME
        root_game = copy.deepcopy(GLOBAL_GAME)
    

    player_num = player_number
    enemy_num = (player_number + 1) % 2

    best_score = float("-inf")
    best_move = (Direction.RIGHT, False)

    for d, use_boost in possible_moves_for(root_game.agent1 if player_num == 1 else root_game.agent2):
        # assume opponent may respond with any move; we use minimax to aggregate
        worst_score = float("inf")
        worst_move = (Direction.RIGHT, False)

        for enemy_d, enemy_use_boost in possible_moves_for(root_game.agent1 if enemy_num == 1 else root_game.agent2):
            sim = copy.deepcopy(root_game)

            # for simulation, we need to pick an opponent move too; minimax will explore those
            if player_num == 1:
                # try a default opponent move to advance one ply and then call minimax
                sim.step(d, enemy_d, use_boost, enemy_use_boost)
            else:
                sim.step(enemy_d, d, enemy_use_boost, use_boost)

            score = minimax(sim, MAX_DEPTH - 1, float("-inf"), float("inf"), False, player_num)
            print("Me: ", d, "opp: ", enemy_d, score)
            if score < worst_score:
                worst_score = score
                worst_move = (d, use_boost)
        
        if worst_score > best_score:
            best_score = worst_score
            best_move = worst_move

    # Format move string for judge
    dir_str = best_move[0].name
    move = dir_str + (":BOOST" if best_move[1] else "")

    print("Sending: ", move)


    # -----------------end code here--------------------

    return jsonify({"move": move}), 200


@app.route("/end", methods=["POST"])
def end_game():
    """Judge notifies agent that the match finished and provides final state.

    We update local state for record-keeping and return OK.
    """
    data = request.get_json()
    if data:
        _update_local_game_from_post(data)
    return jsonify({"status": "acknowledged"}), 200


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5008"))
    app.run(host="0.0.0.0", port=port, debug=True)
