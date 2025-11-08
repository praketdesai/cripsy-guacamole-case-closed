import os
import uuid
from flask import Flask, request, jsonify
from threading import Lock
from collections import deque

from case_closed_game import Game, Direction, GameResult

# Flask API server setup
app = Flask(__name__)

GLOBAL_GAME = Game()
LAST_POSTED_STATE = {}

game_lock = Lock()
 
PARTICIPANT = "ParticipantTWO"
AGENT_NAME = "somethingTWO"


@app.route("/", methods=["GET"])
def info():
    """Basic health/info endpoint used by the judge to check connectivity.

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
        import random
        from collections import deque

        # Grab latest state and board
        state_board = state.get("board", GLOBAL_GAME.board.grid)
        board_grid = [row[:] for row in state_board]  # shallow copy
        height = len(board_grid)
        width = len(board_grid[0]) if height > 0 else 0

        EMPTY = 0

        # Identify myself and opponent
        if player_number == 1:
            my_agent = GLOBAL_GAME.agent1
            opp_agent = GLOBAL_GAME.agent2
        else:
            my_agent = GLOBAL_GAME.agent2
            opp_agent = GLOBAL_GAME.agent1

        my_trail = list(my_agent.trail)
        opp_trail = list(opp_agent.trail)
        my_head = my_trail[-1]
        opp_head = opp_trail[-1]

        boosts_remaining = my_agent.boosts_remaining
        turn_count = state.get("turn_count", 0)

        def wrap(pos):
            x, y = pos
            if width == 0 or height == 0:
                return (0, 0)
            return (x % width, y % height)

        def neighbors(pos):
            x, y = pos
            for dx, dy in ((0, -1), (0, 1), (1, 0), (-1, 0)):
                yield wrap((x + dx, y + dy))

        # Blocked cells = any non-empty cell on the board (trails)
        BASE_BLOCKED = set()
        for y in range(height):
            row = board_grid[y]
            for x, cell in enumerate(row):
                if cell != EMPTY:
                    BASE_BLOCKED.add((x, y))

        my_trail_set = set(my_trail)
        opp_trail_set = set(opp_trail)

        dirs = [Direction.UP, Direction.DOWN, Direction.LEFT, Direction.RIGHT]

        def flood_fill(start, blocked):
            """Return size of region reachable from start without crossing blocked."""
            start = wrap(start)
            if start in blocked:
                return 0
            visited = {start}
            q = deque([start])
            count = 0
            while q:
                x, y = q.popleft()
                count += 1
                for nx, ny in neighbors((x, y)):
                    if (nx, ny) in visited or (nx, ny) in blocked:
                        continue
                    visited.add((nx, ny))
                    q.append((nx, ny))
            return count

        def local_degree(pos, blocked):
            """Number of open neighbors (mobility)."""
            return sum(1 for n in neighbors(pos) if n not in blocked)

        def torus_distance(a, b):
            """Manhattan distance on torus."""
            ax, ay = a
            bx, by = b
            if width == 0 or height == 0:
                return 0
            dx = min((ax - bx) % width, (bx - ax) % width)
            dy = min((ay - by) % height, (by - ay) % height)
            return dx + dy

        def simulate_my_move(direction, use_boost):
            """
            Apply my move (1 or 2 steps). Return (head, blocked) or (None, None) if I die.
            """
            blocked = set(BASE_BLOCKED)
            steps = 2 if (use_boost and boosts_remaining > 0) else 1
            head = my_head

            for _ in range(steps):
                nx, ny = wrap((head[0] + direction.value[0], head[1] + direction.value[1]))
                new_pos = (nx, ny)

                # Collision with any trail
                if (
                    new_pos in blocked or
                    new_pos in my_trail_set or
                    new_pos in opp_trail_set
                ):
                    return None, None

                head = new_pos
                blocked.add(head)

            return head, blocked

        def evaluate_position(my_pos, opp_pos, blocked):
            """
            Score a static position: both heads fixed, blocked known.
            Focus on area control, trap pressure, safety, and mild aggression.
            """
            my_space = flood_fill(my_pos, blocked)
            opp_space = flood_fill(opp_pos, blocked)

            space_diff = my_space - opp_space
            my_deg = local_degree(my_pos, blocked)
            opp_deg = local_degree(opp_pos, blocked)
            dist = torus_distance(my_pos, opp_pos)

            # Trap pressure: how many safe moves does opponent have?
            opp_safe = 0
            for d in dirs:
                ox, oy = opp_pos
                nx, ny = wrap((ox + d.value[0], oy + d.value[1]))
                cand = (nx, ny)
                if (
                    cand in blocked or
                    cand in my_trail_set or
                    cand in opp_trail_set or
                    cand == my_pos
                ):
                    continue
                opp_safe += 1

            if opp_safe == 0:
                trap_bonus = 90.0
            elif opp_safe == 1:
                trap_bonus = 45.0
            elif opp_safe == 2:
                trap_bonus = 15.0
            else:
                trap_bonus = 0.0

            # Center penalty (very mild)
            if width and height:
                cx = (width - 1) / 2.0
                cy = (height - 1) / 2.0
                dx_c = abs(my_pos[0] - cx)
                dy_c = abs(my_pos[1] - cy)
                center_penalty = dx_c + dy_c
            else:
                center_penalty = 0.0

            score = (
                8.0 * space_diff +     # area control
                3.5 * my_space -
                2.5 * opp_space +
                3.0 * my_deg -
                2.0 * opp_deg +
                trap_bonus -
                0.5 * dist -           # mild aggression (closer to them)
                0.3 * center_penalty
            )

            if turn_count > 120:
                score += 4.0 * space_diff + 0.5 * trap_bonus

            return score

        def evaluate_move(direction, use_boost):
            """
            2-ply worst-case evaluation:
              - simulate my move (maybe with boost),
              - simulate all opponent replies,
              - assume opponent chooses the reply that minimizes my score.
            """
            my_future_head, blocked_after_my = simulate_my_move(direction, use_boost)
            if my_future_head is None:
                return -1e18  # I die immediately

            # If opponent somehow has no trail, huge win already
            if not opp_trail:
                return 1e9

            worst_score = None

            for odir in dirs:
                ox, oy = opp_head
                onx, ony = wrap((ox + odir.value[0], oy + odir.value[1]))
                opp_new = (onx, ony)

                # Head-on collision: avoid if possible
                head_on = (opp_new == my_future_head)

                # Opponent hits something
                opp_hits_trail = (
                    opp_new in blocked_after_my or
                    opp_new in my_trail_set or
                    opp_new in opp_trail_set
                )

                if head_on and not opp_hits_trail:
                    branch_score = -5e7  # mutual death: bad
                elif opp_hits_trail:
                    branch_score = 8e8   # they die, we live: amazing
                else:
                    blocked_world = set(blocked_after_my)
                    blocked_world.add(opp_new)
                    branch_score = evaluate_position(my_future_head, opp_new, blocked_world)

                if worst_score is None or branch_score < worst_score:
                    worst_score = branch_score

            if worst_score is None:
                worst_score = 5e8

            return worst_score

        best_score = -1e19
        best_dir = Direction.RIGHT
        best_use_boost = False

        # First, evaluate *all* directions WITHOUT boost
        no_boost_scores = {}
        for d in dirs:
            s_no = evaluate_move(d, use_boost=False)
            no_boost_scores[d] = s_no
            if s_no > best_score:
                best_score = s_no
                best_dir = d
                best_use_boost = False

        # Now, only allow boosts that *improve* our position vs opponent worst-case
        if boosts_remaining > 0 and turn_count > 4:
            for d in dirs:
                s_no = no_boost_scores.get(d, -1e19)
                s_boost = evaluate_move(d, use_boost=True)

                # ONLY consider boost if it makes our worst-case strictly better
                if s_boost > s_no:
                    if s_boost > best_score:
                        best_score = s_boost
                        best_dir = d
                        best_use_boost = True

        # Fallback if everything is terrible (we're trapped): pick any non-suicidal move
        if best_score < -1e10:
            legal_dirs = []
            for d in dirs:
                nx, ny = wrap((my_head[0] + d.value[0], my_head[1] + d.value[1]))
                if (nx, ny) not in BASE_BLOCKED and (nx, ny) not in my_trail_set:
                    legal_dirs.append(d)
            if not legal_dirs:
                legal_dirs = dirs
            best_dir = random.choice(legal_dirs)
            best_use_boost = False

        move = best_dir.name + (":BOOST" if best_use_boost else "")
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
    port = int(os.environ.get("PORT", "5009"))
    app.run(host="0.0.0.0", port=port, debug=True)
