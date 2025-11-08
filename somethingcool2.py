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

        # Identify myself and opponent (based on player_number)
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

        def wrap(pos: tuple[int, int]) -> tuple[int, int]:
            x, y = pos
            if width == 0 or height == 0:
                return (0, 0)
            return (x % width, y % height)

        def neighbors(pos: tuple[int, int]):
            x, y = pos
            for dx, dy in ((0, -1), (0, 1), (1, 0), (-1, 0)):
                yield wrap((x + dx, y + dy))

        # Base blocked cells from board
        BASE_BLOCKED = set()
        for y in range(height):
            row = board_grid[y]
            for x, cell in enumerate(row):
                if cell != EMPTY:
                    BASE_BLOCKED.add((x, y))

        my_trail_set = set(my_trail)
        opp_trail_set = set(opp_trail)

        def infer_dir_from_trail(trail):
            """Infer current direction from last two trail points (handles wrap)."""
            if len(trail) < 2 or width == 0 or height == 0:
                return None
            (x1, y1), (x2, y2) = trail[-2], trail[-1]

            dx = x2 - x1
            dy = y2 - y1

            # Adjust for torus wrap (board is 20x18)
            if dx == width - 1:
                dx = -1
            elif dx == -(width - 1):
                dx = 1
            if dy == height - 1:
                dy = -1
            elif dy == -(height - 1):
                dy = 1

            mapping = {
                (0, -1): Direction.UP,
                (0, 1): Direction.DOWN,
                (1, 0): Direction.RIGHT,
                (-1, 0): Direction.LEFT,
            }
            return mapping.get((dx, dy))

        my_dir_est = infer_dir_from_trail(my_trail)
        opp_dir_est = infer_dir_from_trail(opp_trail)

        def is_reverse(candidate_dir, current_dir):
            if current_dir is None:
                return False
            cdx, cdy = current_dir.value
            ndx, ndy = candidate_dir.value
            return (ndx, ndy) == (-cdx, -cdy)

        def flood_fill(start, blocked: set) -> int:
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

        def local_degree(pos, blocked: set) -> int:
            """How many open neighbors does this cell have? (higher is safer)."""
            deg = 0
            for n in neighbors(pos):
                if n not in blocked:
                    deg += 1
            return deg

        def torus_distance(a: tuple[int, int], b: tuple[int, int]) -> int:
            """Manhattan distance on torus."""
            ax, ay = a
            bx, by = b
            if width == 0 or height == 0:
                return 0
            dx = min((ax - bx) % width, (bx - ax) % width)
            dy = min((ay - by) % height, (by - ay) % height)
            return dx + dy

        dirs = [Direction.UP, Direction.DOWN, Direction.LEFT, Direction.RIGHT]

        def simulate_my_move(direction, use_boost: bool):
            """
            Simulate my move (1 or 2 steps). Return (head, blocked) or (None, None) if I die.
            """
            if is_reverse(direction, my_dir_est):
                return None, None

            blocked = set(BASE_BLOCKED)
            steps = 2 if (use_boost and boosts_remaining > 0) else 1
            head = my_head

            for _ in range(steps):
                nx, ny = wrap((head[0] + direction.value[0], head[1] + direction.value[1]))
                new_pos = (nx, ny)

                if (
                    new_pos in blocked or
                    new_pos in my_trail_set or
                    new_pos in opp_trail_set
                ):
                    return None, None  # I die

                head = new_pos
                blocked.add(head)

            return head, blocked

        def evaluate_move(direction, use_boost: bool) -> float:
            """
            Robust 2-ply heuristic:
              - simulate my move,
              - consider all opponent replies,
              - assume opponent picks the reply with the worst score for me.
            Score balances:
              - my_space - opp_space,
              - trap pressure (their escape moves),
              - my local safety,
              - distance to opponent (hunting),
              - avoiding bad trades / draws.
            """
            my_future_head, blocked_after_my = simulate_my_move(direction, use_boost)
            if my_future_head is None:
                return -1e18  # immediate death

            # Quick early-out: if opponent somehow already dead
            if not opp_trail:
                return 1e9

            worst_score = None

            # Try all opponent replies; assume they pick the one that minimizes our score
            for odir in dirs:
                if opp_dir_est and is_reverse(odir, opp_dir_est):
                    continue

                ox, oy = opp_head
                onx, ony = wrap((ox + odir.value[0], oy + odir.value[1]))
                opp_new = (onx, ony)

                # Head-on: both collide in same cell
                head_on = (opp_new == my_future_head)

                # Opponent hits something
                opp_hits_trail = (
                    opp_new in blocked_after_my or
                    opp_new in my_trail_set or
                    opp_new in opp_trail_set
                )

                if head_on and not opp_hits_trail:
                    # Avoid mutual suicides: big negative compared to clean win
                    branch_score = -5e7
                elif opp_hits_trail:
                    # They die, we live -> huge win
                    branch_score = 8e8
                else:
                    # Both alive in this branch; evaluate board position
                    blocked_world = set(blocked_after_my)
                    blocked_world.add(opp_new)

                    my_space = flood_fill(my_future_head, blocked_world)
                    opp_space = flood_fill(opp_new, blocked_world)
                    my_deg = local_degree(my_future_head, blocked_world)
                    opp_deg = local_degree(opp_new, blocked_world)
                    dist = torus_distance(my_future_head, opp_new)

                    # How many safe moves do they have FROM opp_new now?
                    opp_safe_moves = 0
                    for od2 in dirs:
                        o2x, o2y = opp_new
                        sdx, sdy = od2.value
                        tx, ty = wrap((o2x + sdx, o2y + sdy))
                        cand2 = (tx, ty)
                        if (
                            cand2 in blocked_world or
                            cand2 in my_trail_set or
                            cand2 in opp_trail_set or
                            cand2 == my_future_head
                        ):
                            continue
                        opp_safe_moves += 1

                    # Trap pressure: fewer safe moves is better
                    if opp_safe_moves == 0:
                        trap_bonus = 90.0
                    elif opp_safe_moves == 1:
                        trap_bonus = 40.0
                    elif opp_safe_moves == 2:
                        trap_bonus = 10.0
                    else:
                        trap_bonus = 0.0

                    space_diff = my_space - opp_space

                    # Core heuristic: area control + safety + pressure + proximity
                    branch_score = (
                        7.0 * space_diff +      # dominate area
                        4.0 * my_space -
                        3.0 * opp_space +
                        3.0 * my_deg -
                        2.0 * opp_deg +
                        trap_bonus -
                        0.6 * dist              # slightly prefer being closer (to hunt)
                    )

                    # Late game, territory and traps matter more
                    if turn_count > 120:
                        branch_score += 4.0 * space_diff + 0.5 * trap_bonus

                if worst_score is None or branch_score < worst_score:
                    worst_score = branch_score

            if worst_score is None:
                worst_score = 5e8  # silly fallback, treat as great

            # Boost adjustment: only keep boosts that are good in worst case
            if use_boost:
                # Base risk penalty
                worst_score -= 15.0

                # Re-estimate my_space vs opp_space in this world for a proxy
                my_space_proxy = flood_fill(my_future_head, blocked_after_my)
                opp_space_proxy = flood_fill(opp_head, blocked_after_my | {my_future_head})
                space_diff_proxy = my_space_proxy - opp_space_proxy

                # Favor boosts when even worst-case keeps strong advantage
                if space_diff_proxy > 25 and my_space_proxy > 30:
                    worst_score += 40.0
                elif space_diff_proxy > 10 and my_space_proxy > 20:
                    worst_score += 15.0

            return worst_score

        best_score = -1e19
        best_dir = Direction.RIGHT
        best_use_boost = False

        # Evaluate all directions with and without boost
        for d in dirs:
            if my_dir_est and is_reverse(d, my_dir_est):
                continue

            s_no = evaluate_move(d, use_boost=False)
            if s_no > best_score:
                best_score = s_no
                best_dir = d
                best_use_boost = False

            if boosts_remaining > 0 and turn_count > 8:
                s_boost = evaluate_move(d, use_boost=True)
                if s_boost > best_score:
                    best_score = s_boost
                    best_dir = d
                    best_use_boost = True

        # Fallback if everything is awful (trapped)
        if best_score < -1e10:
            legal_dirs = []
            for d in dirs:
                if my_dir_est and is_reverse(d, my_dir_est):
                    continue
                nx, ny = wrap((my_head[0] + d.value[0], my_head[1] + d.value[1]))
                if (nx, ny) not in BASE_BLOCKED:
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
