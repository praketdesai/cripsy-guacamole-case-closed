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
 
PARTICIPANT = "ParticipantONE"
AGENT_NAME = "somethingONE"


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
            visited = set([start])
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

        def territory_score(my_start, opp_start, blocked: set) -> int:
            """
            Voronoi-style territory: multi-source BFS from both heads.
            Score = (#cells closer to me) - (#cells closer to opp).
            """
            owner = {}
            dist = {}
            q = deque()

            my_start = wrap(my_start)
            opp_start = wrap(opp_start)

            if my_start not in blocked:
                owner[my_start] = 1
                dist[my_start] = 0
                q.append(my_start)
            if opp_start not in blocked:
                owner[opp_start] = 2
                dist[opp_start] = 0
                q.append(opp_start)

            while q:
                x, y = q.popleft()
                cur_owner = owner[(x, y)]
                cur_d = dist[(x, y)]
                for nx, ny in neighbors((x, y)):
                    if (nx, ny) in blocked:
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
            return my_cells - opp_cells

        dirs = [Direction.UP, Direction.DOWN, Direction.LEFT, Direction.RIGHT]

        def evaluate_move(direction, use_boost: bool) -> float:
            """
            Simulate moving in 'direction' (1 or 2 steps if boosting),
            then score resulting position based on:
              - reachable space via flood-fill
              - territory race vs opponent
              - local freedom (degree)
              - distance from center
              - boost risk
            """
            # Avoid explicit reverse; judge will ignore it anyway, but don’t waste turn.
            if is_reverse(direction, my_dir_est):
                return -1e18

            blocked = set(BASE_BLOCKED)

            # Simulate my move (1 or 2 steps)
            steps = 2 if (use_boost and boosts_remaining > 0) else 1
            head = my_head

            for _ in range(steps):
                nx, ny = wrap((head[0] + direction.value[0], head[1] + direction.value[1]))
                new_pos = (nx, ny)

                # Immediate collision with any trail (mine or theirs)
                if new_pos in blocked or new_pos in my_trail_set or new_pos in opp_trail_set:
                    return -1e18

                head = new_pos
                blocked.add(head)

            my_future_head = head

            # Crude opponent prediction: one step straight if possible
            opp_future_head = opp_head
            opp_alive = True
            if opp_dir_est is not None:
                ox, oy = opp_head
                odx, ody = opp_dir_est.value
                nx, ny = wrap((ox + odx, oy + ody))
                cand = (nx, ny)
                if cand in blocked or cand == my_future_head:
                    opp_alive = False
                else:
                    opp_future_head = cand
                    blocked.add(opp_future_head)

            # Hardcore avoid: predicted head-on collision
            if opp_alive and opp_future_head == my_future_head:
                return -1e18

            # Flood-fill space just for me
            my_space = flood_fill(my_future_head, blocked)

            # Territory differential if opponent survives
            terr = 0
            if opp_alive:
                terr = territory_score(my_future_head, opp_future_head, blocked)

            # Local safety (branching factor around head)
            deg = local_degree(my_future_head, blocked)

            # Distance to board center (prefer slightly towards center)
            if width and height:
                cx = (width - 1) / 2.0
                cy = (height - 1) / 2.0
                dx_center = abs(my_future_head[0] - cx)
                dy_center = abs(my_future_head[1] - cy)
                center_penalty = dx_center + dy_center
            else:
                center_penalty = 0.0

            # Compose final heuristic score
            score = (
                5.0 * my_space +       # survival / reachable area
                2.5 * terr +           # win the territory race
                3.0 * deg -            # avoid tight corridors
                0.8 * center_penalty   # mild push toward center
            )

            # Boost risk: only worth it when space is really good
            if use_boost:
                score -= 10.0  # base risk
                if my_space > 40:
                    score += 8.0  # compensates when we clearly have a big open field

            # Late-game: territory matters more
            if turn_count > 120:
                score += 0.5 * terr

            return score

        best_score = -1e19
        best_dir = Direction.RIGHT
        best_use_boost = False

        # Evaluate all directions with and without boost
        for d in dirs:
            # Avoid reverse if we can estimate our current heading
            if my_dir_est and is_reverse(d, my_dir_est):
                continue

            # No boost
            s_no = evaluate_move(d, use_boost=False)
            if s_no > best_score:
                best_score = s_no
                best_dir = d
                best_use_boost = False

            # With boost: only consider if we actually have boosts and not at the very start
            if boosts_remaining > 0 and turn_count > 15:
                s_boost = evaluate_move(d, use_boost=True)
                if s_boost > best_score:
                    best_score = s_boost
                    best_dir = d
                    best_use_boost = True

        # If literally everything looks terrible (we’re trapped), fall back to any non-wall move
        if best_score < -1e10:
            legal_dirs = []
            for d in dirs:
                if my_dir_est and is_reverse(d, my_dir_est):
                    continue
                nx, ny = wrap((my_head[0] + d.value[0], my_head[1] + d.value[1]))
                if (nx, ny) not in BASE_BLOCKED:
                    legal_dirs.append(d)

            if not legal_dirs:
                legal_dirs = dirs  # totally doomed, just pick something

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
    port = int(os.environ.get("PORT", "5008"))
    app.run(host="0.0.0.0", port=port, debug=True)
