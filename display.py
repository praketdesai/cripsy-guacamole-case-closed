# visualizer.py
import logging
from flask import Flask, render_template, jsonify
import multiprocessing
import threading

app = Flask(__name__)

log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

# Shared state structure (will be replaced by manager proxy in subprocess)
shared_state = {
    "width": 50,
    "height": 50,
    "agent1_trail": [],  # list of (x,y)
    "agent2_trail": [],  # list of (x,y)
    "board": [],         # 2D list that will be built each update
    "turn": 0
}

state_lock = threading.Lock()


def rebuild_board():
    """
    Build a grid based on trails.
    0 = empty
    1 = any trail
    2 = agent1 current position
    3 = agent2 current position
    """
    w = int(shared_state.get("width", 50))
    h = int(shared_state.get("height", 50))

    # Create blank board
    board = [[0 for _ in range(w)] for _ in range(h)]

    # Convert manager lists to regular lists for safe iteration
    agent1_trail = list(shared_state.get("agent1_trail", []))
    agent2_trail = list(shared_state.get("agent2_trail", []))

    # Mark agent trails
    for pos in agent1_trail:
        # Handle both tuple and list formats
        if isinstance(pos, (list, tuple)) and len(pos) >= 2:
            x, y = int(pos[0]), int(pos[1])
            if 0 <= x < w and 0 <= y < h:
                board[y][x] = 1

    for pos in agent2_trail:
        if isinstance(pos, (list, tuple)) and len(pos) >= 2:
            x, y = int(pos[0]), int(pos[1])
            if 0 <= x < w and 0 <= y < h:
                board[y][x] = 1

    # Mark agent positions (last trail entries)
    if agent1_trail:
        pos = agent1_trail[-1]
        if isinstance(pos, (list, tuple)) and len(pos) >= 2:
            x, y = int(pos[0]), int(pos[1])
            if 0 <= x < w and 0 <= y < h:
                board[y][x] = 2

    if agent2_trail:
        pos = agent2_trail[-1]
        if isinstance(pos, (list, tuple)) and len(pos) >= 2:
            x, y = int(pos[0]), int(pos[1])
            if 0 <= x < w and 0 <= y < h:
                board[y][x] = 3

    return board


@app.route("/")
def index():
    return render_template("board.html")


@app.route("/state")
def get_state():
    """
    Build updated board and send state to client.
    """
    try:
        with state_lock:
            # Build board from current trail data
            board = rebuild_board()
            
            # Prepare response with converted data types
            response = {
                "width": int(shared_state.get("width", 50)),
                "height": int(shared_state.get("height", 50)),
                "board": board,
                "agent1_trail": list(shared_state.get("agent1_trail", [])),
                "agent2_trail": list(shared_state.get("agent2_trail", [])),
                "turn": int(shared_state.get("turn", 0))
            }
            
            return jsonify(response)
    
    except Exception as e:
        logging.error(f"Error in get_state: {e}")
        # Return empty state on error
        return jsonify({
            "width": 50,
            "height": 50,
            "board": [[0] * 50 for _ in range(50)],
            "agent1_trail": [],
            "agent2_trail": [],
            "turn": 0
        })


def run_display(shared_state_proxy=None, lock_proxy=None, host="0.0.0.0", port=5000):
    """
    Run Flask app. If manager proxies provided, replace globals.
    """
    global shared_state, state_lock
    
    if shared_state_proxy is not None:
        shared_state = shared_state_proxy
        logging.info("Using shared state proxy from manager")
    
    if lock_proxy is not None:
        state_lock = lock_proxy
        logging.info("Using lock proxy from manager")
    
    app.run(host=host, port=port, debug=False, use_reloader=False, threaded=True)


def start_display_process(shared_state_proxy, lock_proxy, host="0.0.0.0", port=5000):
    """
    Start Flask app in a separate process with manager proxies.
    """
    p = multiprocessing.Process(
        target=run_display,
        args=(shared_state_proxy, lock_proxy, host, port),
        daemon=True
    )
    p.start()
    return p