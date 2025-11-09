import logging
from flask import Flask, render_template, jsonify
import threading

app = Flask(__name__)

log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)
# Shared state updated by judge
shared_state = {
    "width": 20,
    "height": 20,
    "agent1_trail": [],
    "agent2_trail": [],
    "board": None,   # Voronoi / distances
    "turn": 0
}

state_lock = threading.Lock()

@app.route("/")
def index():
    return render_template("board.html")  # Your HTML file

@app.route("/state")
def get_state():
    return jsonify(shared_state)

def run_visualizer(host="0.0.0.0", port=5000):
    app.run(host=host, port=port, debug=False, use_reloader=False, threaded=True)

def start_visualizer_thread(host="0.0.0.0", port=5000):
    thread = threading.Thread(target=run_visualizer, args=(host, port), daemon=True)
    thread.start()
    return thread
