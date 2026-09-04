"""
REST Calculator Server
Also exposes /metrics so the benchmark can measure server-side CPU/memory
cost separately from client-side cost.
"""

from flask import Flask, request, jsonify
from monitor import SystemMonitor

app = Flask(__name__)
monitor = SystemMonitor()
monitor.start()


@app.route('/add', methods=['POST'])
def add():
    data = request.json
    result = data['a'] + data['b']
    return jsonify({'result': result})


@app.route('/subtract', methods=['POST'])
def subtract():
    data = request.json
    result = data['a'] - data['b']
    return jsonify({'result': result})


@app.route('/multiply', methods=['POST'])
def multiply():
    data = request.json
    result = data['a'] * data['b']
    return jsonify({'result': result})


@app.route('/divide', methods=['POST'])
def divide():
    data = request.json
    if data['b'] == 0:
        return jsonify({'error': 'Division by zero'}), 400
    result = data['a'] / data['b']
    return jsonify({'result': result})


@app.route('/metrics', methods=['GET'])
def metrics():
    return jsonify(monitor.stats())


@app.route('/metrics/reset', methods=['POST'])
def metrics_reset():
    monitor.stop()
    monitor.start()
    return jsonify({'status': 'reset'})


if __name__ == '__main__':
    print("Starting Flask server on http://localhost:5000")
    # debug=False: Flask's debug reloader forks a child process, which would
    # make the /metrics numbers measure the wrong process.
    app.run(debug=False, port=5000, threaded=True)
