#!/usr/bin/env python3
"""
Flask backend for Voice Object Controller UI.
Run this AFTER sourcing your ROS2 workspace:
    source install/setup.bash
    python3 server.py
"""

from flask import Flask, jsonify
from flask_cors import CORS
import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool
import threading

app = Flask(__name__)
CORS(app)  # Allow requests from file:// (the popup HTML)

rclpy.init()


class TriggerNode(Node):
    def __init__(self):
        super().__init__('web_trigger_node')
        self.pub = self.create_publisher(Bool, '/start_listening', 10)
        self._listening = False

    def set_listening(self, state: bool):
        self._listening = state
        msg = Bool()
        msg.data = state
        self.pub.publish(msg)

    @property
    def is_listening(self):
        return self._listening


node = TriggerNode()


def ros_spin():
    rclpy.spin(node)


threading.Thread(target=ros_spin, daemon=True).start()


@app.route('/start', methods=['POST'])
def start_listening():
    node.set_listening(True)
    return jsonify({"status": "listening", "message": "Listening started"})


@app.route('/stop', methods=['POST'])
def stop_listening():
    node.set_listening(False)
    return jsonify({"status": "stopped", "message": "Listening stopped"})


@app.route('/status', methods=['GET'])
def get_status():
    return jsonify({"listening": node.is_listening})


if __name__ == '__main__':
    print("🚀 Flask server running at http://localhost:5000")
    print("   POST /start  → begin listening")
    print("   POST /stop   → stop listening")
    print("   GET  /status → current state")
    app.run(port=5000, debug=False)