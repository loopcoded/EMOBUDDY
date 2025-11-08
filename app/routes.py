from flask import Blueprint, jsonify, current_app
from . import emotion_service

main = Blueprint('main', __name__)

@main.route('/', methods=['GET'])
def index():
    """Simple API status check."""
    return jsonify({
        "status": "EMOBUDDY Backend is Running",
        "service_status": "Ready to Monitor" if emotion_service.face_model and emotion_service.voice_model else "Models Missing or Failed to Load",
        "model_paths": [current_app.config['FACE_MODEL_PATH'], current_app.config['VOICE_MODEL_PATH']]
    })

@main.route('/status', methods=['GET'])
def status():
    """Get the current emotion and intervention status."""
    stats = emotion_service.get_emotion_statistics()
    return jsonify({
        "status": "success",
        "data": stats
    })

@main.route('/monitor/start', methods=['POST'])
def start_monitoring():
    """Endpoint to explicitly start monitoring state."""
    emotion_service.start_monitoring()
    return jsonify({"status": "monitoring started"})

@main.route('/monitor/stop', methods=['POST'])
def stop_monitoring():
    """Endpoint to explicitly stop monitoring state."""
    emotion_service.stop_monitoring()
    return jsonify({"status": "monitoring stopped"})