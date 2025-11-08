"""
run.py - Main application entry point
"""
import os
import sys
from app import create_app, socketio

def main():
    """Run the EMOBUDDY application."""
    # Set environment
    env = os.environ.get('FLASK_ENV', 'development')
    
    # Create app
    app = create_app(env)
    
    # Print startup info
    print("\n" + "="*70)
    print("EMOBUDDY - Emotion Recognition System for Autistic Children")
    print("="*70)
    print(f"\nEnvironment: {env}")
    print(f"Server: http://{app.config['HOST']}:{app.config['PORT']}")
    print(f"WebSocket: ws://{app.config['HOST']}:{app.config['PORT']}/ws/emotion")
    print("\nPress CTRL+C to quit")
    print("="*70 + "\n")
    
    # Run app with SocketIO
    socketio.run(
        app,
        host=app.config['HOST'],
        port=app.config['PORT'],
        debug=app.config['DEBUG'],
        allow_unsafe_werkzeug=True  # For development
    )


if __name__ == '__main__':
    main()


