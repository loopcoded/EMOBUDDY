import logging
from flask import Flask
from flask_socketio import SocketIO
from config import get_config, config as default_config

# Initialize SocketIO globally
socketio = SocketIO()

def create_app(config_name=None):
    """
    Application factory function.
    Initializes Flask, loads configuration, and registers components.
    """
    app = Flask(__name__)
    
    # Load configuration
    if config_name:
        ConfigClass = get_config(config_name)
    else:
        ConfigClass = default_config
        
    app.config.from_object(ConfigClass)
    
    # Configure logging
    logging.basicConfig(level=logging.INFO, 
                        format='[%(asctime)s] %(levelname)s in %(name)s: %(message)s')
    
    # Initialize extensions
    socketio.init_app(app, cors_allowed_origins="*", async_mode='threading')
    
    # Import and Register Blueprints
    from .routes import main as main_blueprint
    app.register_blueprint(main_blueprint)
    
    # Import WebSocket handlers to register events
    from . import websocket_handler
    
    app.logger.info("EMOBUDDY Application Initialized.")
    
    return app

# The service must be initialized *after* config is loaded but before it's used.
# We initialize it here to ensure it's a singleton instance available to all handlers.
# NOTE: The models will be loaded into this service instance.
from services.emotion_monitoring_service import EmotionMonitoringService
emotion_service = EmotionMonitoringService(default_config)