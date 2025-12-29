"""
Flask application factory
Creates and configures the Flask app instance
"""
from flask import Flask
from app.config import Config

def create_app():
    """
    Application factory pattern
    Creates Flask app instance with configuration
    
    Returns:
        Flask: Configured Flask application instance
    """
    app = Flask(__name__)
    app.config.from_object(Config)
    
    # Register blueprints
    from app.routes import main_bp
    app.register_blueprint(main_bp)
    
    return app

