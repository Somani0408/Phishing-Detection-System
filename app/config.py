"""
Configuration settings for the Flask application
"""
import os
from pathlib import Path

class Config:
    """Base configuration class"""
    # Secret key for session management (change in production!)
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'dev-secret-key-change-in-production'
    
    # File upload limits
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max file size
    UPLOAD_FOLDER = 'uploads'
    
    # Model paths - relative to project root
    BASE_DIR = Path(__file__).parent.parent
    MODEL_DIR = BASE_DIR / 'models'
    MODEL_PATH = MODEL_DIR / 'best_model.pkl'
    SCALER_PATH = MODEL_DIR / 'scaler.pkl'
    
    # Ensure model directory exists
    MODEL_DIR.mkdir(exist_ok=True)

