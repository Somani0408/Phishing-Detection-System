"""
Flask routes for the phishing detection web application
Handles HTTP requests and responses
"""
from flask import Blueprint, render_template, request, jsonify
from app.controllers.detection_controller import DetectionController
from app.utils.validators import validate_url, validate_email_text
from app.utils.sanitizers import sanitize_input

# Create blueprint for routes
main_bp = Blueprint('main', __name__)

# Initialize controller (singleton pattern)
detection_controller = DetectionController()

@main_bp.route('/')
def index():
    """Render the main page"""
    return render_template('index.html')

@main_bp.route('/predict', methods=['POST'])
def predict():
    """
    Handle prediction requests
    Validates and sanitizes input, then returns prediction result
    
    Returns:
        JSON response with prediction results or error message
    """
    try:
        # Get JSON data from request
        data = request.get_json()
        
        if not data:
            return jsonify({
                'error': 'No data provided',
                'status': 'error'
            }), 400
        
        # Extract input type and value
        input_type = data.get('type', '').lower()
        input_value = data.get('value', '').strip()
        
        # Validate input type
        if input_type not in ['url', 'email']:
            return jsonify({
                'error': 'Invalid input type. Must be "url" or "email"',
                'status': 'error'
            }), 400
        
        # Validate input value is present
        if not input_value:
            return jsonify({
                'error': f'No {input_type} provided',
                'status': 'error'
            }), 400
        
        # Sanitize input to prevent injection attacks
        sanitized_value = sanitize_input(input_value)
        
        # Additional validation based on type
        if input_type == 'url':
            if not validate_url(sanitized_value):
                return jsonify({
                    'error': 'Invalid URL format',
                    'status': 'error'
                }), 400
        elif input_type == 'email':
            if not validate_email_text(sanitized_value):
                return jsonify({
                    'error': 'Invalid email text format',
                    'status': 'error'
                }), 400
        
        # Get prediction from controller
        result = detection_controller.predict(sanitized_value, input_type)
        
        return jsonify(result), 200
        
    except ValueError as e:
        return jsonify({
            'error': f'Validation error: {str(e)}',
            'status': 'error'
        }), 400
    except Exception as e:
        return jsonify({
            'error': f'Server error: {str(e)}',
            'status': 'error'
        }), 500

@main_bp.route('/health', methods=['GET'])
def health():
    """Health check endpoint for monitoring"""
    return jsonify({
        'status': 'healthy',
        'message': 'Phishing Detection System is running'
    }), 200

