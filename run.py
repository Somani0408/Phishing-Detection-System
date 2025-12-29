"""
Main entry point for the Flask application
Run this file to start the web server
"""
from app import create_app

# Create Flask app instance
app = create_app()

if __name__ == '__main__':
    # Run the Flask development server
    # In production, use a proper WSGI server like Gunicorn
    app.run(debug=True, host='0.0.0.0', port=5000)

