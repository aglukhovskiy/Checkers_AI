import os
import sys
import traceback
from waitress import serve

def create_app():
    """Create and configure the Flask app"""
    try:
        print("=== Starting WSGI Application ===")
        print(f"Python version: {sys.version}")
        print(f"Working directory: {os.getcwd()}")
        
        # Import Flask app
        print("Importing Flask app...")
        from api_v2 import app
        print("Flask app imported successfully")
        
        return app
        
    except Exception as e:
        print(f"ERROR: Failed to create app: {str(e)}")
        print("Full traceback:")
        traceback.print_exc()
        
        # Create a minimal Flask app as fallback
        from flask import Flask, jsonify
        fallback_app = Flask(__name__)
        
        @fallback_app.route('/')
        def error_page():
            return jsonify({
                'error': 'Application failed to start',
                'message': str(e),
                'status': 'error'
            }), 500
            
        @fallback_app.route('/health')
        def health():
            return jsonify({'status': 'error', 'message': 'App failed to start'}), 500
            
        return fallback_app

# Create the app
app = create_app()

# Always run the server
port = int(os.environ.get('PORT', 8080))
print(f"Starting server on port {port}")
serve(app, host='0.0.0.0', port=port)