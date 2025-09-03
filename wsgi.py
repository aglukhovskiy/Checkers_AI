"""
WSGI entry point for Railway deployment
"""
import os
import time
from api_v2 import app
from waitress import serve

port = int(os.environ.get('PORT', 8080))
print(f"Starting server on port {port}")

# Railway требует долго работающий процесс
while True:
    try:
        serve(app, host='0.0.0.0', port=port)
    except Exception as e:
        print(f"Server error: {e}")
        print("Restarting server in 5 seconds...")
        time.sleep(5)