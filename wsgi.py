"""
WSGI entry point for Railway deployment
"""
import os
import signal
from api_v2 import app

def handle_exit(signum, frame):
    print("Received exit signal, shutting down gracefully...")
    exit(0)

if __name__ == "__main__":
    # Устанавливаем обработчики сигналов
    signal.signal(signal.SIGTERM, handle_exit)
    signal.signal(signal.SIGINT, handle_exit)
    
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port, debug=False)