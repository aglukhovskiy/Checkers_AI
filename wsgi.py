"""
WSGI entry point for Railway deployment
"""
import os
import signal
import time
from api_v2 import app

class Server:
    def __init__(self):
        self.should_exit = False
        signal.signal(signal.SIGTERM, self.handle_exit)
        signal.signal(signal.SIGINT, self.handle_exit)

    def handle_exit(self, signum, frame):
        print(f"Received signal {signum}, shutting down gracefully...")
        self.should_exit = True

    def run(self):
        port = int(os.environ.get('PORT', 8080))
        from waitress import serve
        serve(app, host='0.0.0.0', port=port)

if __name__ == "__main__":
    server = Server()
    server.run()