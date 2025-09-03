"""
WSGI entry point for Railway deployment
"""
import os
from api_v2 import app

port = int(os.environ.get('PORT', 8080))
print(f"Starting server on port {port}")

# Стандартный запуск Flask с поддержкой многопоточности
if __name__ == "__main__":
    app.run(host='0.0.0.0', port=port, threaded=True)