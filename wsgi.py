"""
WSGI entry point for Railway deployment
"""
import os
from api_v2 import app
from waitress import serve

port = int(os.environ.get('PORT', 8080))
print(f"Starting server on port {port}")

# Просто запускаем сервер без перезапусков
serve(app, host='0.0.0.0', port=port)

# Бесконечный цикл для удержания процесса
while True:
    pass