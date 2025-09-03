from api_v2 import app
from waitress import serve
import os

port = int(os.environ.get('PORT', 8080))
print(f"Starting server on port {port}")

if __name__ == "__main__":
    serve(app, host='0.0.0.0', port=port)