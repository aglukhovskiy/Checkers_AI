"""
WSGI entry point for Railway deployment
"""
import os
from api_v2 import app

if __name__ == "__main__":
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port, debug=False)