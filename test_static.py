#!/usr/bin/env python3
"""
Script for testing static files serving
"""

import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

print("=== Testing static files ===")

try:
    from api_v2 import app
    
    # Test static files existence
    static_files = ['index.html', 'style.css', 'game.js']
    static_dir = 'static'
    
    for file in static_files:
        file_path = os.path.join(static_dir, file)
        if os.path.exists(file_path):
            print(f"[OK] {file} exists in static directory")
        else:
            print(f"[ERROR] {file} not found in static directory")
    
    # Test Flask app static configuration
    print(f"[INFO] Flask static folder: {app.static_folder}")
    print(f"[INFO] Flask static URL path: {app.static_url_path}")
    
    print("\n[SUCCESS] Static files check completed!")
    
except Exception as e:
    print(f"[ERROR] Testing error: {e}")
    sys.exit(1)