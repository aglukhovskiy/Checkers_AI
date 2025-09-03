#!/usr/bin/env python3
"""
Script for testing local API startup before deployment
"""

import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

print("=== Testing local API startup ===")

try:
    # Check basic imports
    import numpy as np
    print("[OK] numpy imported successfully")
    
    import flask
    print("[OK] flask imported successfully")
    
    import flask_cors
    print("[OK] flask_cors imported successfully")
    
    import requests
    print("[OK] requests imported successfully")
    
    # Test game creation
    from src.core.board_v2 import CheckersGame
    game = CheckersGame()
    print("[OK] Game created successfully")
    
    # Test getting possible moves
    moves = game.get_possible_moves()
    print(f"[OK] Got {len(moves)} possible moves")
    
    print("\n[SUCCESS] All tests passed! Application is ready for deployment.")
    
except ImportError as e:
    print(f"[ERROR] Import error: {e}")
    sys.exit(1)
except Exception as e:
    print(f"[ERROR] Testing error: {e}")
    sys.exit(1)