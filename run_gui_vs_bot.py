#!/usr/bin/env python3
"""
Wrapper script to run the GUI vs Bot game.
This script handles the import issues by adding the src directory to the Python path.
"""

import sys
import os

# Add the src directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.gui.gui_vs_bot import play_against_bot

if __name__ == "__main__":
    # Example usage - you can change the model path as needed
    model_path = 'rl/models_n_exp/test_model_custom_fiveteenplane_cbv.hdf5'
    play_against_bot(model_path)