#!/usr/bin/env python3
"""
Main entry point for the Checkers project.
Run with: python run.py [gui|train|test]
"""

import argparse
import sys
from src.core import checkers, board
from src.ai.bot import Bot

def run_gui():
    """Run the graphical user interface"""
    print("Starting Checkers GUI...")
    from src.gui.main import main
    main()

def run_training():
    """Run training scripts"""
    print("Starting training...")
    # You can implement training logic here
    print("Training functionality not yet implemented")

def run_test():
    """Run tests"""
    print("Running tests...")
    # Simple test game - use bot vs bot with automatic moves
    f = board.Field()
    bot1 = Bot(the_depth=2, the_board=f)
    bot2 = Bot(the_depth=2, the_board=f)
    match = checkers.Checkers(control='command', opp=bot1, board=f)
    
    print("Starting test game (bot vs bot)...")
    while match.board.game_is_on == 1:
        # For bot vs bot, we need to handle both sides
        if match.board.whites_turn == 1:
            move = bot1.get_next_move()
        else:
            move = bot2.get_next_move()
        
        result = match.next_turn(move)
        print(f"Turn result: {result}")
    
    print("Game over!")

def main():
    parser = argparse.ArgumentParser(description='Checkers Game Project')
    parser.add_argument('mode', choices=['gui', 'train', 'test'], 
                       nargs='?', default='gui',
                       help='Run mode: gui (default), train, or test')
    
    args = parser.parse_args()
    
    if args.mode == 'gui':
        run_gui()
    elif args.mode == 'train':
        run_training()
    elif args.mode == 'test':
        run_test()

if __name__ == '__main__':
    main()