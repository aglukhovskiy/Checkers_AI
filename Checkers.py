import random
import Bot
import Board
import copy
import numpy as np

from Encoder import OnePlaneEncoder

class Checkers:

    def __init__(self, board, opp, control='gui'):
        self.control = control
        self.opp = opp
        self.opp_colour = 'black' # 0: black
        # self.multiple_jumping_piece = []
        self.board = board

    def next_turn(self, move = None):
        if self.control == 'command':
            return self.next_turn_by_hand(move)
        if move == 'end':
            self.board.game_is_on = 0
            # print('game over')
            return 'game over'
        elif self.board.game_is_on == 1 :
            if self.board.whites_turn == 1:
                self.board.move(opp = self.opp, move=move)
            elif self.board.whites_turn == 0:
                self.board.move(opp = self.opp, move=move)
        elif self.board.game_is_on == 0:
            # print('game over')
            return 'game over'
        else:
            print('Error')
        return self.board.history

    def next_turn_by_hand(self, move):
        if self.board.game_is_on == 1 :
            if self.board.whites_turn == 1:
                # print(self.board.available_moves(), self.board.whites_turn)
                self.board.move(opp = self.opp, move=move)
            elif self.board.whites_turn == 0:
                # print(self.board.available_moves(), self.board.whites_turn)
                self.board.move(opp = self.opp, move=move)
        if len(self.board.available_moves()[0]) == 0:
            self.board.game_is_on = 0
        if self.board.game_is_on == 0:
            print('game over')

# Text game

if __name__ == '__main__':
    f = Board.Field()
    bot = Bot.Bot(the_depth=2, the_board=f)

    match = Checkers(control='command', opp='bot', board=f)

    match.next_turn(move='e3f4')
    match.next_turn(move='d6e5')
    match.next_turn(move='e3f4')
    match.next_turn()
    match.next_turn()

    # while match.board.game_is_on == 1:
    #     match.next_turn()