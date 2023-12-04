# v.10
import random
import Bot
import Board
import copy

class Checkers:

    def __init__(self, board, opp, control='gui'):
        self.control = control
        self.opp = opp
        self.opp_colour = 'black' # 0: black
        # self.multiple_jumping_piece = []
        self.board = board

    def next_turn(self, move = None):
        if self.control == 'command':
            return self.next_turn_by_hand()
        if move == 'end':
            self.board.game_is_on = 0
            # print('game over')
            return 'game over'
        elif self.board.game_is_on == 1 :
            if self.board.whites_turn == 1:
                self.board.move(colour='white', opp = self.opp, move=move)
            elif self.board.whites_turn == 0:
                self.board.move(colour='black', opp = self.opp, move=move)
        elif self.board.game_is_on == 0:
            # print('game over')
            return 'game over'
        else:
            print('Error')
        return self.board.history

    def next_turn_by_hand(self):
        if self.board.game_is_on == 1 :
            if self.board.whites_turn == 1:
                print(self.board.available_moves(colour='white'), self.board.whites_turn)
                self.board.move(colour='white', opp = self.opp)
            elif self.board.whites_turn == 0:
                print(self.board.available_moves(colour='black'), self.board.whites_turn)
                self.board.move(colour='black', opp = self.opp)
        elif self.board.game_is_on == 0:
            print('game over')

f = Board.Field()

bot = Bot.Bot(the_depth=1, the_board=f)

match = Checkers(control='command', opp=bot, board=f)

# while match.board.game_is_on == 1:
#     match.next_turn()

# match.next_turn()
# print(match.board.whites_turn, match.opp, match.opp.alpha_beta(match.board, match.opp.depth, float('-inf'), float('inf'), maximizing_whites=0))
# , match.opp.get_next_move())
# match.next_turn()
# match.next_turn()
# print(match.board.whites_turn, match.opp, match.opp.get_next_move())


# match.next_turn()
# match.next_turn()

# print(f.field)
# f2 = Field()
# f2.field['a1'] = None
#
# f.field = f2.field
#
# print(f.field)

