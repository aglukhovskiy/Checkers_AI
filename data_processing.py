import numpy as np
from Checkers import Checkers
import Board
# from src.board import init_board, print_board
# from src.checkers import play_move, Turn
# from src.util import mapa, map_state


mapa = {1: 'g1',  # Dictionary for data processing
        2: 'e1',
        3: 'c1',
        4: 'a1',
        5: 'h2',
        6: 'f2',
        7: 'd2',
        8: 'b2',
        9: 'g3',
        10: 'e3',
        11: 'c3',
        12: 'a3',
        13: 'h4',
        14: 'f4',
        15: 'd4',
        16: 'b4',
        17: 'g5',
        18: 'e5',
        19: 'c5',
        20: 'a5',
        21: 'h6',
        22: 'f6',
        23: 'd6',
        24: 'b6',
        25: 'g7',
        26: 'e7',
        27: 'c7',
        28: 'a7',
        29: 'h8',
        30: 'f8',
        31: 'd8',
        32: 'b8'}

f = Board.Field()

match = Checkers(control='command', opp='opp', board=f)

# for i in f.field:
#     if f.field[i] is not None:
#         print(i, i[1], f.field[i])

for i in f.matrix:
    print(i)
# print(f.matrix)


states, values = [], []
value = {'1/2-1/2': 0, '0-1': 1, '1-0': -1}
n_games = {'1/2-1/2': 0, '0-1': 0, '1-0': 0}
parsed_game = 0

with open('OCA_2.0.txt') as file:
    for part in file.read().split('\n\n'):
        tokens = part.split()
        print(part)
        print(tokens)
        val = tokens[-1]
        print(val)
        n_games[val] = n_games[val] + 1
        print(f'Parsing game {parsed_game}')

        skiping = True
        for token in tokens:
            if skiping is True:
                if token == '1.':
                    skiping = False
                continue
            else:
                if '.' in token or token == val:
                    continue
                if 'x' in token:  # multiple jump
                    new_token = token.split('x')
                    n = len(new_token)
                    # for i in range(n - 1):
                    for i in range(1):
                        try:
                            # move = [mapa[int(new_token[0])], mapa[int(new_token[0 + 1])]]
                            move = mapa[int(new_token[0])]+mapa[int(new_token[0 + 1])]
                            print(move)
                            match.next_turn(move=move)
                        except:
                            break
                else:
                    t = token.split('-')
                    # move = [mapa[int(t[0])], mapa[int(t[1])]]
                    move = mapa[int(t[0])]+mapa[int(t[1])]
                    # print(token)
                    # print(t)
                    print(move)
                    match.next_turn(move=move)

        for i in f.field:
            if f.field[i] is not None:
                print(i, f.field[i])
        break




# def get_dataset():
#     """
#     Parse checkers OCA dataset and save as numpy array
#
#     1/2-1/2: 0  -> draw
#     0-1: 1      -> black wins
#     1-0: -1     -> white wins
#     """
#
#     states, values = [], []
#     value = {'1/2-1/2': 0, '0-1': 1, '1-0': -1}
#     n_games = {'1/2-1/2': 0, '0-1': 0, '1-0': 0}
#     parsed_game = 0
#     with open('data/OCA_2.0.txt') as file:
#         for part in file.read().split('\n\n'):
#             tokens = part.split()
#             val = tokens[-1]
#             n_games[val] = n_games[val] + 1
#             print(f'Parsing game {parsed_game}')
#
#             board = init_board()
#             skiping = True
#             turn = Turn.WHITE
#             for token in tokens:
#                 if skiping is True:
#                     if token == '1.':
#                         skiping = False
#                     continue
#                 else:
#                     if '.' in token or token == val:
#                         continue
#                     if 'x' in token:  # multiple jump
#                         new_token = token.split('x')
#                         n = len(new_token)
#                         for i in range(n - 1):
#                             try:
#                                 move = [mapa[int(new_token[i])], mapa[int(new_token[i + 1])]]
#                             except:
#                                 break
#                             play_move(board, move, turn)
#                             # print_board(board)
#                             # print()
#                     else:
#                         t = token.split('-')
#                         try:
#                             move = [mapa[int(t[0])], mapa[int(t[1])]]
#                         except:
#                             break
#                         play_move(board, move, turn)
#                         # print_board(board)
#                         # print()
#
#                     states.append(map_state(board))
#                     values.append(value[val])
#                     turn = Turn.BLACK if turn == Turn.WHITE else Turn.WHITE
#
#             parsed_game += 1
#
#     print(f'1/2-1/2 -> {n_games["1/2-1/2"]}')
#     print(f'0-1     -> {n_games["0-1"]}')
#     print(f'1-0     -> {n_games["1-0"]}')
#
#     x, y = np.array(states), np.array(values)
#     np.savez("data/processed.npz", x, y)




    # board = ['w', '-', 'w', '-', 'w', '-', 'w', '-', 7
    #          '-', 'w', '-', 'w', '-', 'w', '-', 'w', 15
    #          'w', '-', 'w', '-', 'w', '-', 'w', '-', 23
    #          '-', '-', '-', '-', '-', '-', '-', '-', 31
    #          '-', '-', '-', '-', '-', '-', '-', '-', 39
    #          '-', 'b', '-', 'b', '-', 'b', '-', 'b', 47
    #          'b', '-', 'b', '-', 'b', '-', 'b', '-', 55
    #          '-', 'b', '-', 'b', '-', 'b', '-', 'b'] 63