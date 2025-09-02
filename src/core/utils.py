import numpy as np
import Board

# def one_plane_encode(board):  # <2>
#     for i in board.field:
#         if isinstance(board.field[i], Board.King):
#             if board.field[i].colour == 'white':
#                 board.matrix[8 - int(i[1])][board.columns_num[i[0]] - 1] = 2
#             elif board.field[i].colour == 'black':
#                 board.matrix[8 - int(i[1])][board.columns_num[i[0]] - 1] = -2
#         elif isinstance(board.field[i], Board.Piece):
#             if board.field[i].colour == 'white':
#                 board.matrix[8 - int(i[1])][board.columns_num[i[0]] - 1] = 1
#             elif board.field[i].colour == 'black':
#                 board.matrix[8 - int(i[1])][board.columns_num[i[0]] - 1] = -1
#         else:
#             board.matrix[8 - int(i[1])][board.columns_num[i[0]] - 1] = 0
#     return board.matrix

# def print_board(board):
#     for i in one_plane_encode(board):
#         print(i)

def move_text_to_point(move):
    a, b = move[0:2], move[2:4]
    columns_num = dict(zip(['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h'], [1, 2, 3, 4, 5, 6, 7, 8]))
    return (8-int(a[1]), columns_num[a[0]]-1), (8-int(b[1]), columns_num[b[0]]-1)