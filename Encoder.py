# from Board import Field
from Board import King, Piece
import numpy as np


class OnePlaneEncoder():

    num_to_column = dict(zip([1, 2, 3, 4, 5, 6, 7, 8], ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']))

    def __init__(self):
        self.num_planes = 1

    def encode(self, board):  # <2>
        board_matrix = np.zeros((1,8,8))
        whites_turn = board.whites_turn

        # matrix_string = "/n".join(" ".join(str(num) for num in row) for row in board_matrix)

        for r in range(8):
            for c in range(8):
                field = self.num_to_column[c+1]+str(r+1)
                piece = board.field[field]
                if piece is None:
                    continue
                if piece.white == whites_turn:
                    if isinstance(piece, King):
                        board_matrix[0, r, c] = 3
                    else:
                        board_matrix[0, r, c] = 1
                else:
                    if isinstance(piece, King):
                        board_matrix[0, r, c] = -3
                    else:
                        board_matrix[0, r, c] = -1
        return board_matrix

    def symbols_change(self, symbol):
        if symbol == 0.0:
            return ' . '
        elif symbol == 1.0:
            return ' x '
        elif symbol == 3.0:
            return ' X '
        elif symbol == -1.0:
            return ' o '
        elif symbol == -3.0:
            return ' O '
        elif symbol == 7.0:
            # return '  |  '
            return '  |@ '

    def show_board(self, board):
        for row in self.encode(board)[0][::-1]:
            print(" ".join(self.symbols_change(num) for num in row))

    def show_several_boards(self, boards):
        matrix_list = []
        for board in boards:
            # matrix_list.append(self.encode(board)[0])
            matrix_list.append(np.hstack(([[7]]*8, self.encode(board)[0], [[7]]*8)))
            # matrix_list.append(np.hstack([self.encode(board)[0],[[7]]*8]))
            # matrix_list.append(np.concatenate((self.encode(board)[0], [[7]]*8), axis=1))

        matrix_list_concat = np.concatenate(matrix_list, axis=1)

        for row in matrix_list_concat[::-1]:
            print(" ".join(self.symbols_change(num) for num in row))
