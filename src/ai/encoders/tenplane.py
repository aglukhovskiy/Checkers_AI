from src.core.board import King
import numpy as np

class TenPlaneEncoder():

    num_to_column = dict(zip([1, 2, 3, 4, 5, 6, 7, 8], ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']))

    def __init__(self):
        self.num_planes = 10

    def name(self):  # <1>
        return 'tenplane'

    def encode(self, board):  # <2>
        board_matrix = np.zeros((self.num_planes,8,8))

        # matrix_string = "/n".join(" ".join(str(num) for num in row) for row in board_matrix)
        to_take = set([y for x, y in board.available_moves()[1].items()])
        takers = set([y for x, y in board.available_moves()[2].items()])

        for r in range(8):
            for c in range(8):
                sel_field = self.num_to_column[c+1]+str(r+1)
                piece = board.field[sel_field]
                piece_to_take = None
                taker_piece = None
                if sel_field in to_take:
                    piece_to_take = board.field[sel_field]
                if sel_field in takers:
                    taker_piece = board.field[sel_field]

                if board.whites_turn==1:
                    board_matrix[4] = 1
                elif board.whites_turn==0:
                    board_matrix[5] = 1

                if piece is None:
                    continue

                if piece.white == 1:
                    if isinstance(piece, King):
                        board_matrix[1, r, c] = 1
                    else:
                        board_matrix[0, r, c] = 1
                else:
                    if isinstance(piece, King):
                        board_matrix[3, r, c] = 1
                    else:
                        board_matrix[2, r, c] = 1

                if getattr(piece_to_take, 'white', None)==1:
                    board_matrix[6, r, c] = 1
                elif getattr(piece_to_take, 'white', None)==0:
                    board_matrix[7, r, c] = 1

                if getattr(taker_piece, 'white', None) == 1:
                    board_matrix[8, r, c] = 1
                elif getattr(taker_piece, 'white', None) == 0:
                    board_matrix[9, r, c] = 1

        return board_matrix

    def encode_to_show(self, board, field=None):  # <2>
        board_matrix = np.zeros((self.num_planes,8,8))

        # matrix_string = "/n".join(" ".join(str(num) for num in row) for row in board_matrix)

        for r in range(8):
            for c in range(8):
                sel_field = self.num_to_column[c+1]+str(r+1)
                if field:
                    piece = field[sel_field]
                else:
                    piece = board.field[sel_field]
                if piece is None:
                    continue
                if piece.white == 1:
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

    def symbols_change(self, symbol, whites_turn):
        turn_sign = 1
        if whites_turn==0:
            turn_sign = -1
        if symbol == 0.0:
            return ' . '
        elif symbol == -0.0:
            return ' . '
        elif symbol*turn_sign == 1.0:
            return ' x '
        elif symbol*turn_sign == 3.0:
            return ' X '
        elif symbol*turn_sign == -1.0:
            return ' o '
        elif symbol*turn_sign == -3.0:
            return ' O '
        elif symbol == 7.0:
            # return '  |  '
            return '  |@ '

    def show_board(self, board):
        if board.whites_turn==1:
            for row in self.encode_to_show(board)[0][::-1]:
                print(" ".join(self.symbols_change(num, board.whites_turn) for num in row))
        else:
            for row in self.encode_to_show(board)[0]*-1:
                print(" ".join(self.symbols_change(num, board.whites_turn) for num in row[::-1]))

    def show_board_from_matrix(self, matrix, whites_turn, fpv=False):
        if whites_turn==1:
            for row in matrix[0][::-1]:
                print(" ".join(self.symbols_change(num, whites_turn) for num in row))
        else:
            if fpv:
                for row in matrix[0]*-1:
                    print(" ".join(self.symbols_change(num, whites_turn) for num in row[::-1]))
            else:
                for row in matrix[0][::-1]*-1:
                    print(" ".join(self.symbols_change(num, whites_turn) for num in row))

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

    def shape(self):
        return self.num_planes, 8, 8

def create():
    return TenPlaneEncoder()