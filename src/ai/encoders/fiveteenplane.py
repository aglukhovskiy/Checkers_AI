import numpy as np
from src.core.board_v2 import CheckersGame

class FiveteenPlaneEncoder:
    """Кодирует доску шашек в 12-плоскостное представление"""

    def __init__(self):
        self.num_planes = 15
        self.rows = 8
        self.cols = 8

    def name(self):
        return 'fiveteenplane'

    def encode(self, game):
        """Кодирует текущее состояние доски в 10-плоскостное представление

        Плоскости:
        0: Обычные белые шашки
        1: Белые дамки
        2: Обычные черные шашки
        3: Черные дамки
        4: Белые шашки под боем
        5: Черные шашки под боем
        6: Белые шашки, которые могут бить
        7: Черные шашки, которые могут бить
        8: Ход белых (1 если ход белых)
        9: Ход черных (1 если ход черных)
        """

        game_matrix = np.zeros((self.num_planes, 8, 8))

        # Отмечаем чей ход
        if game.current_player == 1:  # Ход белых
            game_matrix[8] = 1
        else:  # Ход черных
            game_matrix[9] = 1

        # Определяем шашки под боем и шашки, которые могут бить
        threatened_pieces = {}
        taker_pieces = {}

        # Получаем все возможные взятия
        all_capture_moves = game.get_capture_moves()

        # Извлекаем информацию о шашках под боем и шашках, которые могут бить
        for capture_sequence in all_capture_moves:
            move = capture_sequence[0]
            # Шашка под боем (координаты взятой шашки)
            try:
                threatened_pieces[(move[2], move[3])] = max(threatened_pieces[(move[2], move[3])],len(capture_sequence))
            except:
                threatened_pieces[(move[2], move[3])] = len(capture_sequence)
            # Шашка, которая может бить
            try:
                taker_pieces[(move[0], move[1])] = max(taker_pieces[(move[0], move[1])],len(capture_sequence))
            except:
                taker_pieces[(move[0], move[1])] = len(capture_sequence)

        # Заполняем плоскости для шашек
        for (row, col) in game.pieces:
            piece = game.board[row][col]
        # for row in range(8):
        #     for col in range(8):
        #         piece = game[row][col]

                # if piece is None or piece == 0:
                #     continue

            # Определяем тип шашки
            if piece > 0:  # Белые шашки
                if abs(piece) > 1:  # Дамка
                    game_matrix[1, row, col] = 1
                else:  # Обычная шашка
                    game_matrix[0, row, col] = 1
                if col in (0,7):
                    game_matrix[10, row, col] = 1
                elif row<7:
                    game_matrix[13, row, col] = ((row+1,col+1) in game.pieces) + ((row+1,col-1) in game.pieces)
            else:  # Черные шашки
                if abs(piece) > 1:  # Дамка
                    game_matrix[3, row, col] = 1
                elif piece==-1:  # Обычная шашка
                    game_matrix[2, row, col] = 1
                if col in (0,7):  # Дамка
                    game_matrix[11, row, col] = 1
                elif row>0:  # Обычная шашка
                    game_matrix[14, row, col] = ((row-1,col+1) in game.pieces) + ((row-1,col-1) in game.pieces)

            # Отмечаем шашки под боем
            if (row, col) in threatened_pieces:
                if game.board[row][col] > 0:  # Белая шашка под боем
                    game_matrix[4, row, col] = threatened_pieces[(row, col)]
                else:  # Черная шашка под боем
                    game_matrix[5, row, col] = threatened_pieces[(row, col)]

            # Отмечаем шашки, которые могут бить
            if (row, col) in taker_pieces:
                if game.board[row][col] > 0:  # Белая шашка может бить
                    game_matrix[6, row, col] = taker_pieces[(row, col)]
                else:  # Черная шашка может бить
                    game_matrix[7, row, col] = taker_pieces[(row, col)]

                for row in range(8):
                    for col in range(8):
                        if (row+col)%2==1:
                            game_matrix[12][row][col]=1
                        else:
                            game_matrix[12][row][col]=0
        return game_matrix

    def shape(self):
        return self.num_planes, 8, 8

    def symbols_change(self, symbol):
        if symbol == 0.0:
            return ' . '
        elif symbol == -0.0:
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

    def show_board(self, game):
        for row in game.board:
            print(" ".join(self.symbols_change(num) for num in row))

    def to_one_plane_matrix(self, x_plane):
        one_plane = np.zeros((8,8))
        for row in range(8):
            for col in range(8):
                if x_plane[0][row][col]==1:
                    one_plane[row][col]=1
                if x_plane[1][row][col]==1:
                    one_plane[row][col]=3
                if x_plane[2][row][col]==1:
                    one_plane[row][col]=-1
                if x_plane[3][row][col]==1:
                    one_plane[row][col]=-3
        return one_plane

    # def one_to_two_plane_matrix(self, one_plane):
    #     two_plane = np.zeros((2,8,8))
    #     for row in range(8):
    #         for col in range(8):
    #             if one_plane[row][col]==1:
    #                 two_plane[0][row][col]=1
    #             if one_plane[row][col]==3:
    #                 two_plane[0][row][col]=3
    #             if one_plane[row][col]==-1:
    #                 two_plane[1][row][col]=1
    #             if one_plane[row][col]==-3:
    #                 two_plane[1][row][col]=3
    #     return two_plane
    # def one_to_four_plane_matrix(self, one_plane):
    #     four_plane = np.zeros((4,8,8))
    #     for row in range(8):
    #         for col in range(8):
    #             if one_plane[row][col]==1:
    #                 four_plane[0][row][col]=1
    #             if one_plane[row][col]==3:
    #                 four_plane[1][row][col]=1
    #             if one_plane[row][col]==-1:
    #                 four_plane[2][row][col]=1
    #             if one_plane[row][col]==-3:
    #                 four_plane[3][row][col]=1
    #     return four_plane
    #
    # def ten_to_eight_plane_matrix(self, ten_plane):
    #     eight_plane = np.zeros((8,8,8))
    #     for plane in range(4):
    #         for row in range(8):
    #             for col in range(8):
    #                 eight_plane[plane][row][col] = ten_plane[plane][row][col]
    #
    #     for plane in range(4,8):
    #         for row in range(8):
    #             for col in range(8):
    #                 eight_plane[plane][row][col] = ten_plane[plane+2][row][col]
    #     return eight_plane
    #
    # def ten_to_six_plane_matrix(self, ten_plane):
    #     six_plane = np.zeros((6,8,8))
    #     for plane in range(6):
    #         for row in range(8):
    #             for col in range(8):
    #                 six_plane[plane][row][col] = ten_plane[plane][row][col]
    #     return six_plane
    #
    # def ten_to_five_plane_matrix(self, ten_plane):
    #     five_plane = np.zeros((5,8,8))
    #     for plane in range(5):
    #         for row in range(8):
    #             for col in range(8):
    #                 five_plane[plane][row][col] = ten_plane[plane][row][col]
    #     return five_plane
    #
    def show_board_from_matrix(self, ten_plane):
        one_plane = self.to_one_plane_matrix(ten_plane)
        for row in one_plane:
            print(" ".join(self.symbols_change(num) for num in row))

    def score(self, matrix):
        score = 0
        for row in range(8):
            for col in range(8):
                if matrix[0][row][col]==1:
                    score+=1
                if matrix[1][row][col]==1:
                    score+=3
                if matrix[2][row][col]==1:
                    score-=1
                if matrix[3][row][col]==1:
                    score-=3
        return score

    def ten_to_thirteen_plane_matrix(self, ten_plane):
        thirteen_plane = np.zeros((13,8,8))
        for plane in range(10):
            for row in range(8):
                for col in range(8):
                    thirteen_plane[plane][row][col] = ten_plane[plane][row][col]
                    if plane==0:
                        if (row+col)%2==1:
                            thirteen_plane[12][row][col]==1
                        else:
                            thirteen_plane[12][row][col] == 0

        for row in range(8):
            for col in [0,7]:
                if ten_plane[0][row][col]==1:
                    thirteen_plane[10][row][col]=1
                if ten_plane[1][row][col]==1:
                    thirteen_plane[10][row][col]=1
                if ten_plane[2][row][col]==1:
                    thirteen_plane[11][row][col]=1
                if ten_plane[3][row][col]==1:
                    thirteen_plane[11][row][col]=1
        return thirteen_plane

    def from_wrong_thirteen(self,thirteen_plane):
        game=CheckersGame()
        one_plane = self.to_one_plane_matrix(thirteen_plane)
        game.board = one_plane
        game.pieces = set()
        for row in range(8):
            for col in range(8):
                if one_plane[row][col]!=0:
                    game.pieces.add((row,col))
        game.current_player = -game.current_player
        fiveteenplane = self.encode(game)
        return fiveteenplane



def create():
    return FiveteenPlaneEncoder()