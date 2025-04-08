
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Conv2D, ZeroPadding2D
import numpy as np
import h5py
from encoders.tenplane_v2 import TenPlaneEncoder
from rl.pg_agent import PolicyAgent, load_policy_agent
from Board_v2 import CheckersGame


def layers(input_shape):
    return [
        ZeroPadding2D(padding=3, input_shape=input_shape),  # <1>
        Conv2D(48, (7, 7)),
        Activation('relu'),

        ZeroPadding2D(padding=2),  # <2>
        Conv2D(32, (5, 5)),
        Activation('relu'),

        ZeroPadding2D(padding=2),
        Conv2D(32, (5, 5)),
        Activation('relu'),

        ZeroPadding2D(padding=2),
        Conv2D(32, (5, 5)),

        # ZeroPadding2D(padding=3, input_shape=input_shape, data_format='channels_first'),  # <1>
        # Conv2D(48, (7, 7), data_format='channels_first'),
        # Activation('relu'),
        #
        # ZeroPadding2D(padding=2, data_format='channels_first'),  # <2>
        # Conv2D(32, (5, 5), data_format='channels_first'),
        # Activation('relu'),
        #
        # ZeroPadding2D(padding=2, data_format='channels_first'),
        # Conv2D(32, (5, 5), data_format='channels_first'),
        # Activation('relu'),
        #
        # ZeroPadding2D(padding=2, data_format='channels_first'),
        # Conv2D(32, (5, 5), data_format='channels_first'),
        # Activation('relu'),

        Flatten(),
        Dense(512),
        Activation('relu'),
    ]

def create_model(input_shape=(5, 8, 8)):  # Изменяем порядок размерностей
    """Создаёт модель нейронной сети для агента"""
    model = Sequential()
    for layer in layers(input_shape):
        model.add(layer)
    model.add(Dense(1, activation='linear'))  # Выход - скор
    return model

model =create_model()
encoder = TenPlaneEncoder()
new_agent = PolicyAgent(model, encoder)
with h5py.File('models_n_exp/test_model.hdf5', 'w') as model_outf:
    new_agent.serialize(model_outf)

with h5py.File('models_n_exp/test_model.hdf5', 'r') as agent_file:
    agent1 = load_policy_agent(agent_file)


equal_board = CheckersGame()
# equal_board.current_player=-1
encoder.show_board(equal_board)
print('-------')
board_w_plus_2 = CheckersGame()
# board_w_plus_2.current_player=-1
board_w_plus_2.board[2][1] = 0
board_w_plus_2.board[2][3] = 0
encoder.show_board(board_w_plus_2)
print('-------')
board_w_plus_6 = CheckersGame()
# board_w_plus_6.current_player=-1
board_w_plus_6.board[2][1] = 0
board_w_plus_6.board[2][3] = 0
board_w_plus_6.board[2][5] = 0
board_w_plus_6.board[1][0] = 0
board_w_plus_6.board[1][4] = 0
board_w_plus_6.board[1][6] = 0
encoder.show_board(board_w_plus_6)
print('-------')
board_w_plus_10 = CheckersGame()
# board_w_plus_10.current_player=-1
board_w_plus_10.board[2][1] = 0
board_w_plus_10.board[2][3] = 0
board_w_plus_10.board[2][5] = 0
board_w_plus_10.board[1][0] = 0
board_w_plus_10.board[1][4] = 0
board_w_plus_10.board[1][6] = 0
board_w_plus_10.board[0][1] = 0
board_w_plus_10.board[0][3] = 0
board_w_plus_10.board[0][5] = 0
board_w_plus_10.board[0][7] = 0
encoder.show_board(board_w_plus_10)

print('-------')
board_w_minus_2 = CheckersGame()
# board_w_minus_2.current_player=-1
board_w_minus_2.board[5][0] = 0
board_w_minus_2.board[5][2] = 0
encoder.show_board(board_w_minus_2)
print('-------')
board_w_minus_6 = CheckersGame()
# board_w_minus_6.current_player=-1
board_w_minus_6.board[5][0] = 0
board_w_minus_6.board[5][2] = 0
board_w_minus_6.board[5][4] = 0
board_w_minus_6.board[6][1] = 0
board_w_minus_6.board[6][3] = 0
board_w_minus_6.board[6][5] = 0
encoder.show_board(board_w_minus_6)
print('-------')
board_w_minus_10 = CheckersGame()
# board_w_minus_10.current_player=-1
board_w_minus_10.board[5][0] = 0
board_w_minus_10.board[5][2] = 0
board_w_minus_10.board[5][4] = 0
board_w_minus_10.board[6][1] = 0
board_w_minus_10.board[6][3] = 0
board_w_minus_10.board[6][5] = 0
board_w_minus_10.board[7][0] = 0
board_w_minus_10.board[7][2] = 0
board_w_minus_10.board[7][4] = 0
board_w_minus_10.board[5][6] = 0
encoder.show_board(board_w_minus_10)




class Encoder:
    """Кодирует доску шашек в 10-плоскостное представление"""

    def __init__(self):
        self.num_planes = 10
        self.rows = 8
        self.cols = 8

    def encode(self, game):
        """Кодирует текущее состояние доски в 10-плоскостное представление

        Плоскости:
        0: Обычные белые шашки
        1: Белые дамки
        2: Обычные черные шашки
        3: Черные дамки
        4: Ход белых (1 если ход белых)
        5: Ход черных (1 если ход черных)
        6: Белые шашки под боем
        7: Черные шашки под боем
        8: Белые шашки, которые могут бить
        9: Черные шашки, которые могут бить
        """

        game_matrix = np.zeros((self.num_planes, 8, 8))

        # Отмечаем чей ход
        # if game.current_player == 1:  # Ход белых
        #     game_matrix[4] = 1
        # else:  # Ход черных
        #     game_matrix[5] = 1

        # Определяем шашки под боем и шашки, которые могут бить
        threatened_pieces = set()
        taker_pieces = set()

        # Получаем все возможные взятия
        all_capture_moves = game.get_capture_moves()

        # Извлекаем информацию о шашках под боем и шашках, которые могут бить
        for capture_sequence in all_capture_moves:
            move = capture_sequence[0]
            # Шашка под боем (координаты взятой шашки)
            threatened_pieces.add((move[2], move[3]))
            # Шашка, которая может бить
            taker_pieces.add((move[0], move[1]))

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
            else:  # Черные шашки
                if abs(piece) > 1:  # Дамка
                    game_matrix[3, row, col] = 1
                else:  # Обычная шашка
                    game_matrix[2, row, col] = 1

            # # Отмечаем шашки под боем
            # if (row, col) in threatened_pieces:
            #     if game.board[row][col] > 0:  # Белая шашка под боем
            #         game_matrix[4, row, col] = 1
            #     else:  # Черная шашка под боем
            #         game_matrix[5, row, col] = 1
            #
            # # Отмечаем шашки, которые могут бить
            # if (row, col) in taker_pieces:
            #     if game.board[row][col] > 0:  # Белая шашка может бить
            #         game_matrix[6, row, col] = 1
            #     else:  # Черная шашка может бить
            #         game_matrix[7, row, col] = 1

        return game_matrix

