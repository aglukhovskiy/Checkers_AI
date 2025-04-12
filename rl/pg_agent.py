from rl.kerasutil import kerasutil_save_model_to_hdf5_group, kerasutil_load_model_from_hdf5_group
import random
# from encoders.tenplane_v2 import TenPlaneEncoder
import encoders
from copy import deepcopy
import numpy as np
from keras.optimizers import SGD
from keras.callbacks import CSVLogger

def policy_gradient_loss(y_true, y_pred):
    """Функция потерь для обучения с помощью градиента политики"""
    y_pred = keras.backend.clip(y_pred, keras.backend.epsilon(), 1 - keras.backend.epsilon())
    loss = -1 * y_true * keras.backend.log(y_pred)
    return keras.backend.mean(keras.backend.sum(loss, axis=1))

class PolicyAgent:
    """Агент, использующий глубокую нейронную сеть политики для выбора ходов"""

    def __init__(self, model, encoder):
        self._model = model
        self._encoder = encoder
        self._collector = None
        self._temperature = 0.0

    def set_collector(self, collector):
        self._collector = collector

    def set_temperature(self, temperature):
        self._temperature = temperature

    def _prepare_input(self, board_tensor):
        # Данные уже в формате (10, 8, 8), просто добавляем размерность батча
        return np.expand_dims(np.transpose(board_tensor, (1, 2, 0)), axis=0)
        # return board_tensor.reshape(1, 10, 8, 8)

    def select_move(self, game, game_num_for_record):
        """Выбирает ход на основе текущего состояния игры"""
        moves = game.available_moves()[0]
        num_moves = len(moves)

        if num_moves == 0:
            game.game_is_on = 0
            return None

        # Проверяем, что все ходы имеют правильный формат
        valid_moves = []
        for move in moves:
            if isinstance(move, tuple) and len(move) >= 6:
                valid_moves.append(move)

        if not valid_moves:
            print("Нет допустимых ходов!")
            return None

        num_moves = len(valid_moves)
        simulated_boards = []
        board_tensors = []
        x_list = []

        for i in range(num_moves):
            simulated_game = deepcopy(game)
            move = valid_moves[i]

            # Выполняем текущий ход
            is_capture = move[2] is not None
            simulated_game.move_piece(move, capture_move=is_capture)

            # Проверяем, есть ли множественное взятие
            next_pos = (move[4], move[5])
            while is_capture:
                next_captures = simulated_game.get_first_capture_moves(next_pos)
                if next_captures:
                    # Для симуляции просто выбираем первый возможный ход
                    next_move = next_captures[0]
                    is_capture = next_move[2] is not None
                    simulated_game.move_piece(next_move, capture_move=is_capture)
                    next_pos = (next_move[4], next_move[5])
                else:
                    break

            simulated_boards.append(simulated_game)
            board_tensor = self._encoder.encode(simulated_game)
            board_tensors.append(board_tensor)
            # x = self._prepare_input(board_tensor)
            # x_list.append(x)

        # Либо исследуем случайные ходы, либо следуем текущей политике
        if np.random.random() < self._temperature:
            move_probs = np.ones(num_moves) / num_moves
        else:
            # move_probs = [self._model.predict(np.array([board_tensor]), verbose=0)[0][0] for board_tensor in board_tensors]
            move_probs = self._model.predict(np.array(board_tensors), verbose=0)[:, 0]

        if game.current_player==-1:
            move_probs = [-x for x in move_probs]

        # Предотвращаем вероятности 0 или 1
        eps = 1e-5
        move_probs = np.clip(move_probs, eps, 1 - eps)

        # Нормализуем, чтобы получить распределение вероятностей
        move_probs = move_probs / np.sum(move_probs)

        # Выбираем ход в соответствии с вероятностями
        candidates = np.arange(num_moves)
        try:
            ranked_moves = np.random.choice(candidates, num_moves, replace=False, p=move_probs)
            chosen_move = valid_moves[ranked_moves[0]]

            # Записываем решение, если есть коллектор
            if self._collector is not None:
                self._collector.record_decision(
                    state=self._encoder.encode(game),
                    action_result=board_tensors[ranked_moves[0]],
                    white_turns=game.current_player,
                    game_nums=game_num_for_record
                )

            return chosen_move
        except:
            # В случае ошибки выбираем случайный ход
            if valid_moves:
                return random.choice(valid_moves)
            return None

    def serialize(self, h5file):
        """Сохраняет агента в HDF5 файл"""
        h5file.create_group('encoder')
        h5file['encoder'].attrs['name'] = self._encoder.name()
        h5file.create_group('model')
        kerasutil_save_model_to_hdf5_group(self._model, h5file['model'])

    def train(self, experience, lr=0.01, clipnorm=1.0, batch_size=512, epochs=1, loss='mse'):
        """Обучает модель на основе опыта"""
        # opt = SGD(learning_rate=lr, clipnorm=clipnorm)
        opt = SGD(learning_rate=lr)
        # self._model.compile(loss='mean_absolute_error', optimizer=opt)
        self._model.compile(loss=loss, optimizer=opt)

        n = experience.action_results.shape[0]
        # Translate the actions/rewards.
        y = np.zeros(n)
        for i in range(n):
            advantage = experience.advantages[i]
            y[i] = advantage

        # Данные уже в формате (None, 10, 8, 8)
        x = experience.action_results
        csv_logger = CSVLogger('training.log', append=True)

        self._model.fit(
            x=x, batch_size=batch_size, y=y, epochs=epochs, callbacks=csv_logger)


def load_policy_agent(h5file):
    """Загружает агента из HDF5 файла"""
    model = kerasutil_load_model_from_hdf5_group(
        h5file['model'],
        custom_objects={'policy_gradient_loss': policy_gradient_loss})
    encoder_name = h5file['encoder'].attrs['name']
    if not isinstance(encoder_name, str):
        encoder_name = encoder_name.decode('ascii')
    encoder = encoders.get_encoder_by_name(
        encoder_name)
    return PolicyAgent(model, encoder)