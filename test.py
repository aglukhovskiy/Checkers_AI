import numpy as np
import random
import time
import os
import h5py
from copy import deepcopy, copy
from collections import namedtuple
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Conv2D, ZeroPadding2D
from keras.optimizers import SGD


class TenPlaneEncoder:
    """Кодирует доску шашек в 10-плоскостное представление"""

    def __init__(self):
        self.num_planes = 10
        self.rows = 8
        self.cols = 8

    def name(self):
        return 'tenplane'

    def encode(self, board):
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

        board_matrix = np.zeros((self.num_planes, 8, 8))

        # Отмечаем чей ход
        if board.current_player == 1:  # Ход белых
            board_matrix[4] = 1
        else:  # Ход черных
            board_matrix[5] = 1

        # Определяем шашки под боем и шашки, которые могут бить
        threatened_pieces = set()
        taker_pieces = set()

        # Получаем все возможные взятия
        all_capture_moves = board.get_capture_moves()

        # Извлекаем информацию о шашках под боем и шашках, которые могут бить
        for capture_sequence in all_capture_moves:
            for move in capture_sequence:
                # Шашка под боем (координаты взятой шашки)
                threatened_pieces.add((move[2], move[3]))
                # Шашка, которая может бить
                taker_pieces.add((move[0], move[1]))

        # Заполняем плоскости для шашек
        for row in range(8):
            for col in range(8):
                piece = board.get_piece(row, col)

                if piece is None or piece == 0:
                    continue

                # Определяем тип шашки
                if piece > 0:  # Белые шашки
                    if abs(piece) > 1:  # Дамка
                        board_matrix[1, row, col] = 1
                    else:  # Обычная шашка
                        board_matrix[0, row, col] = 1
                else:  # Черные шашки
                    if abs(piece) > 1:  # Дамка
                        board_matrix[3, row, col] = 1
                    else:  # Обычная шашка
                        board_matrix[2, row, col] = 1

                # Отмечаем шашки под боем
                if (row, col) in threatened_pieces:
                    if piece > 0:  # Белая шашка под боем
                        board_matrix[6, row, col] = 1
                    else:  # Черная шашка под боем
                        board_matrix[7, row, col] = 1

                # Отмечаем шашки, которые могут бить
                if (row, col) in taker_pieces:
                    if piece > 0:  # Белая шашка может бить
                        board_matrix[8, row, col] = 1
                    else:  # Черная шашка может бить
                        board_matrix[9, row, col] = 1

        return board_matrix

    def shape(self):
        return self.num_planes, 8, 8


class ExperienceCollector:
    """Собирает опыт игры для обучения"""

    def __init__(self):
        self.states = []
        self.action_results = []
        self.rewards = []
        self.white_turns = []
        self.game_nums = []
        self._current_episode_states = []
        self._current_episode_action_results = []
        self._current_episode_white_turns = []
        self._current_episode_game_nums = []

    def begin_episode(self):
        self._current_episode_states = []
        self._current_episode_action_results = []
        self._current_episode_white_turns = []
        self._current_episode_game_nums = []

    def record_decision(self, state, action_result, white_turns, game_nums):
        self._current_episode_states.append(state)
        self._current_episode_action_results.append(action_result)
        self._current_episode_white_turns.append(white_turns)
        self._current_episode_game_nums.append(game_nums)

    def complete_episode(self, reward):
        # Устанавливаем вознаграждение всем действиям этого эпизода
        num_states = len(self._current_episode_states)
        self.states += self._current_episode_states
        self.action_results += self._current_episode_action_results
        self.white_turns += self._current_episode_white_turns
        self.game_nums += self._current_episode_game_nums
        self.rewards += [reward] * num_states

    def serialize(self, h5file):
        """Сохраняет опыт в HDF5 файл"""
        h5file.create_group('experience')
        h5file['experience'].create_dataset('states', data=np.array(self.states))
        h5file['experience'].create_dataset('action_results', data=np.array(self.action_results))
        h5file['experience'].create_dataset('rewards', data=np.array(self.rewards))
        h5file['experience'].create_dataset('white_turns', data=np.array(self.white_turns))
        h5file['experience'].create_dataset('game_nums', data=np.array(self.game_nums))


def combine_experience(collectors):
    """Объединяет опыт из нескольких коллекторов"""
    combined = ExperienceCollector()

    for collector in collectors:
        if not collector.states:
            continue
        combined.states += collector.states
        combined.action_results += collector.action_results
        combined.rewards += collector.rewards
        combined.white_turns += collector.white_turns
        combined.game_nums += collector.game_nums

    return combined


def load_experience(h5file):
    """Загружает опыт из HDF5 файла"""
    collector = ExperienceCollector()
    collector.states = list(h5file['experience']['states'])
    collector.action_results = list(h5file['experience']['action_results'])
    collector.rewards = list(h5file['experience']['rewards'])
    collector.white_turns = list(h5file['experience']['white_turns'])
    collector.game_nums = list(h5file['experience']['game_nums'])
    return collector


def kerasutil_save_model_to_hdf5_group(model, h5group):
    """Сохраняет модель в формате HDF5"""
    model_json = model.to_json()
    h5group.attrs['model_json'] = model_json
    for layer_index, layer in enumerate(model.layers):
        g = h5group.create_group(f'layer_{layer_index}')
        weights = layer.get_weights()
        for weight_index, weight in enumerate(weights):
            g.create_dataset(f'weight_{weight_index}', data=weight)


def kerasutil_load_model_from_hdf5_group(h5group, custom_objects=None):
    """Загружает модель из формата HDF5"""
    model_json = h5group.attrs['model_json']
    model = keras.models.model_from_json(model_json, custom_objects=custom_objects)

    for layer_index, layer in enumerate(model.layers):
        g = h5group[f'layer_{layer_index}']
        weights = []
        for i in range(len(g.keys())):
            weight = g[f'weight_{i}'][()]
            weights.append(weight)
        layer.set_weights(weights)

    return model


def policy_gradient_loss(y_true, y_pred):
    """Функция потерь для обучения с помощью градиента политики"""
    y_pred = keras.backend.clip(y_pred, keras.backend.epsilon(), 1 - keras.backend.epsilon())
    loss = -1 * y_true * keras.backend.log(y_pred)
    return keras.backend.mean(keras.backend.sum(loss, axis=1))


class CheckersGame:
    """Класс для игры в шашки для агента и симуляций"""

    def __init__(self, board=None):
        if board:
            self.board = board
        else:
            self.board = CheckersGameBoard()

    def next_turn(self, move=None):
        """Выполняет ход и переключает игрока"""
        if move:
            # Это может быть список ходов (для множественного взятия)
            if isinstance(move, list):
                for single_move in move:
                    self.board.move_piece(single_move, capture_move=(single_move[2] is not None))
            else:
                self.board.move_piece(move, capture_move=(move[2] is not None))

            self.board.current_player = -self.board.current_player
            self.board.check_winner()


class CheckersGameBoard:
    """Базовый класс для шашечной доски, адаптированный для работы с агентом"""

    def __init__(self):
        self.pieces = set()
        self.kings = set()
        self.board = self._create_board()
        self.current_player = 1  # 1 для белых, -1 для черных
        self.selected_piece = None
        self.possible_moves = []
        self.capture_moves = []
        self.game_is_on = 1
        self.game_result = []
        self.winner = 0
        self.whites_turn = 1  # Для совместимости с TenPlaneEncoder

    def _create_board(self):
        """Создает начальное расположение шашек"""
        board = np.zeros((8, 8))

        # Расставляем начальные позиции белых шашек
        for row in range(5, 8):
            for col in range(8):
                if (row + col) % 2 == 1:  # только на темных клетках
                    board[row][col] = 1
                    self.pieces.add((row, col))

        # Расставляем начальные позиции черных шашек
        for row in range(0, 3):
            for col in range(8):
                if (row + col) % 2 == 1:  # только на темных клетках
                    board[row][col] = -1
                    self.pieces.add((row, col))

        return board

    def get_piece(self, row, col):
        """Возвращает шашку на указанной позиции"""
        if 0 <= row < 8 and 0 <= col < 8:
            return self.board[row][col]
        return 0

    def is_king(self, row, col):
        """Проверяет, является ли шашка дамкой"""
        piece = self.get_piece(row, col)
        return abs(piece) > 1

    def get_regular_moves_for_piece(self, row, col):
        """Получает обычные ходы для конкретной шашки"""
        moves = []
        piece = self.board[row][col]

        if abs(piece) > 1:  # Дамка
            directions = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
            for dr, dc in directions:
                r, c = row + dr, col + dc
                while 0 <= r < 8 and 0 <= c < 8 and self.board[r][c] == 0:
                    moves.append((row, col, None, None, r, c))
                    r += dr
                    c += dc
        else:  # Обычная шашка
            directions = [(-1, -1), (-1, 1)] if piece > 0 else [(1, -1), (1, 1)]
            for dr, dc in directions:
                r, c = row + dr, col + dc
                if 0 <= r < 8 and 0 <= c < 8 and self.board[r][c] == 0:
                    moves.append((row, col, None, None, r, c))

        return moves

    def get_regular_moves(self):
        """Получает обычные ходы для всех шашек текущего игрока"""
        moves = []
        for row, col in self.pieces:
            piece = self.board[row][col]
            if np.sign(piece) == self.current_player:
                moves.extend(self.get_regular_moves_for_piece(row, col))
        return moves

    def get_capture_moves(self, multicapture_piece=None, current_capture_series=None):
        """Получает все возможные пути взятия, включая множественные взятия"""
        all_capture_sequences = []

        if current_capture_series is None:
            current_capture_series = []

        first_captures = self.get_first_capture_moves(multicapture_piece)

        if not first_captures:
            if current_capture_series:
                return [current_capture_series]
            return []

        for capture in first_captures:
            original_board = self.board.copy()
            original_pieces = self.pieces.copy()

            self.move_piece(capture, capture_move=True)

            new_sequence = current_capture_series + [capture]

            next_pos = (capture[4], capture[5])
            multicapture_sequences = self.get_capture_moves(next_pos, new_sequence)

            if not multicapture_sequences:
                all_capture_sequences.append(new_sequence)
            else:
                all_capture_sequences.extend(multicapture_sequences)

            self.board = original_board.copy()
            self.pieces = original_pieces.copy()

        return all_capture_sequences

    def get_first_capture_moves(self, multicapture_piece=None):
        """Получает ходы со взятием для шашки или всех шашек текущего игрока"""
        first_capture_moves = []

        if multicapture_piece:
            pieces_to_check = [multicapture_piece]
        else:
            pieces_to_check = [piece for piece in self.pieces
                               if np.sign(self.board[piece[0]][piece[1]]) == self.current_player]

        for piece in pieces_to_check:
            row, col = piece[0], piece[1]
            piece_value = self.board[row][col]

            if abs(piece_value) > 1:  # Дамка
                directions = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
                for dr, dc in directions:
                    r, c = row + dr, col + dc
                    enemy_found = False
                    enemy_pos = None

                    while 0 <= r < 8 and 0 <= c < 8:
                        if self.board[r][c] == 0:
                            if enemy_found:
                                first_capture_moves.append((row, col, enemy_pos[0], enemy_pos[1], r, c))
                            r += dr
                            c += dc
                            continue

                        if not enemy_found and np.sign(self.board[r][c]) == -np.sign(piece_value):
                            enemy_found = True
                            enemy_pos = (r, c)
                            r += dr
                            c += dc
                        else:
                            break
            else:  # Обычная шашка
                directions = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
                for dr, dc in directions:
                    r, c = row + dr, col + dc
                    if 0 <= r < 8 and 0 <= c < 8 and self.board[r][c] != 0 and np.sign(self.board[r][c]) == -np.sign(
                            piece_value):
                        capture_r, capture_c = r + dr, c + dc
                        if 0 <= capture_r < 8 and 0 <= capture_c < 8 and self.board[capture_r][capture_c] == 0:
                            first_capture_moves.append((row, col, r, c, capture_r, capture_c))

        return first_capture_moves

    def check_for_kings(self):
        """Проверяет, достигли ли какие-либо шашки противоположного края доски"""
        for col in range(8):
            if self.board[0][col] == 1:  # Белые шашки становятся дамками в верхнем ряду
                self.board[0][col] = 3
            if self.board[7][col] == -1:  # Черные шашки становятся дамками в нижнем ряду
                self.board[7][col] = -3

    def check_for_king(self, row, col):
        """Проверяет, достигла ли конкретная шашка противоположного края доски"""
        if row == 0 and self.board[row][col] == 1:
            self.board[row][col] = 3
        if row == 7 and self.board[row][col] == -1:
            self.board[row][col] = -3

    def available_moves(self):
        """Возвращает доступные ходы для совместимости с TenPlaneEncoder"""
        all_moves = []
        all_capture_sequences = self.get_capture_moves()

        # Если есть взятия, они обязательны
        if all_capture_sequences:
            # Берем первые ходы из каждой последовательности
            for sequence in all_capture_sequences:
                if sequence:
                    all_moves.append(sequence[0])
        else:
            all_moves = self.get_regular_moves()

        # Создаем структуру, совместимую с форматом TenPlaneEncoder
        threatened_pieces = {}
        taker_pieces = {}

        # Извлекаем информацию о шашках под боем и шашках, которые могут бить
        for move in all_moves:
            if move[2] is not None:  # Это взятие
                # Шашка под боем
                threatened_pieces[f'{move[2]},{move[3]}'] = (move[2], move[3])
                # Шашка, которая может бить
                taker_pieces[f'{move[0]},{move[1]}'] = (move[0], move[1])

        return [all_moves, threatened_pieces, taker_pieces]

    def get_possible_moves(self):
        """Получает все возможные ходы"""
        capture_moves = self.get_capture_moves()
        if capture_moves:
            # Объединяем все последовательности взятий
            all_capture_moves = []
            for sequence in capture_moves:
                all_capture_moves.extend(sequence)
            return all_capture_moves
        else:
            return self.get_regular_moves()

    def move_piece(self, move, capture_move=False):
        """Выполняет ход шашкой"""
        if capture_move:
            # Выполняем взятие
            self.board[move[4]][move[5]] = self.board[move[0]][move[1]]
            self.board[move[0]][move[1]] = 0
            # Удаляем взятую шашку
            self.board[move[2]][move[3]] = 0
            # Обновляем позиции
            self.pieces.remove((move[0], move[1]))
            self.pieces.remove((move[2], move[3]))
            self.pieces.add((move[4], move[5]))
            self.check_for_king(move[4], move[5])
        else:
            # Выполняем обычный ход
            self.board[move[4]][move[5]] = self.board[move[0]][move[1]]
            self.board[move[0]][move[1]] = 0
            # Обновляем позиции
            self.pieces.remove((move[0], move[1]))
            self.pieces.add((move[4], move[5]))
            self.check_for_king(move[4], move[5])

    def get_number_of_pieces_and_kings(self):
        """Подсчитывает количество шашек и дамок для каждого игрока"""
        p_w = 0  # белые шашки
        p_b = 0  # черные шашки
        k_w = 0  # белые дамки
        k_b = 0  # черные дамки

        for pos in self.pieces:
            piece = self.board[pos[0]][pos[1]]
            if piece > 0:
                if piece > 1:
                    k_w += 1
                else:
                    p_w += 1
            elif piece < 0:
                if piece < -1:
                    k_b += 1
                else:
                    p_b += 1

        return [p_w, p_b, k_w, k_b]

    def check_winner(self):
        """Проверяет, есть ли победитель"""
        pieces = self.get_number_of_pieces_and_kings()

        if pieces[0] + pieces[2] == 0:  # У белых нет шашек
            self.game_is_on = 0
            self.game_result = pieces
            self.winner = -1
            self.whites_turn = 0  # Для совместимости с TenPlaneEncoder
            return self.winner

        elif pieces[1] + pieces[3] == 0:  # У черных нет шашек
            self.game_is_on = 0
            self.game_result = pieces
            self.winner = 1
            self.whites_turn = 1  # Для совместимости с TenPlaneEncoder
            return self.winner

        # Проверка на наличие возможных ходов
        if not self.get_possible_moves():
            self.game_is_on = 0
            self.game_result = pieces
            if pieces[0] + pieces[2] > pieces[1] + pieces[3]:
                self.winner = 1
            elif pieces[0] + pieces[2] < pieces[1] + pieces[3]:
                self.winner = -1
            else:
                self.winner = 0.5  # Ничья
            return self.winner

        return None

    def compute_results(self):
        """Вычисляет результат игры и разницу в очках"""
        pieces = self.get_number_of_pieces_and_kings()

        # Очки: обычная шашка - 1, дамка - 3
        white_score = pieces[0] + 3 * pieces[2]
        black_score = pieces[1] + 3 * pieces[3]

        margin = white_score - black_score

        if self.winner == 1:
            return 1, margin
        elif self.winner == -1:
            return 0, margin
        elif self.winner == 0.5:
            return 0.5, margin

        # Если нет победителя, определяем по количеству оставшихся шашек
        if white_score > black_score:
            return 1, margin
        elif white_score < black_score:
            return 0, margin
        else:
            return 0.5, margin  # Ничья


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

    def select_move(self, game, game_num_for_record):
        """Выбирает ход на основе текущего состояния игры"""
        moves = game.board.available_moves()[0]
        num_moves = len(moves)

        if num_moves == 0:
            game.board.game_is_on = 0
            return

        simulated_boards = []
        board_tensors = []
        x_list = []

        for i in range(num_moves):
            simulated_game = deepcopy(game)
            # Выполняем ход
            simulated_game.next_turn(move=moves[i])

            # Если после хода ход снова того же игрока (множественное взятие),
            # выполняем случайные ходы, пока ход не перейдет к другому игроку
            while simulated_game.board.current_player == game.board.current_player and simulated_game.board.game_is_on == 1:
                available_moves = simulated_game.board.available_moves()[0]
                if not available_moves:
                    break
                random_move = random.choice(available_moves)
                simulated_game.next_turn(move=random_move)

            simulated_boards.append(simulated_game)
            board_tensor = self._encoder.encode(simulated_game.board)
            board_tensors.append(board_tensor)

            # Преобразуем из (10, 8, 8) в (8, 8, 10)
            x = np.transpose(board_tensor, (1, 2, 0)).reshape(1, 8, 8, 10)
            x_list.append(x)

        # Либо исследуем случайные ходы, либо следуем текущей политике
        if np.random.random() < self._temperature:
            move_probs = np.ones(num_moves) / num_moves
        else:
            move_probs = [self._model.predict(x, verbose=0)[0][0] for x in x_list]

        # Предотвращаем вероятности 0 или 1
        eps = 1e-5
        move_probs = np.clip(move_probs, eps, 1 - eps)

        # Инвертируем вероятности для черных (т.к. модель обучена на максимизацию выигрыша белых)
        if game.board.current_player == -1:
            move_probs = [1 - x for x in move_probs]

        # Нормализуем, чтобы получить распределение вероятностей
        move_probs = move_probs / np.sum(move_probs)

        # Выбираем ход в соответствии с вероятностями
        candidates = np.arange(num_moves)
        ranked_moves = np.random.choice(candidates, num_moves, replace=False, p=move_probs)
        chosen_move = moves[ranked_moves[0]]

        # Записываем решение, если есть коллектор
        if self._collector is not None:
            self._collector.record_decision(
                state=self._encoder.encode(game.board),
                action_result=board_tensors[ranked_moves[0]],
                white_turns=1 if game.board.current_player == 1 else 0,
                game_nums=game_num_for_record
            )

        return chosen_move

    def serialize(self, h5file):
        """Сохраняет агента в HDF5 файл"""
        h5file.create_group('encoder')
        h5file['encoder'].attrs['name'] = self._encoder.name()
        h5file.create_group('model')
        kerasutil_save_model_to_hdf5_group(self._model, h5file['model'])

    def train(self, experience, lr=0.0000001, clipnorm=1.0, batch_size=512, epochs=1):
        """Обучает модель на основе опыта"""
        opt = SGD(learning_rate=lr, clipnorm=clipnorm)

        self._model.compile(loss='binary_crossentropy', optimizer=opt)

        n = experience.action_results.shape[0]
        y = np.zeros(n)
        for i in range(n):
            reward = experience.rewards[i]
            y[i] = reward

        x = experience.action_results

        self._model.fit(x=x, batch_size=batch_size, y=y, epochs=epochs)


def load_policy_agent(h5file):
    """Загружает агента из HDF5 файла"""
    model = kerasutil_load_model_from_hdf5_group(
        h5file['model'],
        custom_objects={'policy_gradient_loss': policy_gradient_loss})

    encoder_name = h5file['encoder'].attrs['name']
    if not isinstance(encoder_name, str):
        encoder_name = encoder_name.decode('ascii')

    encoder = TenPlaneEncoder()
    return PolicyAgent(model, encoder)


class GameRecord(namedtuple('GameRecord', 'moves winner margin')):
    """Запись о сыгранной игре"""
    pass


def simulate_game(white_player, black_player, game_num_for_record):
    """Симулирует игру между двумя агентами"""
    moves = []
    board = CheckersGameBoard()
    game = CheckersGame(board=board)

    agents = {
        1: white_player,  # белые
        -1: black_player,  # черные
    }

    moves_counter = 0
    max_moves = 80  # Ограничиваем количество ходов для предотвращения бесконечных игр

    while game.board.game_is_on == 1 and moves_counter < max_moves:
        moves_counter += 1
        current_agent = agents[game.board.current_player]

        next_move = current_agent.select_move(game, game_num_for_record)
        if next_move is None:
            break

        moves.append(next_move)
        game.next_turn(next_move)

        # Проверяем, закончилась ли игра
        if game.board.game_is_on == 0:
            break

    # Если достигнут лимит ходов, определяем победителя по количеству шашек
    if moves_counter >= max_moves and game.board.game_is_on == 1:
        game.board.game_is_on = 0

    game_result, game_margin = game.board.compute_results()

    return GameRecord(
        moves=moves,
        winner=game_result,
        margin=game_margin,
    )


def create_model(input_shape=(8, 8, 10)):
    """Создаёт модель нейронной сети для агента"""
    model = Sequential([
        ZeroPadding2D((3, 3), input_shape=input_shape),
        Conv2D(64, (7, 7), padding='valid', activation='relu'),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(1, activation='sigmoid')  # Выход - вероятность победы
    ])
    return model


def do_self_play(agent_filename, num_games, temperature, experience_filename):
    """Выполняет игры агента против самого себя и сохраняет полученный опыт"""

    random.seed(int(time.time()) + os.getpid())
    np.random.seed(int(time.time()) + os.getpid())

    # Загружаем агента из файла или создаем нового, если файл не существует
    try:
        with h5py.File(agent_filename, 'r') as agent_file:
            agent1 = load_policy_agent(agent_file)
            agent2 = load_policy_agent(agent_file)
    except (FileNotFoundError, IOError):
        # Если файл не найден, создаем нового агента
        encoder = TenPlaneEncoder()
        model = create_model()
        agent1 = PolicyAgent(model, encoder)
        agent2 = PolicyAgent(model, encoder)
        # Сохраняем нового агента
        with h5py.File(agent_filename, 'w') as agent_file:
            agent1.serialize(agent_file)

    # Устанавливаем температуру для исследования
    agent1.set_temperature(temperature)
    agent2.set_temperature(temperature)

    # Создаем коллекторы опыта
    collector1 = ExperienceCollector()
    collector2 = ExperienceCollector()

    # Играем заданное количество игр
    color1 = 1  # Начинаем с белых

    for i in range(num_games):
        print(f'Симуляция игры {i + 1}/{num_games}...')

        # Начинаем новый эпизод для коллекторов
        collector1.begin_episode()
        agent1.set_collector(collector1)
        collector2.begin_episode()
        agent2.set_collector(collector2)

        # Распределяем цвета
        if color1 == 1:
            white_player, black_player = agent1, agent2
        else:
            black_player, white_player = agent1, agent2

        # Симулируем игру
        game_record = simulate_game(white_player, black_player, game_num_for_record=i)

        # Определяем вознаграждения на основе результата игры
        if game_record.winner == 1:  # Победа белых
            collector1.complete_episode(reward=1 if color1 == 1 else 0)
            collector2.complete_episode(reward=1 if color1 == 0 else 0)
        elif game_record.winner == 0:  # Победа черных
            collector1.complete_episode(reward=0 if color1 == 1 else 1)
            collector2.complete_episode(reward=0 if color1 == 0 else 1)
        else:  # Ничья
            collector1.complete_episode(reward=0.5)
            collector2.complete_episode(reward=0.5)

        # Меняем цвета для следующей игры
        color1 = 1 - color1

        # Периодически сохраняем опыт
        if i % 100 == 0 and i > 0:
            experience = combine_experience([collector1, collector2])
            save_filename = f'{experience_filename}_{i}_games.hdf5'
            print(f'Сохранение буфера опыта в {save_filename}')
            with h5py.File(save_filename, 'w') as experience_file:
                experience.serialize(experience_file)

    # Сохраняем окончательный опыт
    experience = combine_experience([collector1, collector2])
    save_filename = f'{experience_filename}_{num_games}_games.hdf5'
    print(f'Сохранение буфера опыта в {save_filename}')
    with h5py.File(save_filename, 'w') as experience_file:
        experience.serialize(experience_file)


def train_agent(agent_filename, experience_filename, learning_rate=0.0000001, batch_size=512, epochs=1):
    """Обучает агента на основе опыта"""

    # Загружаем агента
    with h5py.File(agent_filename, 'r') as agent_file:
        agent = load_policy_agent(agent_file)

    # Загружаем опыт
    with h5py.File(experience_filename, 'r') as experience_file:
        experience = load_experience(experience_file)

    # Обучаем агента
    agent.train(experience, lr=learning_rate, batch_size=batch_size, epochs=epochs)

    # Сохраняем обновленного агента
    with h5py.File(agent_filename, 'w') as agent_file:
        agent.serialize(agent_file)


def play_against_bot(agent_filename):
    """Играет против обученного агента"""

    # Загружаем агента
    with h5py.File(agent_filename, 'r') as agent_file:
        bot = load_policy_agent(agent_file)

    # Инициализируем Pygame
    pygame.init()

    # Константы
    BOARD_SIZE = 600
    SQUARE_SIZE = BOARD_SIZE // 8
    PANEL_WIDTH = 300
    WIDTH = BOARD_SIZE + PANEL_WIDTH
    HEIGHT = BOARD_SIZE

    WHITE = (255, 255, 255)
    BLACK = (0, 0, 0)
    DARK_SQUARE = (101, 67, 33)
    LIGHT_SQUARE = (255, 248, 220)
    HIGHLIGHT = (124, 252, 0)
    POSSIBLE_MOVE = (173, 216, 230)
    PANEL_BG = (240, 240, 240)
    PANEL_BORDER = (200, 200, 200)
    TEXT_COLOR = (50, 50, 50)
    HIGHLIGHT_TEXT = (0, 100, 0)

    FPS = 60

    # Создаем игру
    board = CheckersGameBoard()
    game = CheckersGame(board=board)

    # Создаем окно
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption('Шашки против бота')
    clock = pygame.time.Clock()

    # Шрифты
    title_font = pygame.font.SysFont('Arial', 36)
    info_font = pygame.font.SysFont('Arial', 24)
    small_font = pygame.font.SysFont('Arial', 18)

    # Интерфейсные переменные
    selected_piece = None
    possible_moves = []
    player_color = 1  # 1 - белые, -1 - черные
    bot_thinking = False

    def draw_board():
        """Рисует доску"""
        for row in range(8):
            for col in range(8):
                color = LIGHT_SQUARE if (row + col) % 2 == 0 else DARK_SQUARE
                pygame.draw.rect(screen, color, (col * SQUARE_SIZE, row * SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE))

    def draw_pieces():
        """Рисует шашки"""
        for row in range(8):
            for col in range(8):
                piece = game.board.get_piece(row, col)
                if piece != 0:
                    color = WHITE if piece > 0 else BLACK
                    border_color = BLACK if piece > 0 else WHITE
                    center_x = col * SQUARE_SIZE + SQUARE_SIZE // 2
                    center_y = row * SQUARE_SIZE + SQUARE_SIZE // 2

                    pygame.draw.circle(screen, color, (center_x, center_y), SQUARE_SIZE // 2 - 10)
                    pygame.draw.circle(screen, border_color, (center_x, center_y), SQUARE_SIZE // 2 - 10, 2)

                    if abs(piece) > 1:  # Дамка
                        pygame.draw.circle(screen, border_color, (center_x, center_y), SQUARE_SIZE // 4)
                        pygame.draw.circle(screen, color, (center_x, center_y), SQUARE_SIZE // 4 - 5)

    def draw_selected():
        """Отображает выделенную шашку и возможные ходы"""
        if selected_piece:
            row, col = selected_piece
            pygame.draw.rect(screen, HIGHLIGHT,
                             (col * SQUARE_SIZE, row * SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE), 4)

            for move in possible_moves:
                target_row, target_col = move[4], move[5]
                center_x = target_col * SQUARE_SIZE + SQUARE_SIZE // 2
                center_y = target_row * SQUARE_SIZE + SQUARE_SIZE // 2
                pygame.draw.circle(screen, POSSIBLE_MOVE, (center_x, center_y), SQUARE_SIZE // 4)

    def draw_info_panel():
        """Рисует информационную панель"""
        panel_rect = pygame.Rect(BOARD_SIZE, 0, PANEL_WIDTH, HEIGHT)
        pygame.draw.rect(screen, PANEL_BG, panel_rect)
        pygame.draw.rect(screen, PANEL_BORDER, panel_rect, 2)

        title_surface = title_font.render("ШАШКИ VS БОТ", True, TEXT_COLOR)
        screen.blit(title_surface, (BOARD_SIZE + 20, 20))

        pygame.draw.line(screen, PANEL_BORDER,
                         (BOARD_SIZE + 10, 70),
                         (WIDTH - 10, 70),
                         2)

        if bot_thinking:
            player_text = "Ход бота..."
        else:
            player_text = "Ваш ход" if game.board.current_player == player_color else "Ход бота"

        player_color_text = "Вы играете за " + ("белых" if player_color == 1 else "черных")
        player_surface = info_font.render(player_color_text, True, TEXT_COLOR)
        screen.blit(player_surface, (BOARD_SIZE + 20, 90))

        turn_surface = info_font.render(player_text, True, HIGHLIGHT_TEXT)
        screen.blit(turn_surface, (BOARD_SIZE + 20, 120))

        # Счет игры
        pieces = game.board.get_number_of_pieces_and_kings()
        score_text = f"Счет: Белые {pieces[0] + pieces[2]} - {pieces[1] + pieces[3]} Черные"
        score_surface = info_font.render(score_text, True, TEXT_COLOR)
        screen.blit(score_surface, (BOARD_SIZE + 20, 160))

        # Детали счета
        details_text = f"Белые: {pieces[0]} шашек, {pieces[2]} дамок"
        details_surface = small_font.render(details_text, True, TEXT_COLOR)
        screen.blit(details_surface, (BOARD_SIZE + 20, 190))

        details_text = f"Черные: {pieces[1]} шашек, {pieces[3]} дамок"
        details_surface = small_font.render(details_text, True, TEXT_COLOR)
        screen.blit(details_surface, (BOARD_SIZE + 20, 210))

        # Инструкции
        pygame.draw.line(screen, PANEL_BORDER,
                         (BOARD_SIZE + 10, HEIGHT - 120),
                         (WIDTH - 10, HEIGHT - 120),
                         2)

        instructions = [
            "Управление:",
            "- Левый клик: выбор шашки / ход",
            "- R: начать новую игру",
            "- C: сменить цвет"
        ]

        for i, text in enumerate(instructions):
            instr_surface = small_font.render(text, True, TEXT_COLOR)
            screen.blit(instr_surface, (BOARD_SIZE + 20, HEIGHT - 110 + i * 25))

        # Если игра окончена, показываем победителя
        if game.board.winner != 0:
            if game.board.winner == player_color:
                winner_text = "Вы победили!"
            elif game.board.winner == -player_color:
                winner_text = "Бот победил!"
            else:
                winner_text = "Ничья!"

            winner_surface = title_font.render(winner_text, True, (255, 0, 0))
            overlay = pygame.Surface((350, 50), pygame.SRCALPHA)
            overlay.fill((255, 255, 255, 200))
            screen.blit(overlay, (BOARD_SIZE // 2 - 175, BOARD_SIZE // 2 - 25))
            win_rect = winner_surface.get_rect(center=(BOARD_SIZE // 2, BOARD_SIZE // 2))
            screen.blit(winner_surface, win_rect)

    def check_and_select_piece(row, col):
        """Проверяет возможность выбора шашки и определяет возможные ходы"""
        nonlocal selected_piece, possible_moves

        piece = game.board.get_piece(row, col)

        # Проверяем, что это шашка игрока
        if piece != 0 and np.sign(piece) == player_color:

