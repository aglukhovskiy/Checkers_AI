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
from encoders.tenplane_v2 import TenPlaneEncoder
from rl.experience import ExperienceCollector, combine_experience, load_experience
from Board_v2 import CheckersGame
import pygame

# class ExperienceCollector:
#     """Собирает опыт игры для обучения"""
#
#     def __init__(self):
#         self.states = []
#         self.action_results = []
#         self.rewards = []
#         self.white_turns = []
#         self.game_nums = []
#         self._current_episode_states = []
#         self._current_episode_action_results = []
#         self._current_episode_white_turns = []
#         self._current_episode_game_nums = []
#
#     def begin_episode(self):
#         self._current_episode_states = []
#         self._current_episode_action_results = []
#         self._current_episode_white_turns = []
#         self._current_episode_game_nums = []
#
#     def record_decision(self, state, action_result, white_turns, game_nums):
#         self._current_episode_states.append(state)
#         self._current_episode_action_results.append(action_result)
#         self._current_episode_white_turns.append(white_turns)
#         self._current_episode_game_nums.append(game_nums)
#
#     def complete_episode(self, reward):
#         # Устанавливаем вознаграждение всем действиям этого эпизода
#         num_states = len(self._current_episode_states)
#         self.states += self._current_episode_states
#         self.action_results += self._current_episode_action_results
#         self.white_turns += self._current_episode_white_turns
#         self.game_nums += self._current_episode_game_nums
#         self.rewards += [reward] * num_states
#
#     def serialize(self, h5file):
#         """Сохраняет опыт в HDF5 файл"""
#         h5file.create_group('experience')
#         h5file['experience'].create_dataset('states', data=np.array(self.states))
#         h5file['experience'].create_dataset('action_results', data=np.array(self.action_results))
#         h5file['experience'].create_dataset('rewards', data=np.array(self.rewards))
#         h5file['experience'].create_dataset('white_turns', data=np.array(self.white_turns))
#         h5file['experience'].create_dataset('game_nums', data=np.array(self.game_nums))

# def combine_experience(collectors):
#     """Объединяет опыт из нескольких коллекторов"""
#     combined = ExperienceCollector()
#
#     for collector in collectors:
#         if not collector.states:
#             continue
#         combined.states += collector.states
#         combined.action_results += collector.action_results
#         combined.rewards += collector.rewards
#         combined.white_turns += collector.white_turns
#         combined.game_nums += collector.game_nums
#
#     return combined

# def load_experience(h5file):
#     """Загружает опыт из HDF5 файла"""
#     collector = ExperienceCollector()
#     collector.states = list(h5file['experience']['states'])
#     collector.action_results = list(h5file['experience']['action_results'])
#     collector.rewards = list(h5file['experience']['rewards'])
#     collector.white_turns = list(h5file['experience']['white_turns'])
#     collector.game_nums = list(h5file['experience']['game_nums'])
#     return collector


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

# class CheckersGame:
#     """Класс для игры в шашки для агента и симуляций"""
#
#     def __init__(self, board=None):
#         if board:
#             self.board = board
#         else:
#             self.board = CheckersGameBoard()
#
#     def next_turn(self, move=None):
#         """Выполняет ход и переключает игрока"""
#         if move:
#             # Это может быть список ходов (для множественного взятия)
#             if isinstance(move, list):
#                 for single_move in move:
#                     self.board.move_piece(single_move, capture_move=(single_move[2] is not None))
#             else:
#                 self.board.move_piece(move, capture_move=(move[2] is not None))
#
#             self.board.current_player = -self.board.current_player
#             self.board.check_winner()
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
        moves = game.available_moves()[0]
        num_moves = len(moves)

        if num_moves == 0:
            game.game_is_on = 0
            return None

        simulated_boards = []
        board_tensors = []
        x_list = []

        for i in range(num_moves):
            simulated_game = deepcopy(game)
            move = moves[i]

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

            # Переходим к следующему игроку
            simulated_game.current_player = -simulated_game.current_player
            simulated_game.whites_turn = 1 if simulated_game.current_player == 1 else 0

            # Если после хода ход снова того же игрока (множественное взятие),
            # выполняем случайные ходы, пока ход не перейдет к другому игроку
            while simulated_game.current_player != game.current_player and simulated_game.game_is_on == 1:
                available_moves = simulated_game.available_moves()[0]
                if not available_moves:
                    break
                random_move = random.choice(available_moves)
                is_capture = random_move[2] is not None
                simulated_game.move_piece(random_move, capture_move=is_capture)

                # Проверяем множественное взятие для случайного хода
                if is_capture:
                    next_pos = (random_move[4], random_move[5])
                    continue_capture = True
                    while continue_capture:
                        next_captures = simulated_game.get_first_capture_moves(next_pos)
                        if next_captures:
                            next_move = next_captures[0]  # Берем первый возможный ход
                            is_capture = next_move[2] is not None
                            simulated_game.move_piece(next_move, capture_move=is_capture)
                            next_pos = (next_move[4], next_move[5])
                        else:
                            continue_capture = False

                simulated_game.current_player = -simulated_game.current_player
                simulated_game.whites_turn = 1 if simulated_game.current_player == 1 else 0

            simulated_boards.append(simulated_game)
            board_tensor = self._encoder.encode(simulated_game)
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
        if game.current_player == -1:
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
                state=self._encoder.encode(game),
                action_result=board_tensors[ranked_moves[0]],
                white_turns=1 if game.current_player == 1 else 0,
                game_nums=game_num_for_record
            )

        return chosen_move

    def serialize(self, h5file):
        """Сохраняет агента в HDF5 файл"""
        h5file.create_group('encoder')
        h5file['encoder'].attrs['name'] = self._encoder.name()
        h5file.create_group('model')
        kerasutil_save_model_to_hdf5_group(self._model, h5file['model'])

    def train(self, experience, lr=0.01, clipnorm=1.0, batch_size=512, epochs=1):
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
    board = CheckersGame()

    agents = {
        1: white_player,  # белые
        -1: black_player,  # черные
    }

    moves_counter = 0
    max_moves = 80  # Ограничиваем количество ходов для предотвращения бесконечных игр

    while board.game_is_on == 1 and moves_counter < max_moves:
        moves_counter += 1
        current_agent = agents[board.current_player]

        next_move = current_agent.select_move(board, game_num_for_record)
        if next_move is None:
            break

        moves.append(next_move)
        board.next_turn(next_move)

        # Проверяем, закончилась ли игра
        if board.game_is_on == 0:
            break

    # Если достигнут лимит ходов, определяем победителя по количеству шашек
    if moves_counter >= max_moves and board.game_is_on == 1:
        board.game_is_on = 0

    game_result, game_margin = board.compute_results()

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
    game = CheckersGame()
    # board = game

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
                piece = game.get_piece(row, col)
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
            player_text = "Ваш ход" if game.current_player == player_color else "Ход бота"

        player_color_text = "Вы играете за " + ("белых" if player_color == 1 else "черных")
        player_surface = info_font.render(player_color_text, True, TEXT_COLOR)
        screen.blit(player_surface, (BOARD_SIZE + 20, 90))

        turn_surface = info_font.render(player_text, True, HIGHLIGHT_TEXT)
        screen.blit(turn_surface, (BOARD_SIZE + 20, 120))

        # Счет игры
        pieces = game.get_number_of_pieces_and_kings()
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
        if game.winner != 0:
            if game.winner == player_color:
                winner_text = "Вы победили!"
            elif game.winner == -player_color:
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

        piece = game.get_piece(row, col)

        # Проверяем, что это шашка игрока
        if piece != 0 and np.sign(piece) == player_color:
            # Проверяем наличие обязательных взятий у любой шашки
            all_captures = game.get_capture_moves()

            if all_captures:
                # Есть обязательные взятия, проверяем для этой шашки
                piece_captures = []
                for sequence in all_captures:
                    if sequence and sequence[0][0] == row and sequence[0][1] == col:
                        # Добавляем только первый ход из последовательности
                        piece_captures.append(sequence[0])

                if piece_captures:
                    # Эта шашка может совершить взятие, выбираем её
                    selected_piece = (row, col)
                    possible_moves = piece_captures
                else:
                    # Эта шашка не может совершать взятие, но другие могут
                    print("У этой шашки нет обязательных взятий!")
            else:
                # Нет обязательных взятий, проверяем обычные ходы
                regular_moves = game.get_regular_moves_for_piece(row, col)
                if regular_moves:
                    selected_piece = (row, col)
                    possible_moves = regular_moves
                else:
                    print("У этой шашки нет доступных ходов!")

    def handle_click(pos):
        """Обрабатывает клик мыши"""
        nonlocal selected_piece, possible_moves

        # Если клик за пределами доски или ход бота, игнорируем
        if pos[0] >= BOARD_SIZE or bot_thinking or game.current_player != player_color:
            return

        # Если игра закончена, игнорируем клики
        if game.game_is_on == 0:
            return

        col = pos[0] // SQUARE_SIZE
        row = pos[1] // SQUARE_SIZE

        # Если шашка уже выбрана
        if selected_piece:
            # Проверяем, выбрал ли игрок клетку для хода
            target_move = None
            for move in possible_moves:
                if move[4] == row and move[5] == col:
                    target_move = move
                    break

            # Если выбран допустимый ход
            if target_move:
                # Определяем, является ли это взятием
                is_capture = target_move[2] is not None

                # Выполняем ход
                game.move_piece(target_move, capture_move=is_capture)

                # Если это было взятие, проверяем возможность продолжения взятия
                if is_capture:
                    next_captures = game.get_first_capture_moves((row, col))
                    if next_captures:
                        # Если можно продолжить взятие, обновляем выбор и возможные ходы
                        selected_piece = (row, col)
                        possible_moves = next_captures
                        return

                # Если нет продолжения взятия или это был обычный ход, переходим к следующему игроку
                game.current_player = -game.current_player
                game.whites_turn = 1 if game.current_player == 1 else 0
                selected_piece = None
                possible_moves = []
                game.check_winner()
            else:
                # Если клик не по возможному ходу, проверяем, выбрал ли игрок свою шашку
                piece = game.get_piece(row, col)
                if piece != 0 and np.sign(piece) == player_color:
                    # Пытаемся выбрать новую шашку
                    check_and_select_piece(row, col)
                else:
                    # Отменяем выбор
                    selected_piece = None
                    possible_moves = []
        else:
            # Если шашка не выбрана, пытаемся выбрать
            check_and_select_piece(row, col)

    def bot_move():
        """Выполняет ход бота"""
        nonlocal bot_thinking

        if game.game_is_on == 0 or game.current_player == player_color:
            return

        bot_thinking = True

        # Получаем ход от бота
        move = bot.select_move(game, 0)

        if move:
            # Выполняем ход
            is_capture = move[2] is not None
            game.move_piece(move, capture_move=is_capture)

            # Проверяем, есть ли дальнейшие взятия (множественное взятие)
            next_pos = (move[4], move[5])
            while is_capture:
                next_captures = game.get_first_capture_moves(next_pos)
                if next_captures:
                    # Выбираем следующий ход бота
                    next_move = bot.select_move(game, 0)
                    if next_move:
                        is_capture = next_move[2] is not None
                        game.move_piece(next_move, capture_move=is_capture)
                        next_pos = (next_move[4], next_move[5])
                    else:
                        break
                else:
                    break

            # Переключаем игрока
            game.current_player = -game.current_player
            game.whites_turn = 1 if game.current_player == 1 else 0
            game.check_winner()

        bot_thinking = False

    def reset_game():
        """Сбрасывает игру в начальное состояние"""
        nonlocal selected_piece, possible_moves

        game = CheckersGame().board
        selected_piece = None
        possible_moves = []

    def change_color():
        """Меняет цвет игрока"""
        nonlocal player_color, selected_piece, possible_moves

        player_color = -player_color
        reset_game()

    # Основной игровой цикл
    running = True
    while running:
        # Обработка событий
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:  # Левая кнопка мыши
                    handle_click(event.pos)
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:  # Клавиша R для рестарта
                    reset_game()
                elif event.key == pygame.K_c:  # Клавиша C для смены цвета
                    change_color()

        # Если ходит бот, выполняем его ход
        if game.current_player != player_color and game.game_is_on == 1 and not bot_thinking:
            bot_move()

        # Отрисовка
        screen.fill(BLACK)
        draw_board()
        draw_selected()
        draw_pieces()
        draw_info_panel()

        # Обновление экрана
        pygame.display.flip()
        clock.tick(FPS)

    # Выход
    pygame.quit()

# Пример использования
if __name__ == "__main__":
    pass
    # Укажите, какую функцию вы хотите выполнить

    # 1. Для самоигры и генерации опыта
    # do_self_play(
    #     agent_filename='model_checkers.hdf5',
    #     num_games=500,
    #     temperature=0.1,
    #     experience_filename='experience_checkers'
    # )

    # 2. Для обучения агента на основе собранного опыта
    # train_agent(
    #     agent_filename='model_checkers.hdf5',
    #     experience_filename='experience_checkers_500_games.hdf5',
    #     learning_rate=0.0000001,
    #     batch_size=64,
    #     epochs=5
    # )

    # 3. Для игры против бота
    # play_against_bot('model_checkers.hdf5')

    # По умолчанию запускаем игру против бота
    play_against_bot('models_n_exp/test_model.hdf5')

    # model=create_model()
    # encoder = TenPlaneEncoder()
    # new_agent = PolicyAgent(model, encoder)
    # with h5py.File('models_n_exp/test_model.hdf5', 'w') as model_outf:
    #     new_agent.serialize(model_outf)