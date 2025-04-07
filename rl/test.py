import numpy as np
import random
import time
import os
import h5py
from collections import namedtuple
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from encoders.tenplane_v2 import TenPlaneEncoder
from rl.experience import ExperienceCollector, combine_experience, load_experience
from Board_v2 import CheckersGame
import pygame
import timeit
from copy import deepcopy

# Определение устройства для PyTorch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Замена keras.models.Sequential и слоев на PyTorch
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        out = F.relu(out)
        return out


class PolicyModel(nn.Module):
    def __init__(self, input_shape=(8, 8, 10)):
        super(PolicyModel, self).__init__()

        # Преобразуем (H, W, C) в (C, H, W)
        channels = input_shape[2]

        # Аналог ZeroPadding2D и Conv2D в Keras
        self.conv1 = nn.Conv2d(channels, 48, kernel_size=7, padding=3)
        self.conv2 = nn.Conv2d(48, 32, kernel_size=5, padding=2)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=5, padding=2)
        self.conv4 = nn.Conv2d(32, 32, kernel_size=5, padding=2)

        # Вычисляем размер входа для полносвязного слоя
        # После серии сверток размер останется таким же, как входной из-за padding
        dense_input_size = 32 * input_shape[0] * input_shape[1]

        # Аналог Dense в Keras
        self.fc1 = nn.Linear(dense_input_size, 512)
        self.fc2 = nn.Linear(512, 1)

    def forward(self, x):
        # x имеет форму (batch_size, channels, height, width)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))

        # Flatten
        x = x.view(x.size(0), -1)

        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x


def create_model(input_shape=(8, 8, 10)):
    # Преобразуем (H, W, C) в (C, H, W) для PyTorch
    return PolicyModel(input_shape).to(device)


# Функция для сохранения модели в HDF5
def torchutil_save_model_to_hdf5_group(model, h5group):
    state_dict = model.state_dict()
    for key, value in state_dict.items():
        h5group.create_dataset(f"weight_{key}", data=value.cpu().numpy())

    # Сохраняем архитектуру модели как строку
    architecture = str(model)
    h5group.attrs['architecture'] = architecture

    # Сохраняем информацию об устройстве
    h5group.attrs['device'] = str(device)


# Функция для загрузки модели из HDF5
def torchutil_load_model_from_hdf5_group(h5group, input_shape=(8, 8, 10)):
    model = create_model(input_shape)
    state_dict = {}

    for key in h5group.keys():
        if key.startswith('weight_'):
            param_name = key[len('weight_'):]
            weight_data = h5group[key][()]
            state_dict[param_name] = torch.tensor(weight_data, device=device)

    model.load_state_dict(state_dict)
    return model


# Класс GameRecord
GameRecord = namedtuple('GameRecord', 'moves winner margin')


# Функция для симуляции игры
def simulate_game(white_player, black_player, game_num_for_record):
    """Симулирует игру между двумя агентами"""
    moves = []
    game = CheckersGame()

    agents = {
        1: white_player,  # белые
        -1: black_player,  # черные
    }

    moves_counter = 0
    max_moves = 80  # Ограничиваем количество ходов для предотвращения бесконечных игр

    while game.game_is_on == 1 and moves_counter < max_moves:
        moves_counter += 1
        current_agent = agents[game.current_player]

        # Получаем ход от агента
        next_move = current_agent.select_move(game, game_num_for_record)
        if next_move is None:
            break

        # Добавляем ход в историю
        moves.append(next_move)

        # Проверяем тип next_move и выполняем его
        if isinstance(next_move, tuple) and len(next_move) >= 6:
            # Выполняем одиночный ход
            is_capture = next_move[2] is not None
            game.move_piece(next_move, capture_move=is_capture)

            # Проверяем множественное взятие
            if is_capture:
                next_pos = (next_move[4], next_move[5])
                while True:
                    next_captures = game.get_first_capture_moves(next_pos)
                    if not next_captures:
                        break

                    # Выполняем следующее взятие
                    next_capture = next_captures[0]
                    is_capture = next_capture[2] is not None
                    game.move_piece(next_capture, capture_move=is_capture)
                    next_pos = (next_capture[4], next_capture[5])
                    moves.append(next_capture)

        # Переключаем игрока
        game.current_player = -game.current_player
        game.check_winner()

        # Проверяем, закончилась ли игра
        if game.game_is_on == 0:
            break

    # Если достигнут лимит ходов, определяем победителя по количеству шашек
    if moves_counter >= max_moves and game.game_is_on == 1:
        game.game_is_on = 0

    game_result, game_margin = game.compute_results()

    return GameRecord(
        moves=moves,
        winner=game_result,
        margin=game_margin,
    )


# Класс PolicyAgent с использованием PyTorch
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
        # Преобразуем (10, 8, 8) в (1, 10, 8, 8) для батча
        tensor = torch.tensor(board_tensor, dtype=torch.float32, device=device)
        return tensor.unsqueeze(0)  # Добавляем размерность батча

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
            x = self._prepare_input(board_tensor)
            x_list.append(x)

        # Либо исследуем случайные ходы, либо следуем текущей политике
        if np.random.random() < self._temperature:
            move_probs = np.ones(num_moves) / num_moves
        else:
            self._model.eval()  # Устанавливаем модель в режим оценки
            with torch.no_grad():  # Отключаем вычисление градиентов для ускорения
                move_probs = [self._model(x).item() for x in x_list]

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
                    white_turns=1 if game.current_player == 1 else 0,
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
        torchutil_save_model_to_hdf5_group(self._model, h5file['model'])

    def train(self, experience, lr=0.01, clipnorm=1.0, batch_size=512, epochs=1):
        """Обучает модель на основе опыта"""
        # Создаем простой оптимизатор без использования torch._dynamo
        parameters = list(self._model.parameters())
        optimizer = optim.SGD([
            {'params': parameters, 'lr': lr}
        ])

        # Преобразуем данные опыта в тензоры PyTorch
        n = len(experience.action_results)
        x_tensor_list = []
        y_tensor_list = []

        # Обрабатываем данные пакетами, чтобы не перегружать память
        batch_size_process = 1000
        for idx in range(0, n, batch_size_process):
            end_idx = min(idx + batch_size_process, n)
            batch_actions = experience.action_results[idx:end_idx]
            batch_rewards = experience.rewards[idx:end_idx]

            # Преобразуем действия в нужный формат (C, H, W)
            batch_actions_tensors = []
            for action in batch_actions:
                # Преобразуем из (10, 8, 8) в (10, 8, 8) для PyTorch
                action_tensor = torch.tensor(action, dtype=torch.float32)
                batch_actions_tensors.append(action_tensor)

            x_batch = torch.stack(batch_actions_tensors)
            y_batch = torch.tensor(batch_rewards, dtype=torch.float32).view(-1, 1)

            x_tensor_list.append(x_batch)
            y_tensor_list.append(y_batch)

        # Объединяем все обработанные данные
        x_all = torch.cat(x_tensor_list)
        y_all = torch.cat(y_tensor_list)

        # Создаем DataLoader
        dataset = TensorDataset(x_all, y_all)
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        # Обучение модели
        self._model.train()

        for epoch in range(epochs):
            total_loss = 0.0
            for x_batch, y_batch in data_loader:
                # Перемещаем данные на устройство
                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)

                # Очищаем градиенты
                optimizer.zero_grad()

                # Прямой проход
                predictions = self._model(x_batch)

                # Вычисляем функцию потерь (mean_absolute_error)
                loss = F.l1_loss(predictions, y_batch)

                # Обратное распространение
                loss.backward()

                # Ограничиваем градиент вручную (clipnorm)
                torch.nn.utils.clip_grad_norm_(self._model.parameters(), clipnorm)

                # Делаем шаг оптимизации
                optimizer.step()

                total_loss += loss.item()

            print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(data_loader):.6f}")


def load_policy_agent(h5file, input_shape=(8, 8, 10)):
    """Загружает агента из HDF5 файла"""
    model = torchutil_load_model_from_hdf5_group(h5file['model'], input_shape)

    encoder_name = h5file['encoder'].attrs['name']
    if not isinstance(encoder_name, str):
        encoder_name = encoder_name.decode('ascii')

    encoder = TenPlaneEncoder()
    return PolicyAgent(model, encoder)


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
    start = timeit.default_timer()
    for i in range(num_games):
        if i % 10 == 0:
            print(f'Симуляция игры {i + 1}/{num_games}...')
            print("time spent :", timeit.default_timer() - start)
            start = timeit.default_timer()

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
        if game_record.winner == color1:  # Победа игрока 1
            collector1.complete_episode(reward=1)
            collector2.complete_episode(reward=-1)
        elif game_record.winner == -color1:  # Победа игрока 2
            collector2.complete_episode(reward=1)
            collector1.complete_episode(reward=-1)
        else:  # Ничья
            collector1.complete_episode(reward=0)
            collector2.complete_episode(reward=0)

        # Меняем цвета для следующей игры
        color1 = 0 - color1

        # Периодически сохраняем опыт
        if i % 100 == 0 and i > 0:
            experience = combine_experience([collector1, collector2])
            # Можно добавить сохранение промежуточных результатов при необходимости

    # Сохраняем окончательный опыт
    experience = combine_experience([collector1, collector2])
    save_filename = f'{experience_filename}.hdf5'
    print(f'Сохранение буфера опыта в {save_filename}')
    with h5py.File(save_filename, 'w') as experience_file:
        experience.serialize(experience_file)


def train_agent(agent_filename, experience_filename, learning_rate=0.01, batch_size=512, epochs=1):
    """Обучает агента на основе опыта"""

    # Загружаем агента
    with h5py.File(agent_filename, 'r') as agent_file:
        agent = load_policy_agent(agent_file)

    # Загружаем опыт
    with h5py.File(experience_filename, 'r') as experience_file:
        experience = load_experience(experience_file)

    # Обучаем агента
    agent.train(experience, lr=learning_rate, batch_size=batch_size, epochs=epochs)

    return agent


# Пример использования
if __name__ == "__main__":
    # Создание новой модели и агента
    model = create_model()
    encoder = TenPlaneEncoder()
    new_agent = PolicyAgent(model, encoder)

    # Сохранение агента в файл
    with h5py.File('models_n_exp/test_model_pytorch.hdf5', 'w') as model_outf:
        new_agent.serialize(model_outf)

    # Обучение агента на существующем опыте
    trained_agent = train_agent(
        agent_filename='models_n_exp/test_model_pytorch.hdf5',
        experience_filename='models_n_exp/experience_checkers_all_iters.hdf5',
        learning_rate=0.01,
        batch_size=128,
        epochs=1
    )

    # Сохранение обученного агента
    with h5py.File('models_n_exp/test_model_pytorch_trained.hdf5', 'w') as model_outf:
        trained_agent.serialize(model_outf)