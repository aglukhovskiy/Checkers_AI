import numpy as np
import random
import time
import os
import h5py
from collections import namedtuple
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Conv2D, ZeroPadding2D
from keras.optimizers import SGD
from encoders.tenplane_v2 import TenPlaneEncoder
from rl.experience import ExperienceCollector, combine_experience, load_experience
from Board_v2 import CheckersGame
import pygame
import timeit
from rl.kerasutil import kerasutil_save_model_to_hdf5_group, kerasutil_load_model_from_hdf5_group
from rl.pg_agent import PolicyAgent, load_policy_agent
import tensorflow as tf

# Настраиваем TensorFlow для использования GPU
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    for device in physical_devices:
        tf.config.experimental.set_memory_growth(device, True)
    print("GPU доступен и настроен")
else:
    print("GPU не доступен, будет использоваться CPU")

class GameRecord(namedtuple('GameRecord', 'moves winner margin')):
    """Запись о сыгранной игре"""
    pass


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

    # white_player._encoder.show_board(game)
    # print(game.game_is_on, game.winner)
    # print('------------------------------')
    # Если достигнут лимит ходов, определяем победителя по количеству шашек
    if moves_counter >= max_moves and game.game_is_on == 1:
        game.game_is_on = 0

    game_result, game_margin = game.compute_results()

    return GameRecord(
        moves=moves,
        winner=game_result,
        margin=game_margin,
    )

def layers(input_shape):
    return [
        ZeroPadding2D(padding=3, input_shape=input_shape, data_format='channels_first'),  # <1>
        Conv2D(48, (7, 7), data_format='channels_first'),
        Activation('relu'),

        ZeroPadding2D(padding=2, data_format='channels_first'),  # <2>
        Conv2D(32, (5, 5), data_format='channels_first'),
        Activation('relu'),

        ZeroPadding2D(padding=2, data_format='channels_first'),
        Conv2D(32, (5, 5), data_format='channels_first'),
        Activation('relu'),

        ZeroPadding2D(padding=2, data_format='channels_first'),
        Conv2D(32, (5, 5), data_format='channels_first'),
        Activation('relu'),

        Flatten(),
        Dense(512),
        Activation('relu'),
    ]

def create_model(input_shape=(10, 8, 8)):  # Изменяем порядок размерностей
    """Создаёт модель нейронной сети для агента"""
    model = Sequential()
    for layer in layers(input_shape):
        model.add(layer)
    model.add(Dense(1, activation='linear'))  # Выход - скор
    return model


def do_self_play(agent_filename, num_games, temperature, experience_filename):
    """Выполняет игры агента против самого себя и сохраняет полученный опыт"""

    random.seed(int(time.time()) + os.getpid())
    np.random.seed(int(time.time()) + os.getpid())

    # Загружаем агента из файла или создаем нового, если файл не существует
    with h5py.File(agent_filename, 'r') as agent_file:
        agent1 = load_policy_agent(agent_file)
        agent2 = load_policy_agent(agent_file)

    # except (FileNotFoundError, IOError):
    #     # Если файл не найден, создаем нового агента
    #     encoder = TenPlaneEncoder()
    #     model = create_model()
    #     agent1 = PolicyAgent(model, encoder)
    #     agent2 = PolicyAgent(model, encoder)
    #     # Сохраняем нового агента
    #     with h5py.File(agent_filename, 'w') as agent_file:
    #         agent1.serialize(agent_file)

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
        if i%10==0:
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
        if game_record.winner == color1:  # Победа белых
            collector1.complete_episode(reward=1)
            collector2.complete_episode(reward=-1)
        elif game_record.winner == -color1:  # Победа черных
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
            # save_filename = f'{experience_filename}_{num_games}.hdf5'
            # print(f'Сохранение буфера опыта в {save_filename}')
            # # with h5py.File(save_filename, 'w') as experience_file:
            # #     experience.serialize(experience_file)

    # Сохраняем окончательный опыт
    experience = combine_experience([collector1, collector2])
    # save_filename = f'{experience_filename}_{num_games}.hdf5'
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

    # # Сохраняем обновленного агента
    # with h5py.File(agent_filename, 'w') as agent_file:
    #     agent.serialize(agent_file)

    return agent


# Пример использования
if __name__ == "__main__":
    pass
    # Укажите, какую функцию вы хотите выполнить

    # 1. Для самоигры и генерации опыта
    # do_self_play(
    #     agent_filename='models_n_exp/test_model.hdf5',
    #     num_games=250,
    #     temperature=0.01,
    #     experience_filename='experience_checkers'
    # )

    # 2. Для обучения агента на основе собранного опыта
    trained_agent=train_agent(
        agent_filename='models_n_exp/test_model_small.hdf5',
        experience_filename='models_n_exp/experience_checkers_all_iters.hdf5',
        learning_rate=0.01,
        batch_size=128,
        epochs=1
    )
    with h5py.File('models_n_exp/test_model_small_trained.hdf5', 'w') as model_outf:
        trained_agent.serialize(model_outf)

    # model=create_model()
    # encoder = TenPlaneEncoder()
    # new_agent = PolicyAgent(model, encoder)
    # with h5py.File('models_n_exp/test_model_small.hdf5', 'w') as model_outf:
    #     new_agent.serialize(model_outf)

    # with h5py.File('models_n_exp/test_model.hdf5', 'r') as agent_file:
    #     bot = load_policy_agent(agent_file)
    # res = simulate_game(bot,bot,1)
    # print(res)




    # trained_agent=train_agent(
    #     agent_filename='models_n_exp/test_model_small.hdf5',
    #     experience_filename='experience_checkers_250.hdf5',
    #     learning_rate=0.01,
    #     batch_size=512,
    #     epochs=1
    # )
    # with h5py.File('models_n_exp/test_model_trained.hdf5', 'w') as model_outf:
    #     trained_agent.serialize(model_outf)
    #
    # for j in range(25):
    #     print('starting {} iteration'.format(j+1))
    #     do_self_play(agent_filename='models_n_exp/test_model_trained.hdf5', num_games=100, temperature=0.1,
    #                      experience_filename='models_n_exp/experience_checkers_{}_iter'.format(j))
    #     trained_agent = train_agent(
    #         agent_filename='models_n_exp/test_model_trained.hdf5',
    #         experience_filename='models_n_exp/experience_checkers_{}_iter.hdf5'.format(j),
    #         learning_rate=0.01,
    #         batch_size=512,
    #         epochs=1
    #     )
    #     with h5py.File('models_n_exp/test_model_trained.hdf5', 'w') as model_outf:
    #         trained_agent.serialize(model_outf)
    #
    # exp_list = []
    #
    # for k in range(25):
    #     exp_list.append(load_experience(h5py.File('models_n_exp/experience_checkers_{}_iter.hdf5'.format(k))))
    #
    # total_exp = combine_experience(exp_list)
    #
    # with h5py.File('models_n_exp/experience_checkers_all_iters.hdf5', 'w') as experience_outf:
    #     total_exp.serialize(experience_outf)

    # for z in range(10):
    #     os.remove('models_n_exp/10_plane_{}_iter_100_games.hdf5'.format(k))


