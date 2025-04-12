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
from encoders import TenPlaneEncoder, ThirteenPlaneEncoder
from rl.experience import ExperienceCollector, combine_experience, load_experience
from Board_v2 import CheckersGame
import pygame
import timeit
from rl.kerasutil import kerasutil_save_model_to_hdf5_group, kerasutil_load_model_from_hdf5_group
from rl.pg_agent import PolicyAgent, load_policy_agent
from rl.q_agent import QAgent, load_q_agent
# from networks.fullyconnected import layers
# from networks.small import layers
# from networks.medium import layers
from eval_pg_bot import eval
import importlib
import csv
from encoders import get_encoder_by_name

from keras.models import Model
from keras.layers import Conv2D, Dense, Flatten, Input
from keras.layers import ZeroPadding2D, concatenate

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

def create_model(input_shape=(15, 8, 8)):  # Изменяем порядок размерностей
    """Создаёт модель нейронной сети для агента"""
    model = Sequential()
    for layer in layers_module.layers(input_shape):
        model.add(layer)
    model.add(Dense(1, activation='linear'))  # Выход - скор
    return model

def create_model_q_training(input_shape=(13, 8, 8)):  # Изменяем порядок размерностей
    """Создаёт модель нейронной сети для агента"""
    board_input = Input(shape=input_shape, name='board_input')
    action_input = Input(shape=input_shape, name='action_input')

    conv1a=ZeroPadding2D(padding=3)(board_input)
    conv1b=Conv2D(48, (7, 7), activation='relu')(conv1a)
    conv2a = ZeroPadding2D((2, 2))(conv1b)
    conv2b = Conv2D(32, (5, 5), activation='relu')(conv2a)
    conv3a = ZeroPadding2D((2, 2))(conv2b)
    conv3b = Conv2D(32, (5, 5), activation='relu')(conv3a)
    conv4a = ZeroPadding2D((2, 2))(conv3b)
    conv4b = Conv2D(32, (5, 5), activation='relu')(conv4a)
    flat = Flatten()(conv4b)
    processed_board = Dense(512, activation='relu')(flat)

    action_conv1a=ZeroPadding2D(padding=3)(action_input)
    action_conv1b=Conv2D(48, (7, 7), activation='relu')(action_conv1a)
    action_conv2a = ZeroPadding2D((2, 2))(action_conv1b)
    action_conv2b = Conv2D(32, (5, 5), activation='relu')(action_conv2a)
    action_conv3a = ZeroPadding2D((2, 2))(action_conv2b)
    action_conv3b = Conv2D(32, (5, 5), activation='relu')(action_conv3a)
    action_conv4a = ZeroPadding2D((2, 2))(action_conv3b)
    action_conv4b = Conv2D(32, (5, 5), activation='relu')(action_conv4a)
    action_flat = Flatten()(action_conv4b)
    action_processed_board = Dense(512, activation='relu')(action_flat)
    board_and_action = concatenate([action_processed_board, processed_board])
    hidden_layer = Dense(256, activation='relu')(board_and_action)
    value_output = Dense(1, activation='linear')(hidden_layer)
    model = Model(inputs=[board_input, action_input], outputs=value_output)
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
    global time_list
    color1 = 1  # Начинаем с белых
    start = timeit.default_timer()
    for i in range(num_games):
        if i%10==0 and i>0:
            print(f'Симуляция игры {i + 1}/{num_games}...')
            print("time spent :", timeit.default_timer() - start)
            time_list.append(timeit.default_timer() - start)
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

        collector1_advantage = []
        collector2_advantage = []

        #Вычисляем advantage
        for i in range(len(collector1._current_episode_action_results)-1):
            collector1_advantage.append(agent1._encoder.score(collector1._current_episode_action_results[i+1])-agent1._encoder.score(collector1._current_episode_action_results[i]))
        collector1_advantage.append(0)

        for i in range(len(collector2._current_episode_action_results)-1):
            collector2_advantage.append(agent2._encoder.score(collector2._current_episode_action_results[i+1])-agent2._encoder.score(collector2._current_episode_action_results[i]))
        collector2_advantage.append(0)

        # Определяем вознаграждения на основе результата игры
        if game_record.winner == color1:  # Победа белых
            collector1.complete_episode(reward=1, advantages=collector1_advantage)
            collector2.complete_episode(reward=-1, advantages=collector2_advantage)
        elif game_record.winner == -color1:  # Победа черных
            collector2.complete_episode(reward=1, advantages=collector2_advantage)
            collector1.complete_episode(reward=-1, advantages=collector1_advantage)
        else:  # Ничья
            collector1.complete_episode(reward=0, advantages=collector1_advantage)
            collector2.complete_episode(reward=0, advantages=collector2_advantage)

        # Меняем цвета для следующей игры
        color1 = 0 - color1

        # Периодически сохраняем опыт
        # if i % 100 == 0 and i > 0:
        #     experience = combine_experience([collector1, collector2])
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


def train_agent(agent_filename, experience_filename, learning_rate=0.01, batch_size=512, epochs=1, loss='mse'):
    """Обучает агента на основе опыта"""

    # Загружаем агента
    with h5py.File(agent_filename, 'r') as agent_file:
        agent = load_policy_agent(agent_file)

    # Загружаем опыт
    with h5py.File(experience_filename, 'r') as experience_file:
        experience = load_experience(experience_file)

    # Обучаем агента
    agent.train(experience, lr=learning_rate, batch_size=batch_size, epochs=epochs, loss=loss)

    # # Сохраняем обновленного агента
    # with h5py.File(agent_filename, 'w') as agent_file:
    #     agent.serialize(agent_file)

    return agent

def train_q_agent(agent_filename, experience_filename, learning_rate=0.01, batch_size=512, epochs=1, loss='mse'):
    """Обучает агента на основе опыта"""

    # Загружаем агента
    with h5py.File(agent_filename, 'r') as agent_file:
        agent = load_q_agent(agent_file)

    # Загружаем опыт
    with h5py.File(experience_filename, 'r') as experience_file:
        experience = load_experience(experience_file)

    # Обучаем агента
    agent.train(experience, lr=learning_rate, batch_size=batch_size, epochs=epochs, loss=loss)

    # # Сохраняем обновленного агента
    # with h5py.File(agent_filename, 'w') as agent_file:
    #     agent.serialize(agent_file)

    return agent

def reinforce(agent_filename, out_experience_filename, num_games, num_iterations, prev_experience_filename=None, learning_rate=0.01, batch_size=512, epochs=1, temperature=0.05):

    for j in range(num_iterations):
        print('starting {} iteration'.format(j+1))
        do_self_play(agent_filename=agent_filename, num_games=num_games, temperature=temperature,
                         experience_filename='models_n_exp/experience_checkers_reinforce_{}_iter'.format(j))
        trained_agent = train_agent(
            agent_filename=agent_filename,
            experience_filename='models_n_exp/experience_checkers_reinforce_{}_iter.hdf5'.format(j),
            learning_rate=learning_rate,
            batch_size=batch_size,
            epochs=epochs
        )
        with h5py.File(agent_filename, 'w') as model_outf:
            trained_agent.serialize(model_outf)

    exp_list = []

    if prev_experience_filename:
        exp_list.append(load_experience(h5py.File(prev_experience_filename)))

    for k in range(num_iterations):
        exp_list.append(load_experience(h5py.File('models_n_exp/experience_checkers_reinforce_{}_iter.hdf5'.format(k))))

    total_exp = combine_experience(exp_list)

    with h5py.File(out_experience_filename, 'w') as experience_outf:
        total_exp.serialize(experience_outf)

    for z in range(num_iterations):
        os.remove('models_n_exp/experience_checkers_reinforce_{}_iter.hdf5'.format(z))


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
    # trained_agent=train_agent(
    #     agent_filename='models_n_exp/test_model_small.hdf5',
    #     experience_filename='models_n_exp/experience_checkers_all_iters_one_plane.hdf5',
    #     learning_rate=0.01,
    #     batch_size=128,
    #     epochs=1
    # )
    # with h5py.File('models_n_exp/test_model_small_trained.hdf5', 'w') as model_outf:
    #     trained_agent.serialize(model_outf)

    model_name = 'small'
    encoder_name = 'fiveteenplane'
    lr = 0.06
    batch_size = 512
    q=False
    loss='mse'
    if q:
        q_str = 'q_'
    else:
        q_str = ''
    time_list = []
    # pool_size = load_experience(h5py.File(
    #     'models_n_exp/experience_checkers_all_iters_thirteen_plane_insubjective_w_advantages_w_reinforce.hdf5')).states.shape[0]
    layers_module = importlib.import_module('networks.' + model_name)

    model=create_model()
    # model=create_model_q_training()
    encoder = get_encoder_by_name(encoder_name)
    new_agent = PolicyAgent(model, encoder)
    # new_agent = QAgent(model, encoder)

    test_agent_filename = 'models_n_exp/test_model_{}_{}_test.hdf5'.format(model_name, encoder_name)
    blank_agent_filename = 'models_n_exp/test_model_{}_{}_cf.hdf5'.format(model_name, encoder_name)
    blank_agent_to_train_filename = 'models_n_exp/test_model_{}_{}_to_train_cf.hdf5'.format(model_name, encoder_name)
    reinforcement_exp_filename = 'models_n_exp/experience_checkers_reinforce_all_iters_{}_{}_test_cf.hdf5'.format(model_name, encoder_name)
    exp_filename = 'models_n_exp/experience_checkers_reinforce_all_iters_fiveteenplane.hdf5'
    train_result_agent_filename = 'models_n_exp/test_model_{}_{}_trained.hdf5'.format(model_name, encoder_name)
    # 'models_n_exp/test_model_{}_{}_{}_{}_{}trained.hdf5'.format(model_name, encoder_name, lr, batch_size, q_str)
    with h5py.File(test_agent_filename, 'w') as model_outf:
        new_agent.serialize(model_outf)
    # with h5py.File(black_agent_to_train_filename, 'w') as model_outf:
    #     new_agent.serialize(model_outf)

    trained_agent=train_agent(
        agent_filename=test_agent_filename,
        experience_filename=exp_filename,
        # experience_checkers_all_iters_thirteen_plane_insubjective_w_advantages_w_reinforce
        # experience_filename='models_n_exp/experience_checkers_reinforce_all_iters.hdf5',
        learning_rate=lr,
        batch_size=batch_size,
        epochs=1,
        loss=loss
    )
    with h5py.File(train_result_agent_filename, 'w') as model_outf:
        trained_agent.serialize(model_outf)
    # reinforce(
    #     agent_filename=blank_agent_to_train_filename,
    #     out_experience_filename = reinforcement_exp_filename,
    #     prev_experience_filename = reinforcement_exp_filename,
    #     num_games=100,
    #     num_iterations=10,
    #     learning_rate=0.04,
    #     batch_size=512,
    #     epochs=1,
    #     temperature=0.05
    #     )

    # agent1 = load_policy_agent(h5py.File('models_n_exp/test_model_small_fiveteenplane_to_train_cf.hdf5'))

    print(time_list)
    res = eval(test_agent_filename,train_result_agent_filename, 100, q=q)
    print(res)
    # with open('training.log', 'r') as f:
    #     file = csv.reader(f)
    #     for row in file:
    #         loss_value = row[1]
    #
    # for i in [ lr, batch_size, pool_size, model_name, q, loss, loss_value]:
    #     res.append(i)
    #
    # prev_data = []
    # with open('results.csv', 'r') as f:
    #     csvFile = csv.reader(f)
    #     for i in csvFile:
    #         prev_data.append(i)
    #
    # with open('results.csv', 'w', newline='') as f:
    #     # csv.writer(f).writerow(fieldnames)
    #     csv.writer(f).writerows(prev_data)
    #     csv.writer(f).writerow(res)


    # ['models_n_exp/test_model_fullyconnected.hdf5', 'models_n_exp/test_model_fullyconnected_trained.hdf5', 59, 100]
    # ['models_n_exp/test_model_large.hdf5', 'models_n_exp/test_model_large_trained.hdf5', 56, 100]

    # small, mse, 0.02, 512
    # ['models_n_exp/test_model_small.hdf5', 'models_n_exp/test_model_small_trained.hdf5', 43, 50]
    # ['models_n_exp/test_model_small.hdf5', 'models_n_exp/test_model_small_trained.hdf5', 46, 50]
    # ['models_n_exp/test_model_small.hdf5', 'models_n_exp/test_model_small_trained.hdf5', 41, 50]


    # exp1 = load_experience(h5py.File('models_n_exp/experience_checkers_all_iters_thirteen_plane_insubjective_w_advantages.hdf5'))
    # exp2 = load_experience(h5py.File('models_n_exp/experience_checkers_reinforce_all_iters.hdf5'))
    # exp_list = [exp1, exp2]
    #
    # total_exp = combine_experience(exp_list)
    #
    # with h5py.File('models_n_exp/experience_checkers_all_iters_thirteen_plane_insubjective_w_advantages_w_reinforce.hdf5', 'w') as experience_outf:
    #     total_exp.serialize(experience_outf)


    # # with h5py.File('models_n_exp/test_model.hdf5', 'r') as agent_file:
    # #     bot = load_policy_agent(agent_file)
    # # res = simulate_game(bot,bot,1)
    # # print(res)

    # with h5py.File('models_n_exp/test_model_small.hdf5', 'r') as agent_file:
    #     agent = load_policy_agent(agent_file)
    #
    # game_record = simulate_game(agent, agent, game_num_for_record=0)

    # do_self_play(
    #     agent_filename='models_n_exp/test_model_small_trained.hdf5',
    #     num_games=1,
    #     temperature=0.01,
    #     experience_filename='experience_checkers'
    # )

    # fieldnames = ['model_1', 'model_2', 'wins', 'num_games','lr','batch_size','pool_size','nn_type','q']
    # res = ['models_n_exp/test_model_small.hdf5', 'models_n_exp/test_model_small_trained.hdf5', 130, 150, lr, batch_size, pool_size, model_name, q]
