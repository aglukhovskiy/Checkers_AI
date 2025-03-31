from pg import PolicyAgent, load_policy_agent
import h5py
import random
import time
import os
import numpy as np
# from encoders.oneplane import OnePlaneEncoder
import encoders
from keras.models import Sequential
from keras.layers import Dense
from keras import metrics
from keras.layers.core import Dense, Activation, Flatten
from keras.layers.convolutional import Conv2D, ZeroPadding2D
from Checkers import Checkers
import Board
from collections import namedtuple
import tensorflow as tf

from rl import large
from rl.experience import ExperienceCollector, combine_experience, load_experience

# random.seed(42)

input_shape = (8,8,6)

layers = [
        ZeroPadding2D((3, 3), input_shape=input_shape),
        Conv2D(64, (7, 7), padding='valid'),
        Activation('relu'),
        Flatten(),
        Dense(128),
        Activation('relu'),
        ]

def create_mlp():
    model = Sequential()
    for layer in layers:
        model.add(layer)
    # model.add(Dense(16, input_shape=input_shape, activation="relu"))
    # model.add(Dense(4, activation="relu"))
    model.add(Dense(1, activation="linear"))
    return model

encoder = encoders.get_encoder_by_name('sixplane')

# model = Sequential()
# # opt = Adam(lr=1e-3, decay=1e-3 / 200)
# for layer in large.layers(encoder.shape()):
#     model.add(layer)
# model.add(Dense(1), activation="linear")
# # model.add(Dense(encoder.num_points()))
# # model.add(Activation('softmax'))

model = create_mlp()
# model.compile(loss="mean_absolute_error", optimizer='SGD', metrics=[metrics.MeanAbsoluteError()])
model.compile(loss="mse", optimizer='SGD', metrics=[metrics.BinaryCrossentropy()])

new_agent1 = PolicyAgent(model = model, encoder = encoder)
new_agent2 = PolicyAgent(model = model, encoder = encoder)
collector1 = ExperienceCollector()
collector2 = ExperienceCollector()
new_agent1.set_collector(collector1)
new_agent2.set_collector(collector2)

# agent1 = load_policy_agent(h5py.File('model_test.hdf5'))
# agent2 = load_policy_agent(h5py.File('model_test.hdf5'))
# collector1 = ExperienceCollector()
# collector2 = ExperienceCollector()
# agent1.set_collector(collector1)
# agent2.set_collector(collector2)

# print(new_agent1)

class GameRecord(namedtuple('GameRecord', 'moves winner margin')):
    pass

def simulate_game(white_player, black_player, game_num_for_record):
    moves = []
    board = Board.Field()
    game = Checkers(opp='simulation_bot', control='simulation', board = board)

    agents = {
        1: white_player,
        0: black_player,
    }
    moves_cntr = 0
    while game.board.game_is_on == 1 and moves_cntr<80:
        moves_cntr+=1
        # if moves_cntr%10==0:
        #     print(moves_cntr)
        next_move = agents[game.board.whites_turn].select_move(game, game_num_for_record)
        moves.append(next_move)
        game.next_turn(next_move)
        # encoder.show_board(game.board)
        if game.board.game_is_on==0:
            break

    encoder.show_board(game.board)
    game_result, game_margin = game.board.compute_results()

    return GameRecord(
        moves=moves,
        winner=game_result,
        margin=game_margin,
    )

# res = simulate_game(new_agent1, new_agent2)
# print('RES')
# print(res)

# collector1.complete_episode(reward=1)
# print(collector1)

# experience = combine_experience([collector1])

# print(experience)
# print(experience.states)

# with h5py.File('test.hdf5', 'w') as experience_outf:
#     experience.serialize(experience_outf)

with h5py.File('model_test_6_plane.hdf5', 'w') as model_outf:
    new_agent1.serialize(model_outf)

# agent1 = load_policy_agent(h5py.File('model_test.hdf5'))
# agent2 = load_policy_agent(h5py.File('model_test.hdf5'))
# collector1 = ExperienceCollector()
# collector2 = ExperienceCollector()
# agent1.set_collector(collector1)
# agent2.set_collector(collector2)

def do_self_play(agent_filename,
                 num_games, temperature,
                 experience_filename,
                 gpu_frac):
    # kerasutil.set_gpu_memory_target(gpu_frac)

    random.seed(int(time.time()) + os.getpid())
    np.random.seed(int(time.time()) + os.getpid())

    agent1 = load_policy_agent(h5py.File(agent_filename))
    # agent1.set_temperature(temperature)
    agent2 = load_policy_agent(h5py.File(agent_filename))
    # agent2.set_temperature(temperature)

    collector1 = ExperienceCollector()
    collector2 = ExperienceCollector()

    color1 = 1
    for i in range(num_games):
        print('Simulating game %d/%d...' % (i + 1, num_games))
        collector1.begin_episode()
        agent1.set_collector(collector1)
        collector2.begin_episode()
        agent2.set_collector(collector2)

        if color1 == 1:
            white_player, black_player = agent1, agent2
        else:
            black_player, white_player = agent1, agent2
        game_record = simulate_game(white_player, black_player, game_num_for_record=i)
        # print('game was simulated')
        if game_record.winner == 1:
            print('Agent 1 wins. (whites) ')
            # collector1.complete_episode(reward=1)
            # collector2.complete_episode(reward=0)
            collector1.complete_episode(reward=1)
            collector2.complete_episode(reward=1)
        elif game_record.winner == 0:
            print('Agent 2 wins. (blacks) ')
            # collector2.complete_episode(reward=1)
            # collector1.complete_episode(reward=0)
            collector2.complete_episode(reward=0)
            collector1.complete_episode(reward=0)
        elif game_record.winner == 0.5:
            print('Draw')
            collector2.complete_episode(reward=0.5)
            collector1.complete_episode(reward=0.5)
        color1 = 1-color1
        if i%100==0:
            experience = combine_experience([collector1, collector2])
            print('Saving experience buffer to %s\n' % experience_filename)
            with h5py.File(experience_filename+'_{}_games.hdf5'.format(num_games), 'w') as experience_outf:
                experience.serialize(experience_outf)
            print(i)

    experience = combine_experience([collector1, collector2])
    print('Saving experience buffer to %s\n' % experience_filename)
    with h5py.File(experience_filename+'_{}_games.hdf5'.format(num_games), 'w') as experience_outf:
        experience.serialize(experience_outf)

do_self_play(agent_filename='model_test_6_plane.hdf5', num_games=300, temperature=0,
                 experience_filename='test_6_plane',
                 gpu_frac=0)



# new_agent1.train(load_experience(h5py.File('test.hdf5')))
# agent1.train(load_experience(h5py.File('test.hdf5')))

# with h5py.File('model_test_trained.hdf5', 'w') as model_outf:
#     new_agent1.serialize(model_outf)


# agent1 = load_policy_agent(h5py.File('model_test.hdf5'))
# agent2 = load_policy_agent(h5py.File('model_test_trained.hdf5'))
# agent3 = load_policy_agent(h5py.File('model_test.hdf5'))

# agent3.train(load_experience(h5py.File('test.hdf5')))
# agent1.train(load_experience(h5py.File('test.hdf5')))

# wins = 0
# losses = 0
# color1 = 1
# num_games = 25
#
# for i in range(num_games):
#     print('Simulating game %d/%d...' % (i + 1, num_games))
#     # if color1 == 1:
#     #     white_player, black_player = agent1, agent2
#     # else:
#     #     black_player, white_player = agent1, agent2
#     black_player, white_player = agent1, agent3
#     game_record = simulate_game(white_player, black_player)
#     if game_record.winner == color1:
#         wins += 1
#     else:
#         losses += 1
#     color1 = 1-color1
#     print('winner - ', game_record.winner)
#
# print('Agent 1 record: %d/%d' % (wins, wins + losses))

