from pg import PolicyAgent, load_policy_agent
import h5py
import random
import time
import os
import numpy as np
from encoders.oneplane import OnePlaneEncoder
from keras.models import Sequential
from keras.layers import Dense
from keras.layers.core import Dense, Activation, Flatten
from keras.layers.convolutional import Conv2D, ZeroPadding2D
from Checkers import Checkers
import Board
from collections import namedtuple

from rl import large
from rl.experience import ExperienceCollector, combine_experience

input_shape = (8,8,1)

layers = [
        # ZeroPadding2D((3, 3), input_shape=input_shape, data_format='channels_first'),
        # Conv2D(64, (7, 7), padding='valid', data_format='channels_first'),
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

encoder = OnePlaneEncoder()
# model = Sequential()
# # opt = Adam(lr=1e-3, decay=1e-3 / 200)
# for layer in large.layers(encoder.shape()):
#     model.add(layer)
# model.add(Dense(1), activation="linear")
# # model.add(Dense(encoder.num_points()))
# # model.add(Activation('softmax'))
model = create_mlp()
model.compile(loss="mean_absolute_error", optimizer='SGD')

model.summary()

new_agent1 = PolicyAgent(model = model, encoder = encoder)
new_agent2 = PolicyAgent(model = model, encoder = encoder)
collector1 = ExperienceCollector()
collector2 = ExperienceCollector()
new_agent1.set_collector(collector1)
new_agent2.set_collector(collector2)

# agent1 = load_policy_agent(h5py.File(agent_filename))
# agent2 = load_policy_agent(h5py.File(agent_filename))
# collector1 = ExperienceCollector()
# collector2 = ExperienceCollector()
# agent1.set_collector(collector1)
# agent2.set_collector(collector2)

# print(new_agent1)

class GameRecord(namedtuple('GameRecord', 'moves winner margin')):
    pass

def simulate_game(white_player, black_player):
    moves = []
    board = Board.Field()
    game = Checkers(opp='simulation_bot', control='simulation', board = board)

    agents = {
        1: white_player,
        0: black_player,
    }
    moves_cntr = 0
    while game.board.game_is_on == 1 and moves_cntr<3:
        moves_cntr+=1
        next_move = agents[game.board.whites_turn].select_move(game)
        moves.append(next_move)
        game.next_turn(next_move)
        encoder.show_board(game.board)
        if game.board.game_is_on==0:
            break

    encoder.show_board(game.board)
    game_result, game_margin = game.board.compute_results()
    print(game_result)

    return GameRecord(
        moves=moves,
        winner=game_result,
        margin=game_margin,
    )

res = simulate_game(new_agent1, new_agent2)
print('RES')
print(res)
# collector1.complete_episode(reward=1)
# print(collector1)

# experience = combine_experience([collector1])

# print(experience)
# print(experience.states)
# print(experience.action_results)
# print(experience.rewards)
# print(experience.advantages)

# with h5py.File('test.hdf5', 'w') as experience_outf:
#     experience.serialize(experience_outf)

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

    color1 = Player.black
    for i in range(num_games):
        print('Simulating game %d/%d...' % (i + 1, num_games))
        collector1.begin_episode()
        agent1.set_collector(collector1)
        collector2.begin_episode()
        agent2.set_collector(collector2)

        if color1 == Player.black:
            black_player, white_player = agent1, agent2
        else:
            white_player, black_player = agent1, agent2
        game_record = simulate_game(black_player, white_player)
        if game_record.winner == color1:
            print('Agent 1 wins.')
            collector1.complete_episode(reward=1)
            collector2.complete_episode(reward=-1)
        else:
            print('Agent 2 wins.')
            collector2.complete_episode(reward=1)
            collector1.complete_episode(reward=-1)
        color1 = color1.other

    experience = combine_experience([collector1, collector2])
    print('Saving experience buffer to %s\n' % experience_filename)
    with h5py.File(experience_filename, 'w') as experience_outf:
        experience.serialize(experience_outf)


# do_self_play(agent_filename='test.hdf5', num_games=2, temperature=0,
#                  experience_filename=,
#                  gpu_frac=0)