from experience import load_experience, combine_experience
import h5py
# from rl.pg_agent import PolicyAgent, load_policy_agent
from rl.q_agent import QAgent, load_q_agent
import numpy as np
import encoders
from Board_v2 import CheckersGame
from scipy.stats import binomtest

# encoder = encoders.get_encoder_by_name('oneplane_singlesided')
encoder = encoders.get_encoder_by_name('thirteenplane')

# exp = load_experience(h5py.File('test_6_plane_1000_games.hdf5'))

# agent1 = load_policy_agent(h5py.File('models_n_exp/test_model_small.hdf5'))
# # agent1.train(load_experience(h5py.File('models_n_exp/test_10_plane_500_games.hdf5')), lr=0.02)
# agent2 = load_policy_agent(h5py.File('models_n_exp/test_model_small_trained.hdf5'))

agent1 = load_q_agent(h5py.File('models_n_exp/test_q_model_small.hdf5'))
agent2 = load_q_agent(h5py.File('models_n_exp/test_q_model_small.hdf5'))

print(agent2._encoder)

# with h5py.File('model_test_trained.hdf5', 'w') as model_outf:
#     new_agent1.serialize(model_outf)


def simulate_game(white_player, black_player, game_num_for_record):
    moves = []
    game = CheckersGame()

    agents = {
        1: white_player,
        -1: black_player,
    }
    moves_cntr = 0
    while game.game_is_on == 1 and moves_cntr<80:
        moves_cntr+=1
        # if moves_cntr%10==0:
        #     print(moves_cntr)
        next_move = agents[game.current_player].select_move(game, game_num_for_record)
        moves.append(next_move)
        game.next_turn(next_move)
        # encoder.show_board(game)
        if game.game_is_on==0:
            break

    encoder.show_board(game)
    game_result, game_margin = game.compute_results()

    return game_result

wins = 0
losses = 0
color1 = 1
num_games = 20

for i in range(num_games):
    print('Simulating game %d/%d...' % (i + 1, num_games))
    # if color1 == 1:
    #     white_player, black_player = agent1, agent2
    # else:
    #     black_player, white_player = agent1, agent2
    white_player, black_player  = agent2, agent1
    game_record = simulate_game(white_player, black_player, i)
    if game_record == color1:
        wins += 1
    else:
        losses += 1
    # color1 = 1-color1
    print('winner - ', game_record)
    print('Agent 1 record: %d/%d' % (wins, wins + losses))

print('Agent 1 record: %d/%d' % (wins, wins + losses))


# 12/20 trained - white
# 8/20 trained - black lr - 0.01, batch - 32, epochs = 3, large

# print(binomtest(34, 100, 0.5))