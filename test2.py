from rl.experience import load_experience
import h5py
import encoders
import Board_v2

exp = load_experience(h5py.File('rl/models_n_exp/experience_checkers_reinforce_all_iters_fiveteenplane.hdf5'))
encoder = encoders.get_encoder_by_name('tenplane_v2')

new_list_w = []
new_list_b = []
new_list_w2 = []
new_list_b2 = []
new_list_rewards = []
new_list_rewards_b = []

for i in range(exp.game_nums.shape[0]):
    if exp.game_nums[i]==0 and exp.white_turns[i]==1:
        new_list_w.append(exp.action_results[i])
        new_list_w2.append(exp.states[i])
        new_list_rewards.append(exp.rewards[i])
    elif exp.game_nums[i]==0 and exp.white_turns[i]==-1:
        new_list_b.append(exp.action_results[i])
        new_list_b2.append(exp.states[i])
        new_list_rewards_b.append(exp.rewards[i])


encoder.show_board_from_matrix(new_list_w[0])
print('---------')
encoder.show_board_from_matrix(new_list_b[0])
print('---------')
encoder.show_board_from_matrix(new_list_w[1])
print('---------')
encoder.show_board_from_matrix(new_list_b[1])
print('---------')
encoder.show_board_from_matrix(new_list_w[2])
print('---------')
encoder.show_board_from_matrix(new_list_b[2])
print('---------')

print(new_list_b[0])

# game = Board_v2.CheckersGame()
# av_moves = game.get_possible_moves()
# print(av_moves)
# for move in av_moves[0]:
#     game.move_piece(move)
#
# av_moves = game.get_possible_moves()
# print(av_moves)
