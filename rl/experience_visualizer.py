from experience import load_experience
import h5py
import encoders

exp = load_experience(h5py.File('test_1000_games.hdf5'))
encoder = encoders.get_encoder_by_name('oneplane_singlesided')

new_list_w = []
new_list_b = []
new_list_rewards = []

for i in range(exp.game_nums.shape[0]):
    if exp.game_nums[i]==1 and exp.white_turns[i]==1:
        new_list_w.append(exp.states[i])
        new_list_rewards.append(exp.rewards[i])
    elif exp.game_nums[i]==1 and exp.white_turns[i]==0:
        new_list_b.append(exp.states[i])

encoder.show_board_from_matrix(new_list_w[-3], whites_turn=1)
print('---------')
encoder.show_board_from_matrix(new_list_b[-3], whites_turn=0)
print('---------')
encoder.show_board_from_matrix(new_list_w[-2], whites_turn=1)
print('---------')
encoder.show_board_from_matrix(new_list_b[-2], whites_turn=0)
print('---------')
encoder.show_board_from_matrix(new_list_w[-1], whites_turn=1)
print('---------')
encoder.show_board_from_matrix(new_list_b[-1], whites_turn=0)
print('---------')
print(new_list_rewards)
