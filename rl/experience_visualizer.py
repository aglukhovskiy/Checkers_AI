from experience import load_experience
import h5py
import encoders

exp = load_experience(h5py.File('models_n_exp/experience_checkers_all_iters_one_plane.hdf5'))
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
    elif exp.game_nums[i]==0 and exp.white_turns[i]==0:
        new_list_b.append(exp.action_results[i])
        new_list_b2.append(exp.states[i])
        new_list_rewards_b.append(exp.rewards[i])

print(new_list_w2[0])
print('---------')
print(new_list_w[0])
print('---------')
print(new_list_b2[0])
print('---------')
print(new_list_b[0])
# print('---------')

# encoder.show_board_from_matrix(new_list_w2[0])
# print('---------')
# encoder.show_board_from_matrix(new_list_w[0])
# print('---------')
# encoder.show_board_from_matrix(new_list_b2[0])
# print('---------')
# encoder.show_board_from_matrix(new_list_b[0])
# print('---------')
# encoder.show_board_from_matrix(new_list_w2[1])
# print('---------')
# encoder.show_board_from_matrix(new_list_w[1])
# print('---------')
# encoder.show_board_from_matrix(new_list_b2[1])
# print('---------')
# encoder.show_board_from_matrix(new_list_b[1])
# print('---------')
# encoder.show_board_from_matrix(new_list_w[2])
# print('---------')

# print(exp.rewards)
# print(exp.game_nums.shape)
# print(new_list_rewards)
# print(new_list_rewards_b)
# print(exp.states[-1])
# print(exp.action_results[-1])

cntr = {0: 0, 1:0, -1:0}
for i in range(exp.rewards.shape[0]):
    cntr[exp.rewards[i]]+=1
print(cntr)
print(exp.rewards[:100])
