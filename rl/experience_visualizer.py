from experience import load_experience
import h5py
import encoders

exp = load_experience(h5py.File('models_n_exp/experience_checkers_reinforce_all_iters_small_fiveteenplane_test_2.hdf5'))
# experience_filename='models_n_exp/experience_checkers_all_iters_thirteen_plane_insubjective_w_advantages.hdf5',
encoder = encoders.get_encoder_by_name('thirteenplane')

# exp1 = load_experience(h5py.File('models_n_exp/experience_checkers_all_iters_thirteen_plane_insubjective_w_advantages.hdf5'))
# exp2 = load_experience(h5py.File('models_n_exp/experience_checkers_reinforce_all_iters.hdf5'))

# (122298, 13, 8, 8)
# {0.0: 66273, -1.0: 19638, 1.0: 19597, 3.0: 1932, 2.0: 4131, -4.0: 1421, 6.0: 105, -5.0: 489, -2.0: 4378, 4.0: 1493, 5.0: 522, -3.0: 2137, -6.0: 114, -7.0: 27, 7.0: 20, 8.0: 9, -8.0: 8, 9.0: 4}

print(exp.states.shape)
print(exp.states[0])
cntr = {}
for i in range(exp.advantages.shape[0]):
    if exp.advantages[i] not in cntr:
        cntr[exp.advantages[i]]=0
    cntr[exp.advantages[i]]+=1
print(cntr)
print(exp.advantages[:100])


# n = experience.action_results.shape[0]
# # Translate the actions/rewards.
# y = np.zeros(n)
# for i in range(n):
#     reward = experience.rewards[i]
#     y[i] = reward
#
# # Данные уже в формате (None, 10, 8, 8)
# x = experience.action_results
#
# self._model.fit(
#     x=x, batch_size=batch_size, y=y, epochs=epochs)
# x = exp.action_results
# print(exp.action_results[0])
# print(x[:10].shape)
# print('---------')
# print('---------')

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

# print(new_list_w2[0])
# print('---------')
# print(new_list_w[0])
# print('---------')
# print(new_list_b2[0])
# print('---------')
# print(new_list_b[0])
# print('---------')

# encoder.show_board_from_matrix(new_list_w2[3])
# print('---------')
encoder.show_board_from_matrix(new_list_w[3])
print('---------')
# encoder.show_board_from_matrix(new_list_b2[0])
# print('---------')
encoder.show_board_from_matrix(new_list_b[3])
print('---------')
# encoder.show_board_from_matrix(new_list_w2[1])
# print('---------')
encoder.show_board_from_matrix(new_list_w[4])
print('---------')
# encoder.show_board_from_matrix(new_list_b2[1])
# print('---------')
encoder.show_board_from_matrix(new_list_b[4])
print('---------')
encoder.show_board_from_matrix(new_list_w[5])
print('---------')
encoder.show_board_from_matrix(new_list_b[5])
print('---------')
encoder.show_board_from_matrix(new_list_w[6])
print('---------')
encoder.show_board_from_matrix(new_list_b[6])
print('---------')

# print(exp.rewards)
# print(exp.game_nums.shape)
# print(new_list_rewards)
# print(new_list_rewards_b)
# print(exp.states[-1])
# print(exp.action_results[-1])
