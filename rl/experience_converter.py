from experience import load_experience, ExperienceBuffer
import h5py
import encoders
import numpy as np

# ### FROM 10 PLANE TO INSUBJECTIVE REWARD 10 PLANE
# exp = load_experience(h5py.File('models_n_exp/experience_checkers_all_iters.hdf5'))
# encoder = encoders.get_encoder_by_name('tenplane_v2')
#
# converted_rewards_array = np.zeros(exp.states.shape[0])
# converted_advantages_array = np.zeros(exp.states.shape[0])
# converted_white_turns_array = np.zeros(exp.states.shape[0])
#
# for i in range(exp.states.shape[0]):
#     if exp.white_turns[i]==1:
#         converted_rewards_array[i]= exp.rewards[i]
#         converted_advantages_array[i]= exp.advantages[i]
#         converted_white_turns_array[i] = 1
#     else:
#         if abs(exp.rewards[i])==1:
#             converted_rewards_array[i]= -exp.rewards[i]
#         elif exp.rewards[i]==0:
#             converted_rewards_array[i] = exp.rewards[i]
#
#         if abs(exp.advantages[i])==1:
#             converted_advantages_array[i]= -exp.advantages[i]
#         elif exp.advantages[i]==0:
#             converted_advantages_array[i] = exp.advantages[i]
#         converted_white_turns_array[i]=-1
#
# buffer = ExperienceBuffer(
#         exp.states,
#         exp.action_results,
#         converted_rewards_array,
#         converted_advantages_array,
#         converted_white_turns_array,
#         exp.game_nums
#         )
#
# with h5py.File('models_n_exp/experience_checkers_all_iters_ten_plane_insubjective.hdf5', 'w') as experience_file:
#     buffer.serialize(experience_file)

### FROM 10 PLANE TO 1 PLANE
# exp = load_experience(h5py.File('models_n_exp/experience_checkers_all_iters.hdf5'))
# encoder = encoders.get_encoder_by_name('tenplane_v2')
#
# converted  = encoder.ten_to_one_plane_matrix(exp.states[0])
# converted_states_list = []
# converted_action_results_list = []
#
# for i in range(exp.states.shape[0]):
#     converted_states_list.append(encoder.ten_to_one_plane_matrix(exp.states[i]))
#     converted_action_results_list.append(encoder.ten_to_one_plane_matrix(exp.action_results[i]))
#
# # print(converted_states_list)
# # print('---------')
# # print(np.array(converted_states_list))
#
# buffer = ExperienceBuffer(
#         np.array(converted_states_list),
#         np.array(converted_action_results_list),
#         exp.rewards,
#         exp.advantages,
#         exp.white_turns,
#         exp.game_nums
#         )
#
# with h5py.File('models_n_exp/experience_checkers_all_iters_one_plane.hdf5', 'w') as experience_file:
#     buffer.serialize(experience_file)


# ## FROM 10 PLANE TO 5 PLANE
# exp = load_experience(h5py.File('models_n_exp/experience_checkers_all_iters_ten_plane_insubjective.hdf5'))
# encoder = encoders.get_encoder_by_name('tenplane_v2')
#
# converted_states_list = []
# converted_action_results_list = []
#
# for i in range(exp.states.shape[0]):
#     converted_states_list.append(encoder.ten_to_five_plane_matrix(exp.states[i]))
#     converted_action_results_list.append(encoder.ten_to_five_plane_matrix(exp.action_results[i]))
#
# # print(converted_states_list)
# # print('---------')
# # print(np.array(converted_states_list))
#
# buffer = ExperienceBuffer(
#         np.array(converted_states_list),
#         np.array(converted_action_results_list),
#         exp.rewards,
#         exp.advantages,
#         exp.white_turns,
#         exp.game_nums
#         )
#
# with h5py.File('models_n_exp/experience_checkers_all_iters_five_plane_insubjective.hdf5', 'w') as experience_file:
#     buffer.serialize(experience_file)

# ### FROM 1 PLANE TO INSUBJECTIVE REWARD 1 PLANE
# exp = load_experience(h5py.File('models_n_exp/experience_checkers_all_iters_one_plane.hdf5'))
# encoder = encoders.get_encoder_by_name('tenplane_v2')
#
# converted_rewards_array = np.zeros(exp.states.shape[0])
# converted_advantages_array = np.zeros(exp.states.shape[0])
#
# for i in range(exp.states.shape[0]):
#     if exp.white_turns[i]==1:
#         converted_rewards_array[i]= exp.rewards[i]
#         converted_advantages_array[i]= exp.advantages[i]
#     else:
#         if abs(exp.rewards[i])==1:
#             converted_rewards_array[i]= -exp.rewards[i]
#         elif exp.rewards[i]==0:
#             converted_rewards_array[i] = exp.rewards[i]
#
#         if abs(exp.advantages[i])==1:
#             converted_advantages_array[i]= -exp.advantages[i]
#         elif exp.advantages[i]==0:
#             converted_advantages_array[i] = exp.advantages[i]
#
# buffer = ExperienceBuffer(
#         exp.states,
#         exp.action_results,
#         converted_rewards_array,
#         converted_advantages_array,
#         exp.white_turns,
#         exp.game_nums
#         )
#
# with h5py.File('models_n_exp/experience_checkers_all_iters_one_plane_insubjective.hdf5', 'w') as experience_file:
#     buffer.serialize(experience_file)

# ### FROM 1 PLANE TO 2 PLANE
# exp = load_experience(h5py.File('models_n_exp/experience_checkers_all_iters_one_plane_insubjective.hdf5'))
# encoder = encoders.get_encoder_by_name('tenplane_v2')
#
# converted_states_list = []
# converted_action_results_list = []
#
# for i in range(exp.states.shape[0]):
#     converted_states_list.append(encoder.one_to_two_plane_matrix(exp.states[i]))
#     converted_action_results_list.append(encoder.one_to_two_plane_matrix(exp.action_results[i]))
#
# # print(converted_states_list)
# # print('---------')
# # print(np.array(converted_states_list))
#
# buffer = ExperienceBuffer(
#         np.array(converted_states_list),
#         np.array(converted_action_results_list),
#         exp.rewards,
#         exp.advantages,
#         exp.white_turns,
#         exp.game_nums
#         )
#
# with h5py.File('models_n_exp/experience_checkers_all_iters_two_plane_insubjective.hdf5', 'w') as experience_file:
#     buffer.serialize(experience_file)
#
# ### FROM 1 PLANE TO 4 PLANE
# exp = load_experience(h5py.File('models_n_exp/experience_checkers_all_iters_one_plane_insubjective.hdf5'))
# encoder = encoders.get_encoder_by_name('tenplane_v2')
#
# converted_states_list = []
# converted_action_results_list = []
#
# for i in range(exp.states.shape[0]):
#     converted_states_list.append(encoder.one_to_four_plane_matrix(exp.states[i]))
#     converted_action_results_list.append(encoder.one_to_four_plane_matrix(exp.action_results[i]))
#
# # print(converted_states_list)
# # print('---------')
# # print(np.array(converted_states_list))
#
# buffer = ExperienceBuffer(
#         np.array(converted_states_list),
#         np.array(converted_action_results_list),
#         exp.rewards,
#         exp.advantages,
#         exp.white_turns,
#         exp.game_nums
#         )
#
# with h5py.File('models_n_exp/experience_checkers_all_iters_four_plane_insubjective.hdf5', 'w') as experience_file:
#     buffer.serialize(experience_file)


# ## Advantage setting
# exp = load_experience(h5py.File('models_n_exp/experience_checkers_all_iters_ten_plane_insubjective.hdf5'))
# encoder = encoders.get_encoder_by_name('tenplane_v2')
#
# converted_advantages_array = np.zeros(exp.states.shape[0])
#
# for i in range(exp.states.shape[0]-1):
#     if exp.game_nums[i]==exp.game_nums[i+1] and exp.white_turns[i]==exp.white_turns[i+1]:
#         converted_advantages_array[i] = encoder.score(exp.action_results[i+1])-encoder.score(exp.action_results[i])
#
# buffer = ExperienceBuffer(
#         exp.states,
#         exp.action_results,
#         exp.rewards,
#         converted_advantages_array,
#         exp.white_turns,
#         exp.game_nums
#         )
#
# with h5py.File('models_n_exp/experience_checkers_all_iters_ten_plane_insubjective_w_advantages.hdf5', 'w') as experience_file:
#     buffer.serialize(experience_file)

## FROM 10 PLANE TO 5 PLANE
exp = load_experience(h5py.File('models_n_exp/experience_checkers_all_iters_ten_plane_insubjective_w_advantages.hdf5'))
encoder = encoders.get_encoder_by_name('thirteenplane')

converted_states_list = []
converted_action_results_list = []

for i in range(exp.states.shape[0]):
    converted_states_list.append(encoder.ten_to_thirteen_plane_matrix(exp.states[i]))
    converted_action_results_list.append(encoder.ten_to_thirteen_plane_matrix(exp.action_results[i]))

# print(converted_states_list)
# print('---------')
# print(np.array(converted_states_list))

buffer = ExperienceBuffer(
        np.array(converted_states_list),
        np.array(converted_action_results_list),
        exp.rewards,
        exp.advantages,
        exp.white_turns,
        exp.game_nums
        )

with h5py.File('models_n_exp/experience_checkers_all_iters_thirteen_plane_insubjective_w_advantages.hdf5', 'w') as experience_file:
    buffer.serialize(experience_file)
