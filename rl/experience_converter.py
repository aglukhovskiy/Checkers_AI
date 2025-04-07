from experience import load_experience, ExperienceBuffer
import h5py
import encoders
import numpy as np

exp = load_experience(h5py.File('models_n_exp/experience_checkers_all_iters.hdf5'))
encoder = encoders.get_encoder_by_name('tenplane_v2')

converted  = encoder.ten_to_one_plane_matrix(exp.states[0])
converted_states_list = []
converted_action_results_list = []

for i in range(exp.states.shape[0]):
    converted_states_list.append(encoder.ten_to_one_plane_matrix(exp.states[i]))
    converted_action_results_list.append(encoder.ten_to_one_plane_matrix(exp.action_results[i]))

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

with h5py.File('models_n_exp/experience_checkers_all_iters_one_plane.hdf5', 'w') as experience_file:
    buffer.serialize(experience_file)

