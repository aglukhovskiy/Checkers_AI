import numpy as np

__all__ = [
    'ExperienceCollector',
    'ExperienceBuffer',
    'combine_experience',
    'load_experience',
]

class ExperienceCollector:
    def __init__(self):
        self.states = []
        # self.actions = []
        self.action_results = []
        self.rewards = []
        self.advantages = []
        self.white_turns = []
        self.game_nums = []
        self._current_episode_states = []
        # self._current_episode_actions = []
        self._current_episode_action_results = []
        self._current_episode_estimated_values = []
        self._current_episode_white_turns = []
        self._current_episode_game_nums = []

    def begin_episode(self):
        self._current_episode_states = []
        # self._current_episode_actions = []
        self._current_episode_action_results = []
        self._current_episode_estimated_values = []
        self._current_episode_white_turns = []
        self._current_episode_game_nums = []

    def record_decision(self, state, action_result, white_turns, game_nums, estimated_value=0):
        self._current_episode_states.append(state)
        # Преобразуем action_result в правильную размерность (8, 8, 6)
        # action_result = np.transpose(action_result, (1, 2, 0))
        self._current_episode_action_results.append(action_result)
        self._current_episode_estimated_values.append(estimated_value)
        self._current_episode_white_turns.append(white_turns)
        self._current_episode_game_nums.append(game_nums)

    def complete_episode(self, reward, advantages):
        num_states = len(self._current_episode_states)
        self.states += self._current_episode_states
        # self.actions += self._current_episode_actions
        self.action_results += self._current_episode_action_results
        self.white_turns += self._current_episode_white_turns
        self.game_nums += self._current_episode_game_nums
        self.rewards += [reward for _ in range(num_states)]
        self.advantages += advantages

        # for i in range(num_states):
        #     advantage = reward - self._current_episode_estimated_values[i]
        #     self.advantages.append(advantage)

        self._current_episode_states = []
        # self._current_episode_actions = []
        self._current_episode_action_results = []
        self._current_episode_estimated_values = []
        self._current_episode_white_turns = []
        self._current_episode_game_nums = []


class ExperienceBuffer:
    def __init__(self, states, action_results, rewards, advantages, white_turns, game_nums):
        self.states = states
        # self.actions = actions
        self.action_results = action_results
        self.rewards = rewards
        self.advantages = advantages
        self.white_turns = white_turns
        self.game_nums = game_nums

    def serialize(self, h5file):
        h5file.create_group('experience')
        h5file['experience'].create_dataset('states', data=self.states)
        # h5file['experience'].create_dataset('actions', data=self.actions)
        h5file['experience'].create_dataset('action_results', data=self.action_results)
        h5file['experience'].create_dataset('rewards', data=self.rewards)
        h5file['experience'].create_dataset('advantages', data=self.advantages)
        h5file['experience'].create_dataset('white_turns', data=self.white_turns)
        h5file['experience'].create_dataset('game_nums', data=self.game_nums)


def combine_experience(collectors):
    combined_states = np.concatenate([np.array(c.states) for c in collectors])
    # combined_actions = np.concatenate([np.array(c.actions) for c in collectors])
    combined_action_results = np.concatenate([np.array(c.action_results) for c in collectors])
    combined_rewards = np.concatenate([np.array(c.rewards) for c in collectors])
    combined_advantages = np.concatenate([
        np.array(c.advantages) for c in collectors])
    combined_white_turns = np.concatenate([np.array(c.white_turns) for c in collectors])
    combined_game_nums = np.concatenate([np.array(c.game_nums) for c in collectors])

    return ExperienceBuffer(
        combined_states,
        # combined_actions,
        combined_action_results,
        combined_rewards,
        combined_advantages,
        combined_white_turns,
        combined_game_nums
        )


def load_experience(h5file):
    return ExperienceBuffer(
        states=np.array(h5file['experience']['states']),
        # actions=np.array(h5file['experience']['actions']),
        action_results=np.array(h5file['experience']['action_results']),
        rewards=np.array(h5file['experience']['rewards']),
        advantages=np.array(h5file['experience']['advantages']),
        white_turns=np.array(h5file['experience']['white_turns']),
        game_nums=np.array(h5file['experience']['game_nums']))
