"""Policy gradient learning."""
import numpy as np
from keras import backend as K
from copy import deepcopy
# from keras.optimizers import SGD

# from dlgo.agent.base import Agent
# from dlgo.agent.helpers import is_point_an_eye
# from dlgo import goboard
import encoders
from rl import kerasutil

# __all__ = [
#     'PolicyAgent',
#     'load_policy_agent',
#     'policy_gradient_loss',
# ]
#
#
# # Keeping this around so we can read existing agents. But from now on
# # we'll use the built-in crossentropy loss.
def policy_gradient_loss(y_true, y_pred):
    clip_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
    loss = -1 * y_true * K.log(clip_pred)
    return K.mean(K.sum(loss, axis=1))


# def normalize(x):
#     total = np.sum(x)
#     return x / total


class PolicyAgent():
    """An agent that uses a deep policy network to select moves."""
    def __init__(self, model, encoder):
        self._model = model
        self._encoder = encoder
        self._collector = None
        self._temperature = 0.0

#     def predict(self, game_state):
#         encoded_state = self._encoder.encode(game_state)
#         input_tensor = np.array([encoded_state])
#         return self._model.predict(input_tensor)[0]
#
#     def set_temperature(self, temperature):
#         self._temperature = temperature
#
    def set_collector(self, collector):
        self._collector = collector
#
    def select_move(self, game):
        moves = game.board.available_moves()[0]
        num_moves = len(moves)

        simulated_boards = []
        board_tensors = []
        x_list = []
        for i in range(num_moves):
            simulated_board = deepcopy(game)
            simulated_board.next_turn(move=moves[i])
            simulated_boards.append(simulated_board)
            board_tensor = self._encoder.encode(simulated_board.board)
            board_tensors.append(board_tensor)
            x = np.array([board_tensor])
            x_list.append(x)
        if np.random.random() < self._temperature:
            # Explore random moves.
            move_probs = np.ones(num_moves) / num_moves
        else:
            # Follow our current policy.
            move_probs = [self._model.predict(x[0])[0][0] for x in x_list]

        # Prevent move probs from getting stuck at 0 or 1.
        eps = 1e-5
        move_probs = np.clip(move_probs, eps, 1 - eps)
        # Re-normalize to get another probability distribution.
        move_probs = move_probs / np.sum(move_probs)

        # Turn the probabilities into a ranked list of moves.
        candidates = np.arange(num_moves)
        # try:
        ranked_moves = np.random.choice(
            candidates, num_moves, replace=False, p=move_probs)
        move = moves[ranked_moves[0]]
        # except:
        #     ranked_moves = []
        #     move = None

        if self._collector is not None:
            self._collector.record_decision(
                state=self._encoder.encode(game.board),
                action=move,
                action_result=board_tensors[ranked_moves[0]]
            )

        return move

    def serialize(self, h5file):
        h5file.create_group('encoder')
        h5file['encoder'].attrs['name'] = self._encoder.name()
        h5file['encoder'].attrs['board_width'] = self._encoder.board_width
        h5file['encoder'].attrs['board_height'] = self._encoder.board_height
        h5file.create_group('model')
        kerasutil.save_model_to_hdf5_group(self._model, h5file['model'])

    def train(self, experience, lr=0.0000001, clipnorm=1.0, batch_size=512):
        opt = SGD(lr=lr, clipnorm=clipnorm)
        self._model.compile(loss='categorical_crossentropy', optimizer=opt)

        n = experience.states.shape[0]
        # Translate the actions/rewards.
        num_moves = self._encoder.board_width * self._encoder.board_height
        y = np.zeros((n, num_moves))
        for i in range(n):
            action = experience.actions[i]
            reward = experience.rewards[i]
            y[i][action] = reward

        self._model.fit(
            experience.states, y,
            batch_size=batch_size,
            epochs=1)


def simulate_game(black_player, white_player, board_size):
    moves = []
    game = GameState.new_game(board_size)
    agents = {
        Player.black: black_player,
        Player.white: white_player,
    }
    while not game.is_over():
        next_move = agents[game.next_player].select_move(game)
        moves.append(next_move)
        game = game.apply_move(next_move)

    game_result = scoring.compute_game_result(game)

    return game_result.winner


def load_policy_agent(h5file):
    model = kerasutil.load_model_from_hdf5_group(
        h5file['model'],
        custom_objects={'policy_gradient_loss': policy_gradient_loss})
    encoder_name = h5file['encoder'].attrs['name']
    if not isinstance(encoder_name, str):
        encoder_name = encoder_name.decode('ascii')
    board_width = h5file['encoder'].attrs['board_width']
    board_height = h5file['encoder'].attrs['board_height']
    encoder = encoders.get_encoder_by_name(
        encoder_name,
        (board_width, board_height))
    return PolicyAgent(model, encoder)
