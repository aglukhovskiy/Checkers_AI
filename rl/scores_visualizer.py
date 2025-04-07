from experience import load_experience, combine_experience
import h5py
from rl.pg_agent import PolicyAgent, load_policy_agent
import numpy as np
import encoders
from Board_v2 import CheckersGame

# encoder = encoders.get_encoder_by_name('oneplane_singlesided')
encoder = encoders.get_encoder_by_name('tenplane_v2')

# exp = load_experience(h5py.File('test_1000_games.hdf5'))

with h5py.File('models_n_exp/test_model_small.hdf5', 'r') as agent_file:
    agent1 = load_policy_agent(agent_file)
with h5py.File('models_n_exp/test_model_small_trained.hdf5', 'r') as agent_file:
    agent2 = load_policy_agent(agent_file)

# agent2.train(load_experience(h5py.File('test_1000_games.hdf5')), lr=0.005)
# agent2.train(load_experience(h5py.File('models_n_exp/test_10_plane_500_games.hdf5')), lr=0.02)

equal_board = CheckersGame()
equal_board.current_player=-1
encoder.show_board(equal_board)
print('-------')
board_w_plus_2 = CheckersGame()
board_w_plus_2.current_player=-1
board_w_plus_2.board[2][1] = 0
board_w_plus_2.board[2][3] = 0
encoder.show_board(board_w_plus_2)
print('-------')
board_w_plus_6 = CheckersGame()
board_w_plus_6.current_player=-1
board_w_plus_6.board[2][1] = 0
board_w_plus_6.board[2][3] = 0
board_w_plus_6.board[2][5] = 0
board_w_plus_6.board[1][0] = 0
board_w_plus_6.board[1][4] = 0
board_w_plus_6.board[1][6] = 0
encoder.show_board(board_w_plus_6)
print('-------')
board_w_plus_10 = CheckersGame()
board_w_plus_10.current_player=-1
board_w_plus_10.board[2][1] = 0
board_w_plus_10.board[2][3] = 0
board_w_plus_10.board[2][5] = 0
board_w_plus_10.board[1][0] = 0
board_w_plus_10.board[1][4] = 0
board_w_plus_10.board[1][6] = 0
board_w_plus_10.board[0][1] = 0
board_w_plus_10.board[0][3] = 0
board_w_plus_10.board[0][5] = 0
board_w_plus_10.board[0][7] = 0
encoder.show_board(board_w_plus_10)

print('-------')
board_w_minus_2 = CheckersGame()
board_w_minus_2.current_player=-1
board_w_minus_2.board[5][0] = 0
board_w_minus_2.board[5][2] = 0
encoder.show_board(board_w_minus_2)
print('-------')
board_w_minus_6 = CheckersGame()
board_w_minus_6.current_player=-1
board_w_minus_6.board[5][0] = 0
board_w_minus_6.board[5][2] = 0
board_w_minus_6.board[5][4] = 0
board_w_minus_6.board[6][1] = 0
board_w_minus_6.board[6][3] = 0
board_w_minus_6.board[6][5] = 0
encoder.show_board(board_w_minus_6)
print('-------')
board_w_minus_10 = CheckersGame()
board_w_minus_10.current_player=-1
board_w_minus_10.board[5][0] = 0
board_w_minus_10.board[5][2] = 0
board_w_minus_10.board[5][4] = 0
board_w_minus_10.board[6][1] = 0
board_w_minus_10.board[6][3] = 0
board_w_minus_10.board[6][5] = 0
board_w_minus_10.board[7][0] = 0
board_w_minus_10.board[7][2] = 0
board_w_minus_10.board[7][4] = 0
board_w_minus_10.board[5][6] = 0
encoder.show_board(board_w_minus_10)

# predict_equal = agent1._model.predict(agent1._prepare_input(encoder.encode(equal_board)), verbose=False)
# predict_w_plus_2 = agent1._model.predict(agent1._prepare_input(encoder.encode(board_w_plus_2)), verbose=False)
# predict_w_plus_6 = agent1._model.predict(agent1._prepare_input(encoder.encode(board_w_plus_6)), verbose=False)
# predict_w_plus_10 = agent1._model.predict(agent1._prepare_input(encoder.encode(board_w_plus_10)), verbose=False)
# predict_w_minus_2 = agent1._model.predict(agent1._prepare_input(encoder.encode(board_w_minus_2)), verbose=False)
# predict_w_minus_6 = agent1._model.predict(agent1._prepare_input(encoder.encode(board_w_minus_6)), verbose=False)
# predict_w_minus_10 = agent1._model.predict(agent1._prepare_input(encoder.encode(board_w_minus_10)), verbose=False)

# trained_predict_equal = agent2._model.predict(np.array([encoder.encode(equal_board)])[0], verbose=False)
# trained_predict_w_plus_2 = agent2._model.predict(np.array([encoder.encode(board_w_plus_2)])[0], verbose=False)
# trained_predict_w_plus_6 = agent2._model.predict(np.array([encoder.encode(board_w_plus_6)])[0], verbose=False)
# trained_predict_w_plus_10 = agent2._model.predict(np.array([encoder.encode(board_w_plus_10)])[0], verbose=False)
# trained_predict_w_minus_2 = agent2._model.predict(np.array([encoder.encode(board_w_minus_2)])[0], verbose=False)
# trained_predict_w_minus_6 = agent2._model.predict(np.array([encoder.encode(board_w_minus_6)])[0], verbose=False)
# trained_predict_w_minus_10 = agent2._model.predict(np.array([encoder.encode(board_w_minus_10)])[0], verbose=False)


# predict_equal = agent1._model.predict(np.transpose(encoder.encode(equal_board), (1, 2, 0)).reshape(1, 8, 8, 10), verbose=False)
# predict_w_plus_2 = agent1._model.predict(np.transpose(encoder.encode(board_w_plus_2), (1, 2, 0)).reshape(1, 8, 8, 10), verbose=False)
# predict_w_plus_6 = agent1._model.predict(np.transpose(encoder.encode(board_w_plus_6), (1, 2, 0)).reshape(1, 8, 8, 10), verbose=False)
# predict_w_plus_10 = agent1._model.predict(np.transpose(encoder.encode(board_w_plus_10), (1, 2, 0)).reshape(1, 8, 8, 10), verbose=False)
# predict_w_minus_2 = agent1._model.predict(np.transpose(encoder.encode(board_w_minus_2), (1, 2, 0)).reshape(1, 8, 8, 10), verbose=False)
# predict_w_minus_6 = agent1._model.predict(np.transpose(encoder.encode(board_w_minus_6), (1, 2, 0)).reshape(1, 8, 8, 10), verbose=False)
# predict_w_minus_10 = agent1._model.predict(np.transpose(encoder.encode(board_w_minus_10), (1, 2, 0)).reshape(1, 8, 8, 10), verbose=False)

# predict_input_equal = np.transpose(encoder.encode(equal_board), (1, 2, 0)).reshape(1, 8, 8, 10)
# predict_input_w_plus_2 = np.transpose(encoder.encode(board_w_plus_2), (1, 2, 0)).reshape(1, 8, 8, 10)
#
# print(predict_input_equal)
#
# print('---------------')
# print(predict_input_w_plus_2)
# print('---------------')
# trained_predict_equal = agent2._model.predict(predict_input_equal, verbose=False)
# trained_predict_w_plus_2 = agent2._model.predict(predict_input_w_plus_2, verbose=False)

# trained_predict_equal = agent2._model.predict(np.transpose(encoder.encode(equal_board), (1, 2, 0)).reshape(1, 8, 8, 10), verbose=False)
# trained_predict_w_plus_2 = agent2._model.predict(np.transpose(encoder.encode(board_w_plus_2), (1, 2, 0)).reshape(1, 8, 8, 10), verbose=False)
# trained_predict_w_plus_6 = agent2._model.predict(np.transpose(encoder.encode(board_w_plus_6), (1, 2, 0)).reshape(1, 8, 8, 10), verbose=False)
# trained_predict_w_plus_10 = agent2._model.predict(np.transpose(encoder.encode(board_w_plus_10), (1, 2, 0)).reshape(1, 8, 8, 10), verbose=False)
# trained_predict_w_minus_2 = agent2._model.predict(np.transpose(encoder.encode(board_w_minus_2), (1, 2, 0)).reshape(1, 8, 8, 10), verbose=False)
# trained_predict_w_minus_6 = agent2._model.predict(np.transpose(encoder.encode(board_w_minus_6), (1, 2, 0)).reshape(1, 8, 8, 10), verbose=False)
# trained_predict_w_minus_10 = agent2._model.predict(np.transpose(encoder.encode(board_w_minus_10), (1, 2, 0)).reshape(1, 8, 8, 10), verbose=False)


# print(np.expand_dims(equal_board.board, axis=(0,-1)))


### Oneplane
predict_equal = agent1._model.predict(np.expand_dims(equal_board.board, axis=(0,-1)), verbose=False)
predict_w_plus_2 = agent1._model.predict(np.expand_dims(board_w_plus_2.board, axis=(0,-1)), verbose=False)
predict_w_plus_6 = agent1._model.predict(np.expand_dims(board_w_plus_6.board, axis=(0,-1)), verbose=False)
predict_w_plus_10 = agent1._model.predict(np.expand_dims(board_w_plus_10.board, axis=(0,-1)), verbose=False)
predict_w_minus_2 = agent1._model.predict(np.expand_dims(board_w_minus_2.board, axis=(0,-1)), verbose=False)
predict_w_minus_6 = agent1._model.predict(np.expand_dims(board_w_minus_6.board, axis=(0,-1)), verbose=False)
predict_w_minus_10 = agent1._model.predict(np.expand_dims(board_w_minus_10.board, axis=(0,-1)), verbose=False)
trained_predict_equal = agent2._model.predict(np.expand_dims(equal_board.board, axis=(0,-1)), verbose=False)
trained_predict_w_plus_2 = agent2._model.predict(np.expand_dims(board_w_plus_2.board, axis=(0,-1)), verbose=False)
trained_predict_w_plus_6 = agent2._model.predict(np.expand_dims(board_w_plus_6.board, axis=(0,-1)), verbose=False)
trained_predict_w_plus_10 = agent2._model.predict(np.expand_dims(board_w_plus_10.board, axis=(0,-1)), verbose=False)
trained_predict_w_minus_2 = agent2._model.predict(np.expand_dims(board_w_minus_2.board, axis=(0,-1)), verbose=False)
trained_predict_w_minus_6 = agent2._model.predict(np.expand_dims(board_w_minus_6.board, axis=(0,-1)), verbose=False)
trained_predict_w_minus_10 = agent2._model.predict(np.expand_dims(board_w_minus_10.board, axis=(0,-1)), verbose=False)

print('Previous model')
print('---------------')
print('predict_equal - ', predict_equal[0][0])
print('predict_w_plus_2 - ', predict_w_plus_2[0][0])
print('predict_w_plus_6 - ', predict_w_plus_6[0][0])
print('predict_w_plus_10 - ', predict_w_plus_10[0][0])
print('predict_w_minus_2 - ', predict_w_minus_2[0][0])
print('predict_w_minus_6 - ', predict_w_minus_6[0][0])
print('predict_w_minus_10 - ', predict_w_minus_10[0][0])
print('---------------')
print('Trained model')
print('---------------')
print('predict_equal - ', trained_predict_equal[0][0])
print('predict_w_plus_2 - ', trained_predict_w_plus_2[0][0])
print('predict_w_plus_6 - ', trained_predict_w_plus_6[0][0])
print('predict_w_plus_10 - ', trained_predict_w_plus_10[0][0])
print('predict_w_minus_2 - ', trained_predict_w_minus_2[0][0])
print('predict_w_minus_6 - ', trained_predict_w_minus_6[0][0])
print('predict_w_minus_10 - ', trained_predict_w_minus_10[0][0])
