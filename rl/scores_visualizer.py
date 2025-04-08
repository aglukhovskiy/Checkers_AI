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
# equal_board.current_player=-1
encoder.show_board(equal_board)
print('-------')
board_w_plus_2 = CheckersGame()
# board_w_plus_2.current_player=-1
board_w_plus_2.board[2][1] = 0
board_w_plus_2.board[2][3] = 0
board_w_plus_2.pieces.remove((2,1))
board_w_plus_2.pieces.remove((2,3))
encoder.show_board(board_w_plus_2)

print(equal_board.pieces)
print(board_w_plus_2.pieces)

print('-------')
board_w_plus_6 = CheckersGame()
# board_w_plus_6.current_player=-1
board_w_plus_6.board[2][1] = 0
board_w_plus_6.board[2][3] = 0
board_w_plus_6.board[2][5] = 0
board_w_plus_6.board[1][0] = 0
board_w_plus_6.board[1][4] = 0
board_w_plus_6.board[1][6] = 0
board_w_plus_6.pieces.remove((2,1))
board_w_plus_6.pieces.remove((2,3))
board_w_plus_6.pieces.remove((2,5))
board_w_plus_6.pieces.remove((1,0))
board_w_plus_6.pieces.remove((1,4))
board_w_plus_6.pieces.remove((1,6))
encoder.show_board(board_w_plus_6)
print('-------')
board_w_plus_10 = CheckersGame()
# board_w_plus_10.current_player=-1
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
board_w_plus_10.pieces.remove((2,1))
board_w_plus_10.pieces.remove((2,3))
board_w_plus_10.pieces.remove((2,5))
board_w_plus_10.pieces.remove((1,0))
board_w_plus_10.pieces.remove((1,4))
board_w_plus_10.pieces.remove((1,6))
board_w_plus_10.pieces.remove((0,1))
board_w_plus_10.pieces.remove((0,3))
board_w_plus_10.pieces.remove((0,5))
board_w_plus_10.pieces.remove((0,7))
encoder.show_board(board_w_plus_10)

print('-------')
board_w_minus_2 = CheckersGame()
# board_w_minus_2.current_player=-1
board_w_minus_2.board[5][0] = 0
board_w_minus_2.board[5][2] = 0
board_w_minus_2.pieces.remove((5,0))
board_w_minus_2.pieces.remove((5,2))
encoder.show_board(board_w_minus_2)
print('-------')
board_w_minus_6 = CheckersGame()
# board_w_minus_6.current_player=-1
board_w_minus_6.board[5][0] = 0
board_w_minus_6.board[5][2] = 0
board_w_minus_6.board[5][4] = 0
board_w_minus_6.board[6][1] = 0
board_w_minus_6.board[6][3] = 0
board_w_minus_6.board[6][5] = 0
board_w_minus_6.pieces.remove((5,0))
board_w_minus_6.pieces.remove((5,2))
board_w_minus_6.pieces.remove((5,4))
board_w_minus_6.pieces.remove((6,1))
board_w_minus_6.pieces.remove((6,3))
board_w_minus_6.pieces.remove((6,5))
encoder.show_board(board_w_minus_6)
print('-------')
board_w_minus_10 = CheckersGame()
# board_w_minus_10.current_player=-1
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
board_w_minus_10.pieces.remove((5,0))
board_w_minus_10.pieces.remove((5,2))
board_w_minus_10.pieces.remove((5,4))
board_w_minus_10.pieces.remove((6,1))
board_w_minus_10.pieces.remove((6,3))
board_w_minus_10.pieces.remove((6,5))
board_w_minus_10.pieces.remove((7,0))
board_w_minus_10.pieces.remove((7,2))
board_w_minus_10.pieces.remove((7,4))
board_w_minus_10.pieces.remove((5,6))
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


# print(np.array([equal_board.board]))
# print(np.array([equal_board.board]).shape)
# print(np.expand_dims(equal_board.board, axis=(0,-1)).shape)
# print('---------------')
# print(np.expand_dims(equal_board.board, axis=(0,-1)))
# print('---------------')
# print(np.expand_dims(equal_board.board, axis=(0)))
### Oneplane
# predict_equal = agent1._model.predict(np.expand_dims(equal_board.board, axis=(0,-1)), verbose=False)
# predict_w_plus_2 = agent1._model.predict(np.expand_dims(board_w_plus_2.board, axis=(0,-1)), verbose=False)
# predict_w_plus_6 = agent1._model.predict(np.expand_dims(board_w_plus_6.board, axis=(0,-1)), verbose=False)
# predict_w_plus_10 = agent1._model.predict(np.expand_dims(board_w_plus_10.board, axis=(0,-1)), verbose=False)
# predict_w_minus_2 = agent1._model.predict(np.expand_dims(board_w_minus_2.board, axis=(0,-1)), verbose=False)
# predict_w_minus_6 = agent1._model.predict(np.expand_dims(board_w_minus_6.board, axis=(0,-1)), verbose=False)
# predict_w_minus_10 = agent1._model.predict(np.expand_dims(board_w_minus_10.board, axis=(0,-1)), verbose=False)
# trained_predict_equal = agent2._model.predict(np.expand_dims(equal_board.board, axis=(0,-1)), verbose=False)
# trained_predict_w_plus_2 = agent2._model.predict(np.expand_dims(board_w_plus_2.board, axis=(0,-1)), verbose=False)
# trained_predict_w_plus_6 = agent2._model.predict(np.expand_dims(board_w_plus_6.board, axis=(0,-1)), verbose=False)
# trained_predict_w_plus_10 = agent2._model.predict(np.expand_dims(board_w_plus_10.board, axis=(0,-1)), verbose=False)
# trained_predict_w_minus_2 = agent2._model.predict(np.expand_dims(board_w_minus_2.board, axis=(0,-1)), verbose=False)
# trained_predict_w_minus_6 = agent2._model.predict(np.expand_dims(board_w_minus_6.board, axis=(0,-1)), verbose=False)
# trained_predict_w_minus_10 = agent2._model.predict(np.expand_dims(board_w_minus_10.board, axis=(0,-1)), verbose=False)

# predict_equal = agent1._model.predict(np.array([equal_board.board]), verbose=False)
# predict_w_plus_2 = agent1._model.predict(np.array([board_w_plus_2.board]), verbose=False)
# predict_w_plus_6 = agent1._model.predict(np.array([board_w_plus_6.board]), verbose=False)
# predict_w_plus_10 = agent1._model.predict(np.array([board_w_plus_10.board]), verbose=False)
# predict_w_minus_2 = agent1._model.predict(np.array([board_w_minus_2.board]), verbose=False)
# predict_w_minus_6 = agent1._model.predict(np.array([board_w_minus_6.board]), verbose=False)
# predict_w_minus_10 = agent1._model.predict(np.array([board_w_minus_10.board]), verbose=False)
# trained_predict_equal = agent2._model.predict(np.array([equal_board.board]), verbose=False)
# trained_predict_w_plus_2 = agent2._model.predict(np.array([board_w_plus_2.board]), verbose=False)
# trained_predict_w_plus_6 = agent2._model.predict(np.array([board_w_plus_6.board]), verbose=False)
# trained_predict_w_plus_10 = agent2._model.predict(np.array([board_w_plus_10.board]), verbose=False)
# trained_predict_w_minus_2 = agent2._model.predict(np.array([board_w_minus_2.board]), verbose=False)
# trained_predict_w_minus_6 = agent2._model.predict(np.array([board_w_minus_6.board]), verbose=False)
# trained_predict_w_minus_10 = agent2._model.predict(np.array([board_w_minus_10.board]), verbose=False)


# ### Twoplane
# predict_equal = agent1._model.predict(np.array([encoder.one_to_two_plane_matrix(equal_board.board)]), verbose=False)
# predict_w_plus_2 = agent1._model.predict(np.array([encoder.one_to_two_plane_matrix(board_w_plus_2.board)]), verbose=False)
# predict_w_plus_6 = agent1._model.predict(np.array([encoder.one_to_two_plane_matrix(board_w_plus_6.board)]), verbose=False)
# predict_w_plus_10 = agent1._model.predict(np.array([encoder.one_to_two_plane_matrix(board_w_plus_10.board)]), verbose=False)
# predict_w_minus_2 = agent1._model.predict(np.array([encoder.one_to_two_plane_matrix(board_w_minus_2.board)]), verbose=False)
# predict_w_minus_6 = agent1._model.predict(np.array([encoder.one_to_two_plane_matrix(board_w_minus_6.board)]), verbose=False)
# predict_w_minus_10 = agent1._model.predict(np.array([encoder.one_to_two_plane_matrix(board_w_minus_10.board)]), verbose=False)
# trained_predict_equal = agent2._model.predict(np.array([encoder.one_to_two_plane_matrix(equal_board.board)]), verbose=False)
# trained_predict_w_plus_2 = agent2._model.predict(np.array([encoder.one_to_two_plane_matrix(board_w_plus_2.board)]), verbose=False)
# trained_predict_w_plus_6 = agent2._model.predict(np.array([encoder.one_to_two_plane_matrix(board_w_plus_6.board)]), verbose=False)
# trained_predict_w_plus_10 = agent2._model.predict(np.array([encoder.one_to_two_plane_matrix(board_w_plus_10.board)]), verbose=False)
# trained_predict_w_minus_2 = agent2._model.predict(np.array([encoder.one_to_two_plane_matrix(board_w_minus_2.board)]), verbose=False)
# trained_predict_w_minus_6 = agent2._model.predict(np.array([encoder.one_to_two_plane_matrix(board_w_minus_6.board)]), verbose=False)
# trained_predict_w_minus_10 = agent2._model.predict(np.array([encoder.one_to_two_plane_matrix(board_w_minus_10.board)]), verbose=False)

## Fourplane
# predict_equal = agent1._model.predict(np.array([encoder.one_to_four_plane_matrix(equal_board.board)]), verbose=False)
# predict_w_plus_2 = agent1._model.predict(np.array([encoder.one_to_four_plane_matrix(board_w_plus_2.board)]), verbose=False)
# predict_w_plus_6 = agent1._model.predict(np.array([encoder.one_to_four_plane_matrix(board_w_plus_6.board)]), verbose=False)
# predict_w_plus_10 = agent1._model.predict(np.array([encoder.one_to_four_plane_matrix(board_w_plus_10.board)]), verbose=False)
# predict_w_minus_2 = agent1._model.predict(np.array([encoder.one_to_four_plane_matrix(board_w_minus_2.board)]), verbose=False)
# predict_w_minus_6 = agent1._model.predict(np.array([encoder.one_to_four_plane_matrix(board_w_minus_6.board)]), verbose=False)
# predict_w_minus_10 = agent1._model.predict(np.array([encoder.one_to_four_plane_matrix(board_w_minus_10.board)]), verbose=False)
# trained_predict_equal = agent2._model.predict(np.array([encoder.one_to_four_plane_matrix(equal_board.board)]), verbose=False)
# trained_predict_w_plus_2 = agent2._model.predict(np.array([encoder.one_to_four_plane_matrix(board_w_plus_2.board)]), verbose=False)
# trained_predict_w_plus_6 = agent2._model.predict(np.array([encoder.one_to_four_plane_matrix(board_w_plus_6.board)]), verbose=False)
# trained_predict_w_plus_10 = agent2._model.predict(np.array([encoder.one_to_four_plane_matrix(board_w_plus_10.board)]), verbose=False)
# trained_predict_w_minus_2 = agent2._model.predict(np.array([encoder.one_to_four_plane_matrix(board_w_minus_2.board)]), verbose=False)
# trained_predict_w_minus_6 = agent2._model.predict(np.array([encoder.one_to_four_plane_matrix(board_w_minus_6.board)]), verbose=False)
# trained_predict_w_minus_10 = agent2._model.predict(np.array([encoder.one_to_four_plane_matrix(board_w_minus_10.board)]), verbose=False)

# ### Tenplane
predict_equal = agent1._model.predict(np.array([encoder.encode(equal_board)]), verbose=False)
predict_w_plus_2 = agent1._model.predict(np.array([encoder.encode(board_w_plus_2)]), verbose=False)
predict_w_plus_6 = agent1._model.predict(np.array([encoder.encode(board_w_plus_6)]), verbose=False)
predict_w_plus_10 = agent1._model.predict(np.array([encoder.encode(board_w_plus_10)]), verbose=False)
predict_w_minus_2 = agent1._model.predict(np.array([encoder.encode(board_w_minus_2)]), verbose=False)
predict_w_minus_6 = agent1._model.predict(np.array([encoder.encode(board_w_minus_6)]), verbose=False)
predict_w_minus_10 = agent1._model.predict(np.array([encoder.encode(board_w_minus_10)]), verbose=False)
trained_predict_equal = agent2._model.predict(np.array([encoder.encode(equal_board)]), verbose=False)
trained_predict_w_plus_2 = agent2._model.predict(np.array([encoder.encode(board_w_plus_2)]), verbose=False)
trained_predict_w_plus_6 = agent2._model.predict(np.array([encoder.encode(board_w_plus_6)]), verbose=False)
trained_predict_w_plus_10 = agent2._model.predict(np.array([encoder.encode(board_w_plus_10)]), verbose=False)
trained_predict_w_minus_2 = agent2._model.predict(np.array([encoder.encode(board_w_minus_2)]), verbose=False)
trained_predict_w_minus_6 = agent2._model.predict(np.array([encoder.encode(board_w_minus_6)]), verbose=False)
trained_predict_w_minus_10 = agent2._model.predict(np.array([encoder.encode(board_w_minus_10)]), verbose=False)

# ### Eightplane
# predict_equal = agent1._model.predict(np.array([encoder.ten_to_eight_plane_matrix(encoder.encode(equal_board))]), verbose=False)
# predict_w_plus_2 = agent1._model.predict(np.array([encoder.ten_to_eight_plane_matrix(encoder.encode(board_w_plus_2))]), verbose=False)
# predict_w_plus_6 = agent1._model.predict(np.array([encoder.ten_to_eight_plane_matrix(encoder.encode(board_w_plus_6))]), verbose=False)
# predict_w_plus_10 = agent1._model.predict(np.array([encoder.ten_to_eight_plane_matrix(encoder.encode(board_w_plus_10))]), verbose=False)
# predict_w_minus_2 = agent1._model.predict(np.array([encoder.ten_to_eight_plane_matrix(encoder.encode(board_w_minus_2))]), verbose=False)
# predict_w_minus_6 = agent1._model.predict(np.array([encoder.ten_to_eight_plane_matrix(encoder.encode(board_w_minus_6))]), verbose=False)
# predict_w_minus_10 = agent1._model.predict(np.array([encoder.ten_to_eight_plane_matrix(encoder.encode(board_w_minus_10))]), verbose=False)
# trained_predict_equal = agent2._model.predict(np.array([encoder.ten_to_eight_plane_matrix(encoder.encode(equal_board))]), verbose=False)
# trained_predict_w_plus_2 = agent2._model.predict(np.array([encoder.ten_to_eight_plane_matrix(encoder.encode(board_w_plus_2))]), verbose=False)
# trained_predict_w_plus_6 = agent2._model.predict(np.array([encoder.ten_to_eight_plane_matrix(encoder.encode(board_w_plus_6))]), verbose=False)
# trained_predict_w_plus_10 = agent2._model.predict(np.array([encoder.ten_to_eight_plane_matrix(encoder.encode(board_w_plus_10))]), verbose=False)
# trained_predict_w_minus_2 = agent2._model.predict(np.array([encoder.ten_to_eight_plane_matrix(encoder.encode(board_w_minus_2))]), verbose=False)
# trained_predict_w_minus_6 = agent2._model.predict(np.array([encoder.ten_to_eight_plane_matrix(encoder.encode(board_w_minus_6))]), verbose=False)
# trained_predict_w_minus_10 = agent2._model.predict(np.array([encoder.ten_to_eight_plane_matrix(encoder.encode(board_w_minus_10))]), verbose=False)

# ### sixplane
# predict_equal = agent1._model.predict(np.array([encoder.ten_to_six_plane_matrix(encoder.encode(equal_board))]), verbose=False)
# predict_w_plus_2 = agent1._model.predict(np.array([encoder.ten_to_six_plane_matrix(encoder.encode(board_w_plus_2))]), verbose=False)
# predict_w_plus_6 = agent1._model.predict(np.array([encoder.ten_to_six_plane_matrix(encoder.encode(board_w_plus_6))]), verbose=False)
# predict_w_plus_10 = agent1._model.predict(np.array([encoder.ten_to_six_plane_matrix(encoder.encode(board_w_plus_10))]), verbose=False)
# predict_w_minus_2 = agent1._model.predict(np.array([encoder.ten_to_six_plane_matrix(encoder.encode(board_w_minus_2))]), verbose=False)
# predict_w_minus_6 = agent1._model.predict(np.array([encoder.ten_to_six_plane_matrix(encoder.encode(board_w_minus_6))]), verbose=False)
# predict_w_minus_10 = agent1._model.predict(np.array([encoder.ten_to_six_plane_matrix(encoder.encode(board_w_minus_10))]), verbose=False)
# trained_predict_equal = agent2._model.predict(np.array([encoder.ten_to_six_plane_matrix(encoder.encode(equal_board))]), verbose=False)
# trained_predict_w_plus_2 = agent2._model.predict(np.array([encoder.ten_to_six_plane_matrix(encoder.encode(board_w_plus_2))]), verbose=False)
# trained_predict_w_plus_6 = agent2._model.predict(np.array([encoder.ten_to_six_plane_matrix(encoder.encode(board_w_plus_6))]), verbose=False)
# trained_predict_w_plus_10 = agent2._model.predict(np.array([encoder.ten_to_six_plane_matrix(encoder.encode(board_w_plus_10))]), verbose=False)
# trained_predict_w_minus_2 = agent2._model.predict(np.array([encoder.ten_to_six_plane_matrix(encoder.encode(board_w_minus_2))]), verbose=False)
# trained_predict_w_minus_6 = agent2._model.predict(np.array([encoder.ten_to_six_plane_matrix(encoder.encode(board_w_minus_6))]), verbose=False)
# trained_predict_w_minus_10 = agent2._model.predict(np.array([encoder.ten_to_six_plane_matrix(encoder.encode(board_w_minus_10))]), verbose=False)

### fiveplane
# predict_equal = agent1._model.predict(np.array([encoder.ten_to_five_plane_matrix(encoder.encode(equal_board))]), verbose=False)
# predict_w_plus_2 = agent1._model.predict(np.array([encoder.ten_to_five_plane_matrix(encoder.encode(board_w_plus_2))]), verbose=False)
# predict_w_plus_6 = agent1._model.predict(np.array([encoder.ten_to_five_plane_matrix(encoder.encode(board_w_plus_6))]), verbose=False)
# predict_w_plus_10 = agent1._model.predict(np.array([encoder.ten_to_five_plane_matrix(encoder.encode(board_w_plus_10))]), verbose=False)
# predict_w_minus_2 = agent1._model.predict(np.array([encoder.ten_to_five_plane_matrix(encoder.encode(board_w_minus_2))]), verbose=False)
# predict_w_minus_6 = agent1._model.predict(np.array([encoder.ten_to_five_plane_matrix(encoder.encode(board_w_minus_6))]), verbose=False)
# predict_w_minus_10 = agent1._model.predict(np.array([encoder.ten_to_five_plane_matrix(encoder.encode(board_w_minus_10))]), verbose=False)
# trained_predict_equal = agent2._model.predict(np.array([encoder.ten_to_five_plane_matrix(encoder.encode(equal_board))]), verbose=False)
# trained_predict_w_plus_2 = agent2._model.predict(np.array([encoder.ten_to_five_plane_matrix(encoder.encode(board_w_plus_2))]), verbose=False)
# trained_predict_w_plus_6 = agent2._model.predict(np.array([encoder.ten_to_five_plane_matrix(encoder.encode(board_w_plus_6))]), verbose=False)
# trained_predict_w_plus_10 = agent2._model.predict(np.array([encoder.ten_to_five_plane_matrix(encoder.encode(board_w_plus_10))]), verbose=False)
# trained_predict_w_minus_2 = agent2._model.predict(np.array([encoder.ten_to_five_plane_matrix(encoder.encode(board_w_minus_2))]), verbose=False)
# trained_predict_w_minus_6 = agent2._model.predict(np.array([encoder.ten_to_five_plane_matrix(encoder.encode(board_w_minus_6))]), verbose=False)
# trained_predict_w_minus_10 = agent2._model.predict(np.array([encoder.ten_to_five_plane_matrix(encoder.encode(board_w_minus_10))]), verbose=False)


print('Previous model')
print('---------------')
print('predict_equal - ', predict_equal)
print('predict_w_plus_2 - ', predict_w_plus_2[0][0])
print('predict_w_plus_6 - ', predict_w_plus_6[0][0])
print('predict_w_plus_10 - ', predict_w_plus_10[0][0])
print('predict_w_minus_2 - ', predict_w_minus_2[0][0])
print('predict_w_minus_6 - ', predict_w_minus_6[0][0])
print('predict_w_minus_10 - ', predict_w_minus_10[0][0])
print('---------------')
print('Trained model')
print('---------------')
print('predict_equal - ', trained_predict_equal)
print('predict_w_plus_2 - ', trained_predict_w_plus_2[0][0])
print('predict_w_plus_6 - ', trained_predict_w_plus_6[0][0])
print('predict_w_plus_10 - ', trained_predict_w_plus_10[0][0])
print('predict_w_minus_2 - ', trained_predict_w_minus_2[0][0])
print('predict_w_minus_6 - ', trained_predict_w_minus_6[0][0])
print('predict_w_minus_10 - ', trained_predict_w_minus_10[0][0])


# predict_equal -  [[-0.07471003]]
# predict_w_plus_2 -  -0.074710034
# predict_w_plus_6 -  -0.38610873
# predict_w_plus_10 -  -0.38610873
# predict_w_minus_2 -  -1.0817989
# predict_w_minus_6 -  -1.0904804
# predict_w_minus_10 -  -1.1680621

# loss = 0.9
# predict_equal -  -0.12878415
# predict_w_plus_2 -  0.51510125
# predict_w_plus_6 -  1.1627402
# predict_w_plus_10 -  1.4958886
# predict_w_minus_2 -  -0.67029035
# predict_w_minus_6 -  -1.0968739
# predict_w_minus_10 -  -1.1971568

# Epoch 1/3
# 478/478 [==============================] - 17s 35ms/step - loss: 0.9030
# Epoch 2/3
# 478/478 [==============================] - 17s 37ms/step - loss: 0.6489
# Epoch 3/3
# 478/478 [==============================] - 18s 37ms/step - loss: 0.6203

# predict_equal -  -0.95424324
# predict_w_plus_2 -  0.44825146
# predict_w_plus_6 -  0.9911151
# predict_w_plus_10 -  1.0595369
# predict_w_minus_2 -  -1.0111594
# predict_w_minus_6 -  -1.0296133
# predict_w_minus_10 -  -1.1374646

# 64
# Epoch 1/4
# 1911/1911 [==============================] - 24s 13ms/step - loss: 0.7010
# Epoch 2/4
# 1911/1911 [==============================] - 28s 14ms/step - loss: 0.6077
# Epoch 3/4
# 1911/1911 [==============================] - 28s 15ms/step - loss: 0.5949
# Epoch 4/4
# 1911/1911 [==============================] - 30s 16ms/step - loss: 0.5884

# predict_equal -  -0.5792365
# predict_w_plus_2 -  0.93097013
# predict_w_plus_6 -  0.96133894
# predict_w_plus_10 -  0.9688373
# predict_w_minus_2 -  -0.92363304
# predict_w_minus_6 -  -0.92480576
# predict_w_minus_10 -  -0.9364788