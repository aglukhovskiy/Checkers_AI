from experience import load_experience, combine_experience
import h5py
from rl.pg import PolicyAgent, load_policy_agent
import numpy as np
import encoders
import Board
from Checkers import Checkers

# encoder = encoders.get_encoder_by_name('oneplane_singlesided')
encoder = encoders.get_encoder_by_name('sixplane')

# exp = load_experience(h5py.File('test_1000_games.hdf5'))
agent1 = load_policy_agent(h5py.File('model_test_6_plane.hdf5'))
agent2 = load_policy_agent(h5py.File('model_test_6_plane.hdf5'))
# agent2.train(load_experience(h5py.File('test_1000_games.hdf5')), lr=0.005)
agent2.train(load_experience(h5py.File('test_6_plane_300_games.hdf5')), lr=0.01)

equal_board = Board.Field()
print('-------')
board_w_plus_2 = Board.Field()
board_w_plus_2.field['b6'] = None
board_w_plus_2.field['d6'] = None
encoder.show_board(board_w_plus_2)
print('-------')
board_w_plus_6 = Board.Field()
board_w_plus_6.field['a7'] = None
board_w_plus_6.field['b8'] = None
board_w_plus_6.field['b6'] = None
board_w_plus_6.field['c7'] = None
board_w_plus_6.field['d8'] = None
board_w_plus_6.field['d6'] = None
encoder.show_board(board_w_plus_6)
print('-------')
board_w_plus_10 = Board.Field()
board_w_plus_10.field['a7'] = None
board_w_plus_10.field['b8'] = None
board_w_plus_10.field['b6'] = None
board_w_plus_10.field['c7'] = None
board_w_plus_10.field['d8'] = None
board_w_plus_10.field['d6'] = None
board_w_plus_10.field['e7'] = None
board_w_plus_10.field['f8'] = None
board_w_plus_10.field['f6'] = None
board_w_plus_10.field['g7'] = None
board_w_plus_10.field['h8'] = None
encoder.show_board(board_w_plus_10)
print('-------')
board_w_minus_2 = Board.Field()
board_w_minus_2.field['a1'] = None
board_w_minus_2.field['a3'] = None
encoder.show_board(board_w_minus_2)
print('-------')
board_w_minus_6 = Board.Field()
board_w_minus_6.field['a1'] = None
board_w_minus_6.field['a3'] = None
board_w_minus_6.field['b2'] = None
board_w_minus_6.field['c1'] = None
board_w_minus_6.field['c3'] = None
board_w_minus_6.field['d2'] = None
encoder.show_board(board_w_minus_6)
print('-------')
board_w_minus_10 = Board.Field()
board_w_minus_10.field['a1'] = None
board_w_minus_10.field['a3'] = None
board_w_minus_10.field['b2'] = None
board_w_minus_10.field['c1'] = None
board_w_minus_10.field['c3'] = None
board_w_minus_10.field['d2'] = None
board_w_minus_10.field['e1'] = None
board_w_minus_10.field['e3'] = None
board_w_minus_10.field['f2'] = None
board_w_minus_10.field['g1'] = None
board_w_minus_10.field['g3'] = None
encoder.show_board(board_w_minus_10)

# predict_equal = agent1._model.predict(np.array([encoder.encode(equal_board)])[0], verbose=False)
# predict_w_plus_2 = agent1._model.predict(np.array([encoder.encode(board_w_plus_2)])[0], verbose=False)
# predict_w_plus_6 = agent1._model.predict(np.array([encoder.encode(board_w_plus_6)])[0], verbose=False)
# predict_w_plus_10 = agent1._model.predict(np.array([encoder.encode(board_w_plus_10)])[0], verbose=False)
# predict_w_minus_2 = agent1._model.predict(np.array([encoder.encode(board_w_minus_2)])[0], verbose=False)
# predict_w_minus_6 = agent1._model.predict(np.array([encoder.encode(board_w_minus_6)])[0], verbose=False)
# predict_w_minus_10 = agent1._model.predict(np.array([encoder.encode(board_w_minus_10)])[0], verbose=False)
#
# trained_predict_equal = agent2._model.predict(np.array([encoder.encode(equal_board)])[0], verbose=False)
# trained_predict_w_plus_2 = agent2._model.predict(np.array([encoder.encode(board_w_plus_2)])[0], verbose=False)
# trained_predict_w_plus_6 = agent2._model.predict(np.array([encoder.encode(board_w_plus_6)])[0], verbose=False)
# trained_predict_w_plus_10 = agent2._model.predict(np.array([encoder.encode(board_w_plus_10)])[0], verbose=False)
# trained_predict_w_minus_2 = agent2._model.predict(np.array([encoder.encode(board_w_minus_2)])[0], verbose=False)
# trained_predict_w_minus_6 = agent2._model.predict(np.array([encoder.encode(board_w_minus_6)])[0], verbose=False)
# trained_predict_w_minus_10 = agent2._model.predict(np.array([encoder.encode(board_w_minus_10)])[0], verbose=False)


predict_equal = agent1._model.predict(np.transpose(encoder.encode(equal_board), (1, 2, 0)).reshape(1, 8, 8, 6), verbose=False)
predict_w_plus_2 = agent1._model.predict(np.transpose(encoder.encode(board_w_plus_2), (1, 2, 0)).reshape(1, 8, 8, 6), verbose=False)
predict_w_plus_6 = agent1._model.predict(np.transpose(encoder.encode(board_w_plus_6), (1, 2, 0)).reshape(1, 8, 8, 6), verbose=False)
predict_w_plus_10 = agent1._model.predict(np.transpose(encoder.encode(board_w_plus_10), (1, 2, 0)).reshape(1, 8, 8, 6), verbose=False)
predict_w_minus_2 = agent1._model.predict(np.transpose(encoder.encode(board_w_minus_2), (1, 2, 0)).reshape(1, 8, 8, 6), verbose=False)
predict_w_minus_6 = agent1._model.predict(np.transpose(encoder.encode(board_w_minus_6), (1, 2, 0)).reshape(1, 8, 8, 6), verbose=False)
predict_w_minus_10 = agent1._model.predict(np.transpose(encoder.encode(board_w_minus_10), (1, 2, 0)).reshape(1, 8, 8, 6), verbose=False)

trained_predict_equal = agent2._model.predict(np.transpose(encoder.encode(equal_board), (1, 2, 0)).reshape(1, 8, 8, 6), verbose=False)
trained_predict_w_plus_2 = agent2._model.predict(np.transpose(encoder.encode(board_w_plus_2), (1, 2, 0)).reshape(1, 8, 8, 6), verbose=False)
trained_predict_w_plus_6 = agent2._model.predict(np.transpose(encoder.encode(board_w_plus_6), (1, 2, 0)).reshape(1, 8, 8, 6), verbose=False)
trained_predict_w_plus_10 = agent2._model.predict(np.transpose(encoder.encode(board_w_plus_10), (1, 2, 0)).reshape(1, 8, 8, 6), verbose=False)
trained_predict_w_minus_2 = agent2._model.predict(np.transpose(encoder.encode(board_w_minus_2), (1, 2, 0)).reshape(1, 8, 8, 6), verbose=False)
trained_predict_w_minus_6 = agent2._model.predict(np.transpose(encoder.encode(board_w_minus_6), (1, 2, 0)).reshape(1, 8, 8, 6), verbose=False)
trained_predict_w_minus_10 = agent2._model.predict(np.transpose(encoder.encode(board_w_minus_10), (1, 2, 0)).reshape(1, 8, 8, 6), verbose=False)

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