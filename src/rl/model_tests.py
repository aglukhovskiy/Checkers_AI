
import encoders
import Board
from Checkers import Checkers
import experience
import h5py
import os
from keras.models import Sequential
from keras.layers import Dense
from keras import metrics
from pg import PolicyAgent
from rl.experience import ExperienceCollector, combine_experience, load_experience
import Board
import numpy as np

exp1 = experience.load_experience(h5py.File('models_n_exp/test_10_plane_500_games.hdf5'))
exp2 = experience.load_experience(h5py.File('models_n_exp/10_plane_iter_1000_games.hdf5'))
exp3 = experience.combine_experience([exp1,exp2])

# os.remove("models_n_exp/10_plane_0_iter_100_games.hdf5")

print(exp1.rewards.shape)
print(exp2.rewards.shape)
print(exp3.rewards.shape)

input_shape = (8,8,10)

# from networks.small import layers
# from networks.medium import layers
from networks.large import layers
# from networks.leaky import layers
# from networks.fullyconnected import layers
# from networks.custom import layers

lr = 0.02
batch_size = 512
epochs=2
activation='linear' # sigmoid


def create_mlp():
    model = Sequential()
    for layer in layers(input_shape):
        model.add(layer)
    model.add(Dense(1, activation=activation))
    return model

encoder = encoders.get_encoder_by_name('tenplane')

model = create_mlp()
# model.compile(loss="mse", optimizer='SGD', metrics=[metrics.BinaryCrossentropy()])
# model.compile(loss="mse", optimizer='adagrad', metrics=[metrics.BinaryCrossentropy()])

agent1 = PolicyAgent(model = model, encoder = encoder)
collector1 = ExperienceCollector()
agent1.set_collector(collector1)


equal_board = Board.Field()
# equal_board.whites_turn=0
print('-------')
board_w_plus_2 = Board.Field()
# board_w_plus_2.whites_turn=0
board_w_plus_2.field['b6'] = None
board_w_plus_2.field['d6'] = None
encoder.show_board(board_w_plus_2)
print('-------')
board_w_plus_6 = Board.Field()
# board_w_plus_6.whites_turn=0
board_w_plus_6.field['a7'] = None
board_w_plus_6.field['b8'] = None
board_w_plus_6.field['b6'] = None
board_w_plus_6.field['c7'] = None
board_w_plus_6.field['d8'] = None
board_w_plus_6.field['d6'] = None
encoder.show_board(board_w_plus_6)
print('-------')
board_w_plus_10 = Board.Field()
# board_w_plus_10.whites_turn=0
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
# board_w_minus_2.whites_turn=0
board_w_minus_2.field['a1'] = None
board_w_minus_2.field['a3'] = None
encoder.show_board(board_w_minus_2)
print('-------')
board_w_minus_6 = Board.Field()
# board_w_minus_6.whites_turn=0
board_w_minus_6.field['a1'] = None
board_w_minus_6.field['a3'] = None
board_w_minus_6.field['b2'] = None
board_w_minus_6.field['c1'] = None
board_w_minus_6.field['c3'] = None
board_w_minus_6.field['d2'] = None
encoder.show_board(board_w_minus_6)
print('-------')
board_w_minus_10 = Board.Field()
# board_w_minus_10.whites_turn=0
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


predict_equal = agent1._model.predict(np.transpose(encoder.encode(equal_board), (1, 2, 0)).reshape(1, 8, 8, 10), verbose=False)
predict_w_plus_2 = agent1._model.predict(np.transpose(encoder.encode(board_w_plus_2), (1, 2, 0)).reshape(1, 8, 8, 10), verbose=False)
predict_w_plus_6 = agent1._model.predict(np.transpose(encoder.encode(board_w_plus_6), (1, 2, 0)).reshape(1, 8, 8, 10), verbose=False)
predict_w_plus_10 = agent1._model.predict(np.transpose(encoder.encode(board_w_plus_10), (1, 2, 0)).reshape(1, 8, 8, 10), verbose=False)
predict_w_minus_2 = agent1._model.predict(np.transpose(encoder.encode(board_w_minus_2), (1, 2, 0)).reshape(1, 8, 8, 10), verbose=False)
predict_w_minus_6 = agent1._model.predict(np.transpose(encoder.encode(board_w_minus_6), (1, 2, 0)).reshape(1, 8, 8, 10), verbose=False)
predict_w_minus_10 = agent1._model.predict(np.transpose(encoder.encode(board_w_minus_10), (1, 2, 0)).reshape(1, 8, 8, 10), verbose=False)

print('Previous model')
print('---------------')
print('predict_equal - ', predict_equal[0][0])
print('predict_w_plus_2 - ', predict_w_plus_2[0][0])
print('predict_w_plus_6 - ', predict_w_plus_6[0][0])
print('predict_w_plus_10 - ', predict_w_plus_10[0][0])
print('predict_w_minus_2 - ', predict_w_minus_2[0][0])
print('predict_w_minus_6 - ', predict_w_minus_6[0][0])
print('predict_w_minus_10 - ', predict_w_minus_10[0][0])



agent2 = PolicyAgent(model = model, encoder = encoder)
collector2 = ExperienceCollector()
agent2.set_collector(collector1)

agent2.train(exp3, lr=lr, epochs = epochs, batch_size=batch_size)

trained_predict_equal = agent2._model.predict(np.transpose(encoder.encode(equal_board), (1, 2, 0)).reshape(1, 8, 8, 10), verbose=False)
trained_predict_w_plus_2 = agent2._model.predict(np.transpose(encoder.encode(board_w_plus_2), (1, 2, 0)).reshape(1, 8, 8, 10), verbose=False)
trained_predict_w_plus_6 = agent2._model.predict(np.transpose(encoder.encode(board_w_plus_6), (1, 2, 0)).reshape(1, 8, 8, 10), verbose=False)
trained_predict_w_plus_10 = agent2._model.predict(np.transpose(encoder.encode(board_w_plus_10), (1, 2, 0)).reshape(1, 8, 8, 10), verbose=False)
trained_predict_w_minus_2 = agent2._model.predict(np.transpose(encoder.encode(board_w_minus_2), (1, 2, 0)).reshape(1, 8, 8, 10), verbose=False)
trained_predict_w_minus_6 = agent2._model.predict(np.transpose(encoder.encode(board_w_minus_6), (1, 2, 0)).reshape(1, 8, 8, 10), verbose=False)
trained_predict_w_minus_10 = agent2._model.predict(np.transpose(encoder.encode(board_w_minus_10), (1, 2, 0)).reshape(1, 8, 8, 10), verbose=False)


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


# with h5py.File('models_n_exp/model_test_10_plane_{}_lr_{}_bs_{}_e.hdf5'.format(lr, batch_size, epochs), 'w') as model_outf:
#     agent1.serialize(model_outf)

# print(exp1.states[:2])

# encoder = encoders.get_encoder_by_name('tenplane')
#
# board = Board.Field()
# board.field['b4'] = Board.Piece(colour='black')
#
# print(encoder.encode(board))

# print(set([y for x,y in board.available_moves()[1].items()]))
# print([(x,y) for x,y in board.available_moves()[2].items()])