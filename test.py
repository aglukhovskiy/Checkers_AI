
import Board_test

class Checkers:

    def __init__(self, board, opp, control='gui'):
        self.control = control
        self.opp = opp
        self.opp_colour = 'black' # 0: black
        # self.multiple_jumping_piece = []
        self.board = board

    def next_turn(self, move_text = None):
        if self.control == 'command':
            return self.next_turn_by_hand(move_text)
        if move_text == 'end':
            self.board.game_is_on = 0
            # print('game over')
            return 'game over'
        elif self.board.game_is_on == 1 :
            if self.board.whites_turn == 1:
                self.board.move(opp = self.opp, move_text=move_text)
            elif self.board.whites_turn == 0:
                self.board.move(opp = self.opp, move_text=move_text)
        elif self.board.game_is_on == 0:
            # print('game over')
            return 'game over'
        else:
            print('Error')
        return self.board.history

    def next_turn_by_hand(self, move_text):
        if self.board.game_is_on == 1 :
            if self.board.whites_turn == 1:
                # print(self.board.available_moves(), self.board.whites_turn)
                self.board.move(opp = self.opp, move_text=move_text)
            elif self.board.whites_turn == 0:
                # print(self.board.available_moves(), self.board.whites_turn)
                self.board.move(opp = self.opp, move_text=move_text)
        if len(self.board.available_moves()[0]) == 0:
            self.board.game_is_on = 0
        # if self.board.game_is_on == 0:
            # print('game over')

# Text game

if __name__ == '__main__':
    f = Board_test.Field()

    match = Checkers(control='command', opp='bot', board=f)

    print(f.fields)
    print(f.available_moves())
    match.next_turn(move_text='e3f4')
    print(f.available_moves())
    match.next_turn(move_text='d6e5')
    print(f.available_moves())
    match.next_turn(move_text='d2e3')
    print(f.available_moves())
    # match.next_turn()
    # match.next_turn()
    # encoder.show_board(f)
    # print(encoder.encode(f)[0][::-1])
    # print(match.board.field)



    # while match.board.game_is_on == True:
    #     match.next_turn()










import numpy as np

# from keras.models import load_model

# model = load_model('model_1.keras')

# model.save('model_1.keras')  # creates a HDF5 file 'my_model.h5'

# feature = np.load('features_50r_15m_100_enr.npy')
# features = np.load('features2.npy')
# features_2 = np.load('features_50r_15m.npy')
# features_3 = np.load('features_50r_15m_2.npy')
# labels = np.load('labels2.npy')
# wt = np.load('wt2.npy')

# features_50r_15m = np.load('features_50r_15m_100_enr.npy')
# labels_50r_15m = np.load('label_50r_15m_100_enr.npy')
# wt_50r_15m = np.load('whites_turn_50r_15m_100_enr.npy')
#
# features2 = np.concatenate([features, features_50r_15m])
# labels2 = np.concatenate([labels, labels_50r_15m])
# wt2 = np.concatenate([wt, wt_50r_15m])

# np.save('features2.npy', features2)  # <4>
# np.save('labels2.npy', labels2)
# np.save('wt2.npy', wt2)

# print(len(features))
# print(len(labels))
# print(len(wt))
#
# print(len(features_50r_15m))
# print(len(labels_50r_15m))
# print(len(wt_50r_15m))
#
# print(len(features2))
# print(len(labels2))
# print(len(wt2))

# print(np.mean(abs(labels-0.3438581971374472)))

# print(len(features_50r_15m))
#
# samples = features.shape[0]
# board_size = 8
# X = features[0].reshape(1, board_size, board_size)
#
# preds = model.predict(X)
# print(preds[0][0])

# import Bot
# import Board
# import Encoder
#
# from utils import move_text_to_point
#
#
# f = Board.Field()
#
# encoder = Encoder.OnePlaneEncoder()
#
# l = []
# l.append(move_text_to_point('a1b2'))
# l.append(move_text_to_point('a3b4'))
#
# print(l)

# encoder.show_board(f)
#
# print(one_plane_encode(f)[::-1])
#
# print(encoder.encode(f)[0])
