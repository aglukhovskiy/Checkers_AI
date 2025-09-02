from src.core.board import Field
from .encoders.oneplane import OnePlaneEncoder
import numpy as np
try:
    from keras.models import load_model
    KERAS_AVAILABLE = True
except ImportError:
    KERAS_AVAILABLE = False

encoder = OnePlaneEncoder()

class Bot():
    def __init__(self, the_depth, the_board=None):
        self.board = the_board
        self.depth = the_depth
        self.colour = 'black'

    def __str__(self):
        return 'Bot class'

    def get_next_move(self):
        # return random.choice(self.board.available_moves(colour=self.colour)[0])
        if  len(self.board.available_moves(self.board.white_num_to_colour[self.board.whites_turn])[0])==1:
            return self.board.available_moves(self.board.white_num_to_colour[self.board.whites_turn])[0][0]
        elif  len(self.board.available_moves(self.board.white_num_to_colour[self.board.whites_turn])[0])==0:
            return None
        elif  len(self.board.available_moves(self.board.white_num_to_colour[self.board.whites_turn])[2])>1: # if 2+ multiple takes - choose random
            if len(self.board.multiple_jumping_piece)>0: # for second+ moves at one turn
                for i in self.board.available_moves(self.board.white_num_to_colour[self.board.whites_turn])[0]:
                    if i[0:2] in self.board.multiple_jumping_piece:
                        return i
            return self.board.available_moves(self.board.white_num_to_colour[self.board.whites_turn])[0][0]

        alpha_beta_result = self.alpha_beta(self.board, self.depth, float('-inf'), float('inf'), maximizing_whites=0, visualize_tree=0)
        if alpha_beta_result[1]==None: # all moves leads to defeat
            return self.board.available_moves(self.board.white_num_to_colour[self.board.whites_turn])[0][0]
        return alpha_beta_result[1]

    def alpha_beta(self, board, depth, alpha, beta, maximizing_whites, visualize_tree=0):

        board.bot_test = 1
        players_info = board.get_number_of_pieces_and_kings()
        if visualize_tree==1:
            board.childs = []
            board.params = {'alpha': alpha, 'beta': beta, 'black_score': players_info[1] + 2 * players_info[3] - (players_info[0] + 2 * players_info[2])}

        if board.game_is_on==0:
            if list(map(lambda x:self.board.get_number_of_pieces_and_kings()[x],[0,2])) == [0, 0]:
                if maximizing_whites==1:
                    return -1000000, None
                else:
                    return 1000000, None
            elif list(map(lambda x:self.board.get_number_of_pieces_and_kings()[x],[1,3])) == [0, 0]:
                if maximizing_whites==1:
                    return 1000000, None
                else:
                    return -1000000, None
            else:
                return 0, None

        if depth == 0:
            if board.whites_turn != maximizing_whites:
                return players_info[1] + 2 * players_info[3] - (players_info[0] + 2 * players_info[2]), None
            return players_info[0] + 2 * players_info[2] - (players_info[1] + 2 * players_info[3]), None

        possible_moves = board.available_moves(board.white_num_to_colour[board.whites_turn])[0]

        potential_spots = board.get_potential_spots_from_moves(moves=possible_moves, opp=self)

        desired_move_index = None

        if maximizing_whites==1:
            v = float('-inf')
            for j in range(len(potential_spots)):
                cur_board = Field(preset=potential_spots[j], whites_turn=0) # проверить, какой должен быть turn

                if visualize_tree==1:
                    cur_board.parent = board

                alpha_beta_results = self.alpha_beta(cur_board, depth - 1, alpha, beta, 0, visualize_tree=visualize_tree)
                if v < alpha_beta_results[0]:
                    v = alpha_beta_results[0]
                    alpha = max(alpha, v)
                    desired_move_index = j
                    if visualize_tree==1:
                        board.params['alpha'] = max(alpha, v)
                if visualize_tree==1:
                    board.childs.append(cur_board)
                if beta <= alpha:
                    break
            if desired_move_index is None:
                return v, None
            board.bot_test = 0
            return v, possible_moves[desired_move_index]
        else:
            v = float('inf')
            for j in range(len(potential_spots)):
                cur_board = Field(potential_spots[j], whites_turn=1) # проверить, какой должен быть turn
                if visualize_tree==1:
                    cur_board.parent = board
                alpha_beta_results = self.alpha_beta(cur_board, depth - 1, alpha, beta, 1, visualize_tree=visualize_tree)
                if v > alpha_beta_results[0]:
                    v = alpha_beta_results[0]
                    desired_move_index = j
                    beta = min(beta, v)
                    if visualize_tree==1:
                        board.params['beta'] = min(beta, v)
                if visualize_tree==1:
                    board.childs.append(cur_board)
                if beta <= alpha:
                    break
            if desired_move_index is None:
                return v, None
            board.bot_test = 0

            if visualize_tree == 1:
                print('Childs1')
                queue = []
                cur_parent_id = id(board)
                encoder.show_board(board)
                print('initial board - ', id(board))
                lvl = 0
                counter = 0
                tree = []
                cur_lvl = []

                for child in board.childs:
                    queue.append(child)
                    cur_lvl.append([child, [child.params['alpha'], child.params['beta'], id(child), id(child.parent), child.params['black_score']]])
                while queue:
                    c = queue.pop(0)
                    if id(c.parent) == cur_parent_id:
                        lvl+=1
                        cur_parent_id = id(c)
                        tree.append(cur_lvl)
                        cur_lvl=[]

                    counter+=1

                    for next_child in c.childs:
                        queue.append(next_child)
                        cur_lvl.append([next_child, [next_child.params['alpha'], next_child.params['beta'], id(next_child), id(next_child.parent), next_child.params['black_score']]])
                tree.append(cur_lvl)

                for lvl in tree:
                    lvl_boards = []
                    lvl_data = []
                    for board in lvl:
                        lvl_boards.append(board[0])
                        lvl_data.append(board[1])
                    print([' @ parent - ' + str(t[3]) + '                  @' for t in lvl_data])
                    if len(lvl_boards)>=1:
                        encoder.show_several_boards(lvl_boards)
                    print([' @ alphabeta - ' + str(t[0]) + ', ' + str(t[1]) + ', bscore - ' + str(t[4]) + '@' for t in lvl_data])
                    print([' @ id - ' + str(t[2]) + ' @' for t in lvl_data])
                print(counter)

            return v, possible_moves[desired_move_index]

class BotNN(Bot):
    def __init__(self, the_board=None):
        self.board = the_board
        self.colour = 'black'
        if KERAS_AVAILABLE:
            self.model = load_model('model_1.keras')
        else:
            self.model = None
            print("Warning: Keras is not available. BotNN will not work properly.")

    def __str__(self):
        return 'Bot class'

    def get_next_move(self):
        if self.model is None:
            # Fallback to basic bot behavior if Keras is not available
            return super().get_next_move()
            
        # return random.choice(self.board.available_moves(colour=self.colour)[0])
        if len(self.board.available_moves(self.board.white_num_to_colour[self.board.whites_turn])[0])==1:
            return self.board.available_moves(self.board.white_num_to_colour[self.board.whites_turn])[0][0]
        elif len(self.board.available_moves(self.board.white_num_to_colour[self.board.whites_turn])[0])==0:
            return None
        elif len(self.board.available_moves(self.board.white_num_to_colour[self.board.whites_turn])[0])>1:
            possible_moves = self.board.available_moves(self.board.white_num_to_colour[self.board.whites_turn])[0]
            potential_spots = self.board.get_potential_spots_from_moves(moves=possible_moves, opp=self)
            scores_list = []
            for i in potential_spots:
                matrix=encoder.encode(board=None, field=i)
                predict = self.model.predict(matrix.reshape(8, 8, 1))[0][0]
                scores_list.append(predict)
            if self.board.whites_turn==1:
                return possible_moves[np.argmax(scores_list)]
            if self.board.whites_turn==0:
                return possible_moves[np.argmin(scores_list)]
