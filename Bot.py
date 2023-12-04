import random
import Board

class Bot():
    def __init__(self, the_depth, the_board=None):
        self.board = the_board
        self.depth = the_depth
        self.colour = 'black'

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

        alpha_beta_result = self.alpha_beta(self.board, self.depth, float('-inf'), float('inf'), maximizing_whites=0)
        if alpha_beta_result[1]==None: # all moves leads to defeat
            return self.board.available_moves(self.board.white_num_to_colour[self.board.whites_turn])[0][0]
        return alpha_beta_result[1]

    def alpha_beta(self, board, depth, alpha, beta, maximizing_whites):
        board.bot_test = 1
        if board.game_is_on==0:
            if list(map(lambda x:self.board.get_number_of_pieces_and_kings()[x],[0,2])) == [0, 0]:
                if maximizing_whites==1:
                    return -10000000, None
                else:
                    return 10000000, None
            elif list(map(lambda x:self.board.get_number_of_pieces_and_kings()[x],[1,3])) == [0, 0]:
                if maximizing_whites==1:
                    return 1000000, None
                else:
                    return -1000000, None
            else:
                return 0, None

        if depth == 0:
            players_info = board.get_number_of_pieces_and_kings()
            if board.whites_turn != maximizing_whites:
                return players_info[1] + 2 * players_info[3] - (players_info[0] + 2 * players_info[2]), None
            return players_info[0] + 2 * players_info[2] - (players_info[1] + 2 * players_info[3]), None

        possible_moves = board.available_moves(board.white_num_to_colour[board.whites_turn])[0]

        potential_spots = board.get_potential_spots_from_moves(moves=possible_moves, opp=self)

        desired_move_index = None

        if maximizing_whites==1:
            v = float('-inf')
            for j in range(len(potential_spots)):
                cur_board = Board.Field(preset=potential_spots[j], whites_turn=0) # проверить, какой должен быть turn
                alpha_beta_results = self.alpha_beta(cur_board, depth - 1, alpha, beta, 0)
                if v < alpha_beta_results[0]:
                    v = alpha_beta_results[0]
                    alpha = max(alpha, v)
                    desired_move_index = j
                if beta <= alpha:
                    break
            if desired_move_index is None:
                return v, None
            board.bot_test = 0
            return v, possible_moves[desired_move_index]
        else:
            v = float('inf')
            for j in range(len(potential_spots)):
                cur_board = Board.Field(potential_spots[j], whites_turn=1) # проверить, какой должен быть turn
                alpha_beta_results = self.alpha_beta(cur_board, depth - 1, alpha, beta, 1)
                if v > alpha_beta_results[0]:
                    v = alpha_beta_results[0]
                    desired_move_index = j
                    beta = min(beta, v)
                if beta <= alpha:
                    break
            if desired_move_index is None:
                return v, None
            board.bot_test = 0
            return v, possible_moves[desired_move_index]
