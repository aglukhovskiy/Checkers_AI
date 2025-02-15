
import Bot
import numpy as np
import Checkers
import Board
from Checkers import Bot
from mcts import MCTSAgent
import Encoder
from utils import move_text_to_point


def generate_game(rounds, max_moves, temperature):
    boards, moves = [], []  # <1>
    encoder = Encoder.OnePlaneEncoder()
    f = Board.Field()
    game = Checkers.Checkers(opp='opp', board=f, control='command')  # <3>
    bot = MCTSAgent(num_rounds=rounds, temperature=temperature)  # <4>
    num_moves = 0

    while game.board.game_is_on == 1 :
        print(game.board.whites_turn)
        encoder.show_board(game.board)
        move = bot.select_move(game)  # <5>
        move_coors = move_text_to_point(move)
        boards.append(encoder.encode(game.board)[0])  # <6>
        moves.append(move_coors)  # <7>
        print(move)
        game.next_turn(move)  # <8>
        num_moves += 1
        if num_moves > max_moves:  # <9>
            break

    return np.array(boards), np.array(moves)  # <10>

if __name__ == '__main__':
    xs = []
    ys = []
    for i in range(2):
        x, y = generate_game(rounds=10, max_moves=15, temperature=3)
        xs.append(x)
        ys.append(y)

    # x = np.concatenate(xs)  # <3>
    # y = np.concatenate(ys)
    #
    # np.save('features.npy', x)  # <4>
    # np.save('label.npy', y)

