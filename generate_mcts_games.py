
import Bot
import numpy as np
import Checkers
import Board
from Checkers import Bot
from mcts import MCTSAgent
import Encoder
from utils import move_text_to_point


def generate_game(rounds, max_moves, temperature, num_rounds_to_enrich):
    boards, moves, whites_win_pct_array, whites_turn_array = [], [], [], []  # <1>
    encoder = Encoder.OnePlaneEncoder()
    f = Board.Field()
    game = Checkers.Checkers(opp='opp', board=f, control='command')  # <3>
    bot = MCTSAgent(num_rounds=rounds, temperature=temperature, num_rounds_to_enrich=num_rounds_to_enrich)  # <4>
    num_moves = 0

    while game.board.game_is_on == 1 :
        print(game.board.whites_turn)
        encoder.show_board(game.board)
        move = bot.select_move(game)  # <5>
        move_coors = move_text_to_point(move)
        whites_win_pct = bot.enrich_stats(game)
        whites_win_pct_array.append(whites_win_pct)  # <7>
        print(whites_win_pct)
        boards.append(encoder.encode(game.board)[0])  # <6>
        moves.append(move_coors)  # <7>
        whites_turn_array.append(game.board.whites_turn)  # <7>
        print(move)
        game.next_turn(move)  # <8>
        num_moves += 1
        if num_moves > max_moves:  # <9>
            break

    return (np.array(boards), np.array(whites_win_pct_array), np.array(whites_turn_array))  # <10>

if __name__ == '__main__':
    xs = []
    ys = []
    zs = []
    for i in range(4):
        x, y, z = generate_game(rounds=50, max_moves=20, temperature=3, num_rounds_to_enrich=100)
        xs.append(x)
        ys.append(y)
        zs.append(z)

        x = np.concatenate(xs)  # <3>
        y = np.concatenate(ys)
        z = np.concatenate(zs)

        np.save('features_50r_15m_100_enr.npy', x)  # <4>
        np.save('label_50r_15m_100_enr.npy', y)
        np.save('whites_turn_50r_15m_100_enr.npy', z)
        print('end')
