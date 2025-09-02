from eval_pg_bot import  simulate_game
from pg_agent import load_policy_agent
import encoders
from Board_v2 import CheckersGame
import h5py

encoder = encoders.get_encoder_by_name('fiveteenplane')

with h5py.File('models_n_exp/test_model_custom_fiveteenplane_cbv.hdf5', 'r') as agent_file:
    agent1 = load_policy_agent(agent_file)
with h5py.File('models_n_exp/test_model_custom_fiveteenplane_cbv.hdf5', 'r') as agent_file:
    agent2 = load_policy_agent(agent_file)

agent1.moves_list = True
agent2.moves_list = True
agent1.non_relative = True
agent2.non_relative = True

# res = simulate_game(agent1, agent2, 1)
# print(res)


moves = []
game = CheckersGame()

agents = {
    1: agent1,
    -1: agent2,
}
moves_cntr = 0

for i in range(10):
# while game.game_is_on == 1 and moves_cntr < 80:
    moves_cntr += 1
    next_move = agents[game.current_player].select_move(game, 1,
                                                        moves_list=agents[game.current_player].moves_list,
                                                        non_relative=agents[game.current_player].non_relative)
    moves.append(next_move)
    game.next_turn(next_move)
    if game.game_is_on == 0:
        break

encoder.show_board(game)
game_result, game_margin = game.compute_results()

print(game_result, game_margin)

# [ 0.10326408  0.08751197 -0.04020678  0.10074782  0.03135347  0.08765259
#   0.16353805]