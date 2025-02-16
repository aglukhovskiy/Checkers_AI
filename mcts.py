import math
import random
from copy import deepcopy
import Checkers
import Board
from Checkers import Bot

class Player():
    black = 0
    white = 1

    def other(self):
        return Player.black if self == Player.white else Player.white

class MCTSNode(object):
    def __init__(self, game_state, parent=None, move=None):
        self.game_state = game_state
        self.parent = parent
        self.move = move
        self.win_counts = {
            Player.black: 0,
            Player.white: 0,
        }
        self.num_rollouts = 0
        self.children = []
        self.unvisited_moves = game_state.board.available_moves()[0]
# end::mcts-node[]

# tag::mcts-add-child[]
    def add_random_child(self):
        index = random.randint(0, len(self.unvisited_moves) - 1)
        new_move = self.unvisited_moves.pop(index)
        new_game_state = deepcopy(self.game_state)
        new_game_state.next_turn(new_move)
        new_node = MCTSNode(new_game_state, self, new_move)
        self.children.append(new_node)
        return new_node
# end::mcts-add-child[]

# tag::mcts-record-win[]
    def record_win(self, winner):
        if winner is not None:
            self.win_counts[winner] += 1
        self.num_rollouts += 1
# end::mcts-record-win[]

# tag::mcts-readers[]
    def can_add_child(self):
        return len(self.unvisited_moves) > 0

    def is_terminal(self):
        return False if self.game_state.board.is_game_on==0 else True

    def winning_frac(self, player):
        return float(self.win_counts[player]) / float(self.num_rollouts)
# end::mcts-readers[]


class MCTSAgent():
    def __init__(self, num_rounds, temperature, num_rounds_to_enrich):
        self.num_rounds = num_rounds
        self.temperature = temperature
        self.num_rounds_to_enrich = num_rounds_to_enrich
# tag::mcts-signature[]
    def select_move(self, game_state):
        # print('selecting')
        root = MCTSNode(game_state)
        # print('selecting2')
# end::mcts-signature[]

# tag::mcts-rounds[]
        for i in range(self.num_rounds):
            # print('loop')
            node = root
            while (not node.can_add_child()) and (not game_state.board.game_is_on==0):
                # print('starting while loop')
                node = self.select_child(node)
                # print('while loop')

            # Add a new child node into the tree.
            if node.can_add_child():
                # print('adding child')
                node = node.add_random_child()
                # print('added child')

            # Simulate a random game from this node.

            # print('start simulation')
            winner = self.simulate_random_game(node.game_state)
            # print('winner came - ', winner)
            # print('after simulation')
            # print(winner)
            # Propagate scores back up the tree.
            # print('start root record_win - ', root.win_counts)
            while node is not None:
                node.record_win(winner)
                node = node.parent
            # print('root record_win - ', root.win_counts)
# end::mcts-rounds[]

        scored_moves = [
            (child.winning_frac(game_state.board.whites_turn), child.move, child.num_rollouts)
            for child in root.children
        ]
        scored_moves.sort(key=lambda x: x[0], reverse=True)
        for s, m, n in scored_moves[:10]:
            print('%s - %.3f (%d)' % (m, s, n))

# tag::mcts-selection[]
        # Having performed as many MCTS rounds as we have time for, we
        # now pick a move.
        best_move = None
        best_pct = -1.0
        for child in root.children:
            child_pct = child.winning_frac(game_state.board.whites_turn)
            if child_pct > best_pct:
                best_pct = child_pct
                best_move = child.move
        print('Select move %s with win pct %.3f' % (best_move, best_pct))
        return best_move
# end::mcts-selection[]

# tag::mcts-uct[]
    def select_child(self, node):
        # print('start selecting')
        """Select a child according to the upper confidence bound for
        trees (UCT) metric.
        """
        total_rollouts = sum(child.num_rollouts for child in node.children)
        # print('rollouts - ', total_rollouts)
        # print(node.children)
        # print(node.parent)
        # print(sum(child.num_rollouts for child in node.parent.children))
        log_rollouts = math.log(total_rollouts)

        best_score = -1
        best_child = None
        # Loop over each child.
        for child in node.children:
            # Calculate the UCT score.
            win_percentage = child.winning_frac(node.game_state.board.whites_turn)
            exploration_factor = math.sqrt(log_rollouts / child.num_rollouts)
            uct_score = win_percentage + self.temperature * exploration_factor
            # Check if this is the largest we've seen so far.
            if uct_score > best_score:
                best_score = uct_score
                best_child = child
        return best_child
# end::mcts-uct[]

    @staticmethod
    def simulate_random_game(game):
        simulate_game = deepcopy(game)
        cntr=0
        while simulate_game.board.game_is_on==1 and cntr<=50:
            bot_move = random.choice(simulate_game.board.available_moves()[0])
            simulate_game.next_turn(bot_move)
            cntr+=1
        res = simulate_game.board.get_number_of_pieces_and_kings()

        if res[0]+3*res[2]-res[1]-3*res[3]>0:
            winner=1
        elif res[0]+3*res[2]-res[1]-3*res[3]<0:
            winner=0
        else:
            winner = None

        return winner

    def enrich_stats(self, game_state):
        root = MCTSNode(game_state)
        for i in range(self.num_rounds_to_enrich):
            winner = self.simulate_random_game(root.game_state)
            root.record_win(winner)
        win_pct = root.win_counts[1]/(root.win_counts[1]+root.win_counts[0])
        return win_pct

if __name__ == '__main__':
    f = Board.Field()
    match = Checkers.Checkers(opp='opp', board=f, control='command')
    match.next_turn('e3f4')
    agent = MCTSAgent(num_rounds=10, temperature=3)
    agent.select_move(match)
