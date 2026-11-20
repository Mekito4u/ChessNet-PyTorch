import chess
import torch
import numpy as np

from Board import board_to_tensor
from MCTSNode import MCTSNode
from Move import move_to_index


class MCTS:
    def __init__(self, model, num_simulations, batch_size=32, dirichlet_alpha=0.3, dirichlet_epsilon=0.25):
        self.model = model
        self.num_simulations = num_simulations
        self.batch_size = batch_size
        self.model.eval()
        self.device = next(self.model.parameters()).device

        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_epsilon = dirichlet_epsilon

        self.eval_cache = {}

    def search(self, board, add_noise=False):
        root = MCTSNode(board)
        transposition = {board.fen(): root}

        for _ in range(self.num_simulations // self.batch_size):
            leaves = []
            paths = []

            # Selection
            for _ in range(self.batch_size):
                node = root
                temp_board = board.copy()
                path = [node]

                while node.is_expanded() and not temp_board.is_game_over(claim_draw=True):
                    best_move = max(node.children.keys(), key=lambda m: node.children[m].get_value())
                    node.children[best_move].N += 1.0
                    node.children[best_move].W -= 1.0
                    node = node.children[best_move]
                    temp_board.push(best_move)
                    path.append(node)

                leaves.append((node, temp_board))
                paths.append(path)

            # Evaluation
            states, valid_idx = [], []
            for i, (node, temp) in enumerate(leaves):
                if not temp.is_game_over(claim_draw=True):
                    states.append(board_to_tensor(temp))
                    valid_idx.append(i)

            if states:
                batch_tensor = torch.stack(states).to(self.device)
                with torch.no_grad():
                    policies, vals, _ = self.model(batch_tensor)
                    policies = torch.softmax(policies, dim=1).cpu()
                    vals = vals.cpu()

            # Expansion + Backprop
            for i, (node, temp) in enumerate(leaves):
                if temp.is_game_over(claim_draw=True):
                    if temp.is_checkmate():
                        value = -1.0 if temp.turn == chess.WHITE else 1.0
                    else:
                        value = 0.0
                else:
                    idx = valid_idx.index(i)
                    policy, value = policies[idx], vals[idx].item()

                    if not node.is_expanded():
                        legal_moves = list(temp.legal_moves)

                        if node == root and add_noise:
                            noise = np.random.dirichlet([self.dirichlet_alpha] * len(legal_moves))
                            use_noise = True
                        else:
                            use_noise = False

                        for j, move in enumerate(legal_moves):
                            child_board = temp.copy()
                            child_board.push(move)
                            fen = child_board.fen()

                            prob = policy[move_to_index(move)].item()
                            if use_noise:
                                prob = (1 - self.dirichlet_epsilon) * prob + self.dirichlet_epsilon * noise[j]

                            if fen in transposition:
                                child = transposition[fen]
                            else:
                                child = MCTSNode(child_board, parent=node, prior_prob=prob)
                                transposition[fen] = child

                                if child_board.is_checkmate():
                                    child.N = 10000.0
                                    child.W = 10000.0
                                    child.Q = 1.0

                            node.children[move] = child

                # Backprop по пути
                for n in reversed(paths[i]):
                    n.N += 1
                    n.W += value
                    n.Q = n.W / n.N
                    value = -value

        return root
