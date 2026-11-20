import chess
import torch
import numpy as np

from Board import board_to_tensor
from MCTSNode import MCTSNode
from Move import move_to_index


class MCTS:
    def __init__(self, model, num_simulations=200, batch_size=16, dirichlet_alpha=0.3, dirichlet_epsilon=0.25):
        self.model, self.num_simulations, self.batch_size = model, num_simulations, batch_size
        self.model.eval()
        self.device = next(self.model.parameters()).device

        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_epsilon = dirichlet_epsilon

    def search(self, board, add_noise=False):
        root = MCTSNode(board)
        for _ in range(self.num_simulations // self.batch_size):
            leaves = []
            for _ in range(self.batch_size):
                node, temp_board = root, board.copy()
                while node.is_expanded() and not temp_board.is_game_over(claim_draw=True):
                    best_move = max(node.children.keys(), key=lambda m: node.children[m].get_value())
                    node.children[best_move].N += 1.0
                    node.children[best_move].W -= 1.0
                    node = node.children[best_move]
                    temp_board.push(best_move)
                leaves.append((node, temp_board))

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
                            for j, move in enumerate(legal_moves):
                                    p_net = policy[move_to_index(move)].item()
                                    p_mixed = (1 - self.dirichlet_epsilon) * p_net + self.dirichlet_epsilon * noise[j]
                                    node.children[move] = MCTSNode(temp.copy(), parent=node, prior_prob=p_mixed)
                        else:
                            for move in legal_moves:
                                prob = policy[move_to_index(move)].item()
                                node.children[move] = MCTSNode(temp.copy(), parent=node, prior_prob=prob)

                while node is not None:
                    node.N += 1
                    node.W += value
                    node.Q = node.W / node.N
                    value = -value
                    node = node.parent
        return root
