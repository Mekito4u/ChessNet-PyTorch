import chess
import torch

from Board import board_to_tensor
from MCTSNode import MCTSNode
from Move import move_to_index


class MCTS:
    def __init__(self, model, num_simulations=200, batch_size=16):
        self.model, self.num_simulations, self.batch_size = model, num_simulations, batch_size
        self.model.eval()
        self.device = next(self.model.parameters()).device

    def search(self, board):
        root = MCTSNode(board)
        for _ in range(self.num_simulations // self.batch_size):
            leaves = []
            for _ in range(self.batch_size):
                node, temp_board = root, board.copy()
                while node.is_expanded() and not temp_board.is_game_over(claim_draw=True):
                    best_move = max(node.children.keys(), key=lambda m: node.children[m].get_value())
                    node.children[best_move].N += 1
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
                    policies, vals = self.model(batch_tensor)
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
                        for move in temp.legal_moves:
                            prob = policy[move_to_index(move)].item()
                            node.children[move] = MCTSNode(temp.copy(), parent=node, prior_prob=prob)

                while node is not None:
                    node.N += 1
                    node.W += value
                    node.Q = node.W / node.N
                    value = -value
                    node = node.parent
        return root
