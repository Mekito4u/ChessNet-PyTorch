import math


class MCTSNode:
    def __init__(self, board, parent=None, prior_prob=0.0):
        self.board, self.parent, self.children = board, parent, {}
        self.N, self.W, self.Q, self.P = 0, 0.0, 0.0, prior_prob

    def is_expanded(self): return len(self.children) > 0

    def get_value(self, c_puct=1.0):
        u = c_puct * self.P * math.sqrt(self.parent.N) / (1 + self.N)
        return self.Q + u
