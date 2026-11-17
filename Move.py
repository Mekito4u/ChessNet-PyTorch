import chess


def move_to_index(move):
    return move.from_square * 64 + move.to_square


def index_to_move(index, board):
    from_sq = index // 64
    to_sq = index % 64
    return chess.Move(from_sq, to_sq)
