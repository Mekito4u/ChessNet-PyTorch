import chess


def move_to_index(move):
    """
    Кодирует ход одним числом: from*64 + to. Диапазон 0..4095.
    """
    return move.from_square * 64 + move.to_square


def index_to_move(index, board):
    """
    Обратно: делит на 64 — получаем from, остаток — to.
    """
    from_sq = index // 64
    to_sq = index % 64
    return chess.Move(from_sq, to_sq)
