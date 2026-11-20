import chess
import torch


def board_to_tensor(board):
    """
    Превращает шахматную доску в тензор 13×8×8.

    Каналы 0–5: белые фигуры (пешка…король).
    Каналы 6–11: чёрные те же фигуры.
    Канал 12: чей ход — 1.0 если белые, иначе 0.0.
    """
    tensor = torch.zeros(13, 8, 8)
    piece_to_channel = {
        chess.PAWN: 0, chess.KNIGHT: 1, chess.BISHOP: 2,
        chess.ROOK: 3, chess.QUEEN: 4, chess.KING: 5
    }

    for piece_type in piece_to_channel:
        for color in (chess.WHITE, chess.BLACK):
            ch = piece_to_channel[piece_type] + (0 if color == chess.WHITE else 6)
            for sq in board.pieces(piece_type, color):
                tensor[ch, 7 - sq // 8, sq % 8] = 1.0
    if board.turn == chess.WHITE:
        tensor[12, :, :] = 1.0
    return tensor
