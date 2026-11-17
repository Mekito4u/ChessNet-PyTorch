import glob
import os
import shutil

import chess
import chess.pgn
import torch

from Board import board_to_tensor
from Move import move_to_index


def load_all_pgns_from_folder(folder_path):
    all_data = []
    if not os.path.exists(folder_path): return all_data

    pgn_files = glob.glob(os.path.join(folder_path, "*.pgn"))
    for file in pgn_files:
        with open(file, 'r', encoding='utf-8') as f:
            while True:
                game = chess.pgn.read_game(f)
                if game is None: break

                board = game.board()
                result = game.headers.get("Result", "*")
                val_white = 1.0 if result == "1-0" else (-1.0 if result == "0-1" else 0.0)

                for move in game.mainline_moves():
                    state_tensor = board_to_tensor(board)
                    move_idx = move_to_index(move)
                    value = val_white if board.turn == chess.WHITE else -val_white

                    # ОРИГИНАЛ
                    all_data.append((state_tensor, move_idx, value))

                    # ЗЕРКАЛО
                    mirrored_tensor = torch.flip(state_tensor, dims=[2])
                    mirrored_tensor = torch.cat([
                        mirrored_tensor[6:12],
                        mirrored_tensor[0:6],
                        mirrored_tensor[12:13]
                    ], dim=0)
                    mirrored_tensor[12] = 1.0 - mirrored_tensor[12]

                    from_sq = move.from_square
                    to_sq = move.to_square
                    m_from = (from_sq // 8) * 8 + (7 - (from_sq % 8))
                    m_to = (to_sq // 8) * 8 + (7 - (to_sq % 8))
                    mirrored_move_idx = m_from * 64 + m_to

                    all_data.append((mirrored_tensor, mirrored_move_idx, -value))
                    board.push(move)
    return all_data
