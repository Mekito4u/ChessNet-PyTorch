import glob
import os
import shutil
import re
import math

import chess
import chess.pgn
import torch

from Board import board_to_tensor
from Move import move_to_index

def parse_eval(comment):
  match = re.search(r'\[%eval\s+(-?\d*\.?\d+)\]', comment)
  if match:
        return math.tanh(float(match.group(1)) / 2.0)
  match_mate = re.search(r'\[%eval\s+#(-?\d+)\]', comment)
  if match_mate:
      return 1.0 if int(match_mate.group(1)) > 0 else -1.0
  return None


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

                node = game
                while node.variations:
                    next_node = node.variation(0)
                    move = next_node.move

                    state_tensor = board_to_tensor(board)
                    move_idx = move_to_index(move)

                    mcts_eval = parse_eval(next_node.comment)
                    if mcts_eval is not None:
                        value = mcts_eval if board.turn == chess.WHITE else -mcts_eval
                    else:
                        value = val_white if board.turn == chess.WHITE else -val_white

                    # ОРИГИНАЛ
                    all_data.append((state_tensor, move_idx, value))

                    # ЗЕРКАЛО
                    mirrored_board = board.mirror()
                    mirrored_tensor = board_to_tensor(mirrored_board)

                    from_sq, to_sq = move.from_square, move.to_square
                    m_from = chess.square_mirror(from_sq)
                    m_to = chess.square_mirror(to_sq)
                    mirrored_move = chess.Move(m_from, m_to, promotion=move.promotion)
                    mirrored_move_idx = move_to_index(mirrored_move)


                    all_data.append((mirrored_tensor, mirrored_move_idx, -value))

                    board.push(move)
                    node = next_node
    return all_data
