import datetime
import os

import chess
import chess.pgn
import torch

from ChessNet import ChessNet
from MCTS import MCTS


def get_best_move_mcts(model, board, num_simulations=200):
    mcts = MCTS(model, num_simulations=num_simulations)
    root = mcts.search(board)
    if root is None or not root.children:
        return list(board.legal_moves)[0], 0.0
    best_move = max(root.children.keys(), key=lambda m: root.children[m].N)
    return best_move, root.children[best_move].Q


def save_game_to_pgn(board, white, black, prefix):
    game = chess.pgn.Game()
    game.headers["Event"] = f"{white} vs {black}"
    game.headers["Date"] = datetime.datetime.now().strftime("%Y.%m.%d")
    game.headers["White"] = white
    game.headers["Black"] = black

    if board.is_game_over(claim_draw=True):
        if board.is_checkmate():
            game.headers["Result"] = "0-1" if board.turn == chess.WHITE else "1-0"
        else:
            game.headers["Result"] = "1/2-1/2"
    else:
        game.headers["Result"] = "*"

    node = game
    for move in board.move_stack:
        node = node.add_variation(move)

    if not os.path.exists("data"): os.makedirs("data")
    filename = f"data/game_{prefix}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.pgn"
    with open(filename, "w", encoding="utf-8") as f:
        print(game, file=f, end="\n\n")
    print(f"Партия сохранена в {filename}")


def print_game_result(board):
    print("\n" + str(board))
    print("Игра окончена!")
    if board.is_checkmate():
        winner = "Чёрные" if board.turn == chess.WHITE else "Белые"
        print(f"МАТ! Победили {winner}!")
    elif board.is_stalemate():
        print("ПАТ (ничья).")
    elif board.is_insufficient_material():
        print("НИЧЬЯ. Недостаточно фигур.")
    elif board.can_claim_threefold_repetition():
        print("НИЧЬЯ. Троекратное повторение.")
    elif board.can_claim_fifty_moves():
        print("НИЧЬЯ. Правило 50 ходов.")
    else:
        print(f"Результат: {board.result()}")


def play_with_human(model_path):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = ChessNet()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    board = chess.Board()

    print(f"Загружена модель: {model_path}")
    print("Вы играете за БЕЛЫХ.")

    while not board.is_game_over(claim_draw=True):
        if board.turn == chess.WHITE:
            print("\n" + str(board))
            if board.is_check(): print("ШАХ!")
            while True:
                move_str = input("Ваш ход: ")
                try:
                    move = chess.Move.from_uci(move_str)
                    if move in board.legal_moves:
                        board.push(move)
                        break
                    else:
                        print("Нелегальный ход!")
                except:
                    print("Неверный формат (введите e2e4).")
        else:
            print("Сеть думает...")
            move, val = get_best_move_mcts(model, board)
            board.push(move)
            print(f"Сеть походила: {move} (Оценка: {val:.2f})")
            if board.is_check(): print("СЕТЬ ОБЪЯВИЛА ШАХ!")

    print_game_result(board)
    save_game_to_pgn(board, "Human", "ChessNet", "human_vs_network")


def play_self_game(model_path):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = ChessNet()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    board = chess.Board()

    print("Игра сети с самой собой...")
    while not board.is_game_over(claim_draw=True):
        move, val = get_best_move_mcts(model, board)
        board.push(move)
        print(f"{move} (Оценка: {val:.2f})", end=" ")
    print("\n")
    print_game_result(board)
    save_game_to_pgn(board, "ChessNet", "ChessNet", "selfplay")
