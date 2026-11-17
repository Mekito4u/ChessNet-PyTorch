import datetime
import os
import sys
from random import randint

import numpy as np
import chess
import chess.pgn
import torch

from ChessNet import ChessNet
from MCTS import MCTS


def get_best_move_mcts(model, board, num_simulations=200, temperature=0.0, add_noise=False):
    mcts = MCTS(model, num_simulations=num_simulations)
    root = mcts.search(board)
    if root is None or not root.children:
        return list(board.legal_moves)[0], 0.0

    moves = list(root.children.keys())
    visits = np.array([root.children[m].N for m in moves], dtype=np.float64)

    if temperature == 0.0:
        # Жадный выбор
        best_move = moves[np.argmax(visits)]
    else:
        # Выбор с температурой
        visits = visits ** (1.0 / temperature)
        probs = visits / visits.sum()
        best_move = np.random.choice(moves, p=probs)

    best_value = root.children[best_move].Q
    return best_move, best_value


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


def play_with_human(model_path, num_simulations=100):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = ChessNet()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    board = chess.Board()

    print(f"Загружена модель: {model_path}")
    print("Вы играете за БЕЛЫХ.")
    move_num = 1

    while not board.is_game_over(claim_draw=True):
        if move_num <= 15:
            temp = 1.0
            use_noise = True
        else:
            temp = 0.0
            use_noise = False

        if board.turn == chess.WHITE:
            print("\n" + str(board))
            while True:
                move_str = input("Ваш ход: ")
                try:
                    move = chess.Move.from_uci(move_str)
                    if move in board.legal_moves:
                        board.push(move)
                        if board.is_check(): print("ШАХ!")
                        break
                    else:
                        print("Нелегальный ход!")
                except:
                    print("Неверный формат (введите e2e4).")
        else:
            move, val = get_best_move_mcts(
                model, board, 
                num_simulations=num_simulations, 
                temperature=temp, 
                add_noise=use_noise
            )
            board.push(move)
            print(f"ChessNet: {move} (Оценка: {val:.2f})")
            if board.is_check(): print("ChessNet ШАХ!")
        move_num += 1

    print_game_result(board)
    save_game_to_pgn(board, "Human", "ChessNet", "human_vs_network")


def play_self_game(model_path, num_simulations=100, show_motion=False):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = ChessNet()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    board = chess.Board()

    print("Игра ChessNet с самой собой...")
    move_num = 1

    while not board.is_game_over(claim_draw=True):
        if move_num <= 5:
            temp = 1.0
            use_noise = True
        else:
            temp = 0.0
            use_noise = False

        if board.turn == chess.WHITE:
            if show_motion:
              print(f"\n{move_num}.", end=" ")
            move_num += 1

        move, val = get_best_move_mcts(
            model, board, 
            num_simulations=num_simulations, 
            temperature=temp, 
            add_noise=use_noise
        )
        board.push(move)
        if show_motion:
            val_white = val if board.turn == chess.WHITE else -val
            print(f"{move} ({val_white:+.2f})", end=" ")
            if board.is_check(): print("ШАХ!", end=" ")
        
    print("\nИгра окончена!")
    print(board.result(claim_draw=True))
    save_game_to_pgn(board, "ChessNet", "ChessNet", "selfplay")
    return board.result(claim_draw=True)

def play_with_stockfish(model_path, elo=1500, num_simulations=100, show_motion=False):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = ChessNet()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    if sys.platform.startswith("linux"):
        engine_path = "/usr/games/stockfish"
    else:
        engine_path = "stockfish/stockfish-windows-x86-64-universal.exe"
    engine = chess.engine.SimpleEngine.popen_uci(engine_path)
    random_elo = randint(elo-150, elo+150)
    engine.configure({"UCI_LimitStrength": True, "UCI_Elo": random_elo})
    
    
    
    print(f"Игра против Stockfish (Elo: {random_elo}). Вы играете за Белых.")
    board = chess.Board()
    move_num = 1

    while not board.is_game_over(claim_draw=True):
        if move_num <= 15:
            temp = 1.0
            use_noise = True
        else:
            temp = 0.0
            use_noise = False
        
        if board.turn == chess.WHITE:
            move, val = get_best_move_mcts(
                model, board, 
                num_simulations=num_simulations, 
                temperature=temp, 
                add_noise=use_noise
            )
            board.push(move)
            if show_motion:
              if board.is_check(): print("CN ШАХ!", end=" ")
              print(f"{move_num}. {move} ", end="")
            move_num += 1
        else:
            result = engine.play(board, chess.engine.Limit(time=0.5))
            board.push(result.move)
            if show_motion:
              if board.is_check(): print("Sf ШАХ!", end=" ")
              print(f"{result.move} Оценка: {val:.2f}")
        
    engine.quit()
    
    print("\nИгра окончена!")
    print(board.result(claim_draw=True))
    save_game_to_pgn(board, "ChessNet", f"Stockfish_{elo}", "vs_stockfish")
    return board.result(claim_draw=True)
