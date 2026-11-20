import datetime
import os
import sys
from random import randint
import time

import numpy as np
import chess
import chess.pgn
import torch

from ChessNet import ChessNet
from Board import board_to_tensor
from MCTS import MCTS
from Openings import sample_opening


def net_eval(model, board, device):
    """
    Оценка позиции value-head'ом сети с точки зрения ходящего.
    Отдаёт одно число: чем ближе к +1, тем лучше тому, чей сейчас ход.
    """
    with torch.no_grad():
        x = torch.stack([board_to_tensor(board)]).to(device)
        _, vals, _ = model(x)
        return vals.item()


def get_best_move_mcts(model, board, num_simulations, temperature, add_noise=False):
    """
    Запускает MCTS и выбирает ход: temperature=0 — жадно по визитам,
    иначе сэмплирует пропорционально визитам^(1/T). Возвращает (ход, оценка).
    """
    mcts = MCTS(model, num_simulations=num_simulations)
    root = mcts.search(board, add_noise=add_noise)

    if root is None or not root.children:
        return list(board.legal_moves)[0], 0.0

    moves = list(root.children.keys())
    visits = np.array([root.children[m].N for m in moves], dtype=np.float64)

    if temperature == 0.0:
        best_move = moves[np.argmax(visits)]
    else:
        visits = visits ** (1.0 / temperature)
        total = visits.sum()
        if total == 0:
            probs = np.ones(len(moves)) / len(moves)
        else:
            probs = visits / total
        best_move = np.random.choice(moves, p=probs)

    best_value = root.Q
    return best_move, best_value


def save_game_to_pgn(board, white, black, prefix, evals=None, final_result=None,
                     opening_name=None):
    """
    Сохраняет сыгранную партию в data/game_<prefix>_<дата_время>.pgn.
    Результат: переданный final_result, иначе определяется по доске
    (мат / пат / правило 50 ходов / повторение). Оценки MCTS пишутся
    в комментарии [%eval] к каждому ходу.
    """
    game = chess.pgn.Game()
    game.headers["Event"] = f"{white} vs {black}"
    game.headers["Date"] = datetime.datetime.now().strftime("%Y.%m.%d")
    game.headers["White"] = white
    game.headers["Black"] = black
    if opening_name:
        game.headers["Opening"] = opening_name
        game.headers["ECO"] = opening_name.split(" ")[0]

    if final_result is not None:
        game.headers["Result"] = final_result
    elif board.is_game_over(claim_draw=True):
        if board.is_checkmate():
            game.headers["Result"] = "0-1" if board.turn == chess.WHITE else "1-0"
        else:
            game.headers["Result"] = "1/2-1/2"
    else:
        game.headers["Result"] = "*"

    node = game
    for i, move in enumerate(board.move_stack):
        node = node.add_variation(move)
        if evals and i < len(evals):
            val = evals[i]
            node.comment = f"[%eval {val:.2f}]"

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


def play_with_human(model_path, num_simulations=300, max_move_num=0):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = ChessNet()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    board = chess.Board()
    move_num = 1

    print(f"Загружена модель: {model_path}")
    print("Вы играете за БЕЛЫХ.")
    evals_history = []
    while not board.is_game_over(claim_draw=True):
        if move_num <= max_move_num:
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
                        val = net_eval(model, board, device)
                        evals_history.append(val)
                        board.push(move)
                        if board.is_check(): print("ШАХ!")
                        break
                    else:
                        print("Нелегальный ход!")
                except ValueError:
                    print("Неверный формат (введите e2e4).")
        else:
            move, val = get_best_move_mcts(
                model, board, 
                num_simulations=num_simulations, 
                temperature=temp, 
                add_noise=use_noise
            )
            evals_history.append(val)
            board.push(move)
            move_num += 1
            print(f"ChessNet: {move} (Оценка: {val:.2f})")
            if board.is_check(): print("ChessNet ШАХ!")
        

    print_game_result(board)
    model_name = os.path.basename(model_path).replace('.pth', '')
    save_game_to_pgn(board, "Human", "ChessNet", f"human_vs_{model_name}", evals=evals_history)


def play_self_game(model_white_path, model_black_path, num_simulations=100, show_motion=False, max_move_num=0, opening_name=None):
    """
    Партия сети против самой себя (можно разные чекпоинты белыми/чёрными).
    Первые max_move_num ходов — температура 1.0 (разнообразие дебютов),
    дальше — жадная игра. Лимит 100 ходов, иначе техническая ничья.
    Возвращает строку результата и сохраняет PGN.
    """
    time0 = time.time()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model_white = ChessNet()
    model_white.load_state_dict(torch.load(model_white_path, map_location=device))
    model_white.to(device)
    model_white.eval()

    model_black = ChessNet()
    model_black.load_state_dict(torch.load(model_black_path, map_location=device))
    model_black.to(device)
    model_black.eval()

    MAX_MOVES = 100
    pgn_opening = None
    if opening_name:
        board, opening_moves, eco, opening_name = sample_opening(opening_name, n_plies=12)
        tmp = chess.Board()
        san_str = []
        for u in opening_moves:
            san_str.append(tmp.san(chess.Move.from_uci(u)))
            tmp.push(chess.Move.from_uci(u))
        print(f"Дебют: {eco} {opening_name}: {' '.join(san_str)}")
        pgn_opening = f"{eco} {opening_name}"
        max_move_num = max(max_move_num - len(opening_moves) // 2, 0)
    else:
        board = chess.Board()
    move_num = 1

    print("Игра ChessNet с самой собой...")
    evals_history = []
    while not board.is_game_over(claim_draw=True) and move_num < MAX_MOVES:
        if move_num <= max_move_num:
            temp = 1.0
            use_noise = True
        else:
            temp = 0.0
            use_noise = False

        if board.turn == chess.WHITE:
            if show_motion:
              print(f"\n{move_num}.", end=" ")
            move_num += 1
            current_model = model_white
        else:
            current_model = model_black

        move, val = get_best_move_mcts(
            current_model, board, 
            num_simulations=num_simulations, 
            temperature=temp, 
            add_noise=use_noise
        )
        evals_history.append(val)
        board.push(move)

        if show_motion:
            print(f"{move} ({val:+.2f})", end=" ")
            if board.is_check(): print("ШАХ!", end=" ")
        
    print("\nИгра окончена!")
    if board.is_game_over(claim_draw=True):
        result = board.result(claim_draw=True)
    else:
        result = "1/2-1/2"
        print("Достигнут лимит ходов — ничья.")
    print(result)

    time1 = time.time()
    print(f"Заняло: {int(time1-time0)}с, Ходов: {move_num}")


    name_w = os.path.basename(model_white_path).replace('.pth', '')
    name_b = os.path.basename(model_black_path).replace('.pth', '')
    save_game_to_pgn(board, name_w, name_b, f"{name_w}_vs_{name_b}",
                     evals=evals_history, final_result=result,
                     opening_name=pgn_opening)

    return result

def play_with_stockfish(model_path, elo=1500, num_simulations=300, show_motion=False, opening_name=None):
    """
    Матч против Stockfish со случайным Elo в диапазоне ±150 от заданного.
    ChessNet играет белыми, первые 15 ходов с температурой.
    Возвращает результат партии и сохраняет PGN.
    """
    time0 = time.time()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = ChessNet()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    pgn_opening = None

    if sys.platform.startswith("linux"):
        engine_path = "/usr/games/stockfish"
    else:
        engine_path = "stockfish/Stockfish-16.exe"
    engine = chess.engine.SimpleEngine.popen_uci(engine_path)
    random_elo = randint(elo-150, elo+150)
    engine.configure({"UCI_LimitStrength": True, "UCI_Elo": random_elo})
    
    
    if opening_name:
        board, opening_moves, eco, opening_name = sample_opening(opening_name, n_plies=12)
        pgn_opening = f"{eco} {opening_name}"
        max_temp_moves = 15 - (len(opening_moves) + 1) // 2
    else:
        board = chess.Board()
    move_num = 1

    print(f"Игра против Stockfish (Elo: {random_elo}). Вы играете за Белых.")
    if opening_name:
        print(f"Дебют: {opening_name}")

    evals_history = []
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
            evals_history.append(val)
            board.push(move)
            if show_motion:
              if board.is_check(): print("CN ШАХ!", end=" ")
              print(f"{move_num}. {move} ({val:.2f})", end=" ")
            move_num += 1
        else:
            result = engine.play(board, chess.engine.Limit(time=0.5))
            val = net_eval(model, board, device)
            evals_history.append(val)
            board.push(result.move)
            if show_motion:
              if board.is_check(): print("Sf ШАХ!", end=" ")
              print(f"{result.move}")
        
    engine.quit()
    
    print("\nИгра окончена!")
    print(board.result(claim_draw=True))

    time1 = time.time()
    print(f"Заняло: {int(time1-time0)}с")

    name = os.path.basename(model_path).replace('.pth', '')
    save_game_to_pgn(board, "ChessNet", f"Stockfish_{random_elo}",
        f"{name}_vs_stockfish", evals=evals_history, opening_name=pgn_opening)
    return board.result(claim_draw=True)
