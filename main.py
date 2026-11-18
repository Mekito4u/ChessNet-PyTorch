import glob
import os
from collections import defaultdict

from Download import download_lichess_games
from Finetune import finetune_model
from Play import play_self_game, play_with_human, play_with_stockfish
from Train import train_from_scratch


def list_models():
    models = sorted([f for f in glob.glob("models/*.pth") if "checkpoint" not in f])
    if not models: return []
    for i, p in enumerate(models): print(f"{i + 1}. {os.path.basename(p)}")
    return models

def list_games():
    games = defaultdict(list)
    
    for f in glob.glob("data/*.pgn"):
        name = os.path.basename(f)
        if not name.startswith("game_"):
            continue
            
        parts = name.split("_vs_")
        if len(parts) < 2:
            continue
            
        white = parts[0].replace("game_", "")
        black = parts[1][:-20]
        
        if white == black:
            cat = "selfplay"
        elif "stockfish" in white or "stockfish" in black:
            cat = "vs_stockfish"
        else:
            cat = "human_vs_network"
            
        games[cat].append(f)

    print({cat: len(files) for cat, files in games.items()})
    return games

def main():
    while True:
        print("\n1. Скачать партии\n2. Посмотреть партии\n3. Обучить с нуля\n4. Дообучить\n5. Сыграть\n6. Self-play\n7. Stockfish-game\n0. Выход")
        c = input("Выбор: ")
        if c == "0":
            break
        elif c == "1":
            nick = "Epifanovmo"
            count = int(input("Сколько партий: "))
            download_lichess_games(nick, count)
        elif c == "2":
            list_games()
        elif c == "3":
            epochs_cnt = int(input("Кол-во эпох: "))
            train_from_scratch(epochs=epochs_cnt)
        elif c == "4":
            models = list_models()
            if models:
              idx = int(input("Номер модели: ")) - 1
              epochs_cnt = int(input("Кол-во эпох: "))
              finetune_model(models[idx], epochs=epochs_cnt)
            else:
              print("Моделей нет(")
        elif c == "5":
            models = list_models()
            if models:
              idx = int(input("Номер модели: ")) - 1
              play_with_human(models[idx])
            else:
              print("Моделей нет(")
        elif c == "6":
            models = list_models()
            if models:
              idx_w = int(input("Номер модели за БЕЛЫХ: ")) - 1
              idx_b = int(input("Номер модели за ЧЕРНЫХ: ")) - 1
              game_cnt = int(input("Кол-во игр: "))
              games_stats = {}
              show = int(input("Показывать ходы: "))
              for _ in range(game_cnt):
                result = play_self_game(models[idx_w], models[idx_b], show_motion=show, max_move_num=20)
                games_stats[result] = games_stats.get(result, 0) + 1
              for key in games_stats:
                print(f"Итог: {key}, {games_stats[key]}")
            else:
              print("Моделей нет(")
        elif c == "7":
            models = list_models()
            if models:
              idx = int(input("Номер модели: ")) - 1
              game_cnt = int(input("Кол-во игр: "))
              games_stats = {}
              elo = int(input("Elo: "))
              show = int(input("Показывать ходы: "))
              for _ in range(game_cnt):
                result = play_with_stockfish(models[idx], elo=elo, show_motion=show)
                games_stats[result] = games_stats.get(result, 0) + 1
              for key in games_stats:
                print(f"Итог: {key}, {games_stats[key]}")
            else:
              print("Моделей нет(")
        


if __name__ == "__main__":
    main()
