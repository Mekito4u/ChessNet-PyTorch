import glob
import os

from Download import download_lichess_games
from Finetune import finetune_model
from Play import play_self_game, play_with_human
from Train import train_from_scratch


def list_models():
    models = sorted([f for f in glob.glob("models/*.pth") if "checkpoint" not in f])
    if not models: return []
    for i, p in enumerate(models): print(f"{i + 1}. {os.path.basename(p)}")
    return models


def main():
    while True:
        print("\n1. Скачать партии\n2. Обучить с нуля\n3. Дообучить\n4. Сыграть\n5. Self-play\n0. Выход")
        c = input("Выбор: ")
        if c == "0":
            break
        elif c == "1":
            nick = "Epifanovmo"
            count = int(input("Сколько партий: "))
            download_lichess_games(nick, count)
        elif c == "2":
            epochs_cnt = int(input("Кол-во эпох: "))
            train_from_scratch(epochs=epochs_cnt)
        elif c == "3":
            models = list_models()
            if models:
              idx = int(input("Номер модели: ")) - 1
              epochs_cnt = int(input("Кол-во эпох: "))
              finetune_model(models[idx], epochs=epochs_cnt)
            else:
              print("Моделей нет(")
        elif c == "4":
            models = list_models()
            if models:
              idx = int(input("Номер модели: ")) - 1
              play_with_human(models[idx])
            else:
              print("Моделей нет(")
        elif c == "5":
            models = list_models()
            if models:
              idx = int(input("Номер модели: ")) - 1
              play_self_game(models[idx])
            else:
              print("Моделей нет(")


if __name__ == "__main__":
    main()
