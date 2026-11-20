import os

from Cleanup import dedupe, remove_interrupted_games
from Download import download_lichess_games
from Finetune import finetune_model
from Play import play_self_game, play_with_human, play_with_stockfish
from Train import get_model_path, train_from_scratch
from DataStats import list_games


def ask_int(prompt):
    return int(input(prompt))


def ask_model(prompt="Номер версии модели: "):
    """Спрашивает номер версии и возвращает путь models/v<N>.pth. None, если файла нет."""
    version = ask_int(prompt)
    path = get_model_path(version)
    if not os.path.exists(path):
        print(f"Модели v{version} нет — есть только эти:")
        list_model_files()
        return None
    return path


def list_model_files():
    models = sorted(f for f in os.listdir("models")
                    if f.endswith(".pth") and not f.startswith("checkpoint"))
    if not models:
        print("Пока нет ни одной модели.")
        return
    for i, name in enumerate(models, 1):
        print(f"  {i}. {name}")


def run_games(play_fn, path):
    """Прогоняет play_fn нужное число раз и печатает статистику результатов."""
    game_cnt = ask_int("Кол-во игр: ")
    stats = {}
    for _ in range(game_cnt):
        result = play_fn(path)
        stats[result] = stats.get(result, 0) + 1
    print("\nИтоги:")
    for key, cnt in stats.items():
        print(f"  {key}: {cnt}")


def main():
    while True:
        print("\n1. Скачать партии       5. Сыграть с человеком")
        print("2. Посмотреть партии    6. Self-play (серия)")
        print("3. Обучить с нуля       7. Против Stockfish (серия)")
        print("4. Дообучить модель     8. Чистка данных (дубли/недоигранные)")
        print("0. Выход")
        choice = input("Выбор: ")

        if choice == "0":
            break

        elif choice == "1":
            nick = "Epifanovmo"
            count = ask_int("Сколько партий: ")
            download_lichess_games(nick, count)

        elif choice == "2":
            list_games()

        elif choice == "3":
            version = ask_int("Версия новой модели (например 1): ")
            epochs_cnt = ask_int("Кол-во эпох: ")
            train_from_scratch(version=version, epochs=epochs_cnt)

        elif choice == "4":
            base_version = ask_int("Дообучить модель версии: ")
            new_version = ask_int("Сохранить как версию: ")
            epochs_cnt = ask_int("Кол-во эпох: ")
            finetune_model(base_version, new_version, epochs=epochs_cnt)

        elif choice == "5":
            path = ask_model()
            if path:
                play_with_human(path)

        elif choice == "6":
            path_w = ask_model("Версия за БЕЛЫХ: ")
            if not path_w:
                continue
            path_b = ask_model("Версия за ЧЁРНЫХ: ")
            if not path_b:
                continue
            show = input("Показывать ходы [y/N]: ").strip().lower() == "y"
            run_games(lambda p: play_self_game(path_w, path_b, show_motion=show, max_move_num=10), path_w)

        elif choice == "7":
            path = ask_model()
            if not path:
                continue
            elo = ask_int("Elo Stockfish: ")
            show = input("Показывать ходы [y/N]: ").strip().lower() == "y"
            run_games(lambda p: play_with_stockfish(p, elo=elo, show_motion=show), path)

        elif choice == "8":
            print("\n1. Найти и удалить дубли\n2. Удалить недоигранные партии")
            sub = input("Выбор: ")
            if sub == "1":
                dry = input("Сначала посмотреть (без удаления)? [Y/n]: ").strip().lower() != "n"
                dedupe(dry_run=dry)
            elif sub == "2":
                remove_interrupted_games()

        else:
            print("Нет такой команды.")


if __name__ == "__main__":
    main()
