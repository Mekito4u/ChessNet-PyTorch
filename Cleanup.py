import shutil
from collections import defaultdict
import io
import chess.pgn
import glob
import os

def get_game_files():
    """Все партии из data/ — только файлы game_*."""
    return [f for f in os.listdir("data")
            if f.startswith("game_") and f.endswith(".pgn")]

def remove_interrupted_games():
    """Удаляет партии с Result \"*\" — недоигранные, в датасете от них больше вреда."""
    removed = 0
    for f in get_game_files():
        path = os.path.join("data", f)
        with open(path, encoding="utf-8") as pgn:
            text = pgn.read()
        if '[Result "*"]' not in text:
            continue
        if input(f"Удалить недоигранную {f}? [y/N] ").strip().lower() == "y":
            os.remove(path)
            removed += 1
    print(f"\nУдалено: {removed}")


def dedupe(dry_run=True):
    """
    Удаляет партии с полностью совпадающими ходами: ключ — строка UCI-ходов.
    Из группы дублей остаётся первый файл по алфавиту, остальные удаляются.
    dry_run=True — только показать, ничего не удалять.
    """
    groups = defaultdict(list)

    for f in get_game_files():
        path = os.path.join("data", f)
        with open(path, encoding="utf-8") as pgn:
            game = chess.pgn.read_game(pgn)
        if game is None:
            print(f"Не удалось распарсить (пропускаю): {f}")
            continue
        moves = " ".join(m.uci() for m in game.mainline_moves())
        groups[moves].append(path)

    deleted = 0
    for moves, group in groups.items():
        if len(group) < 2:
            continue
        keep = sorted(group)[0]
        for f in group[1:]:
            print(f"Дубль: {f} (остаётся: {keep})")
            if not dry_run:
                os.remove(f)
            deleted += 1

    print(f"\n{'Удалили бы' if dry_run else 'Удалено'} дублей: {deleted}")
