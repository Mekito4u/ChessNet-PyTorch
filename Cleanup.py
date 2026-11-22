import shutil
from collections import defaultdict
import io
import chess.pgn
import glob
import os
import re

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


def rename_duplicate_games(apply=False):
    """
    Переименовывает файлы партий с суффиксом Windows ' (N)':
    убирает ' (1)', а если имя занято — поднимает последнюю цифру
    таймстампа, пока не найдёт свободное.

    apply=False — только предпросмотр, apply=True — переименовать.
    Возвращает список пар (старое_имя, новое_имя).
    """
    pattern = re.compile(
        r"^(?P<base>.+)_(?P<ts>\d{8}_\d{6})(?P<suffix> $\d+$)?(?P<ext>\.pgn)$",
        re.IGNORECASE,
    )

    folder = "data"
    renames = []
    for name in os.listdir(folder):
        m = pattern.match(name)
        if not m:
            continue
        base, ts, ext = m.group("base"), m.group("ts"), m.group("ext")

        
        if m.group("suffix") is None:
            continue

        new_name = f"{base}_{ts}{ext}"

        if not os.path.exists(os.path.join(folder, new_name)) and \
           not any(r[1] == new_name for r in renames):
            renames.append((name, new_name))
            continue

        
        for digit in range(1, 10):
            ts_new = ts[:-1] + str((int(ts[-1]) + digit) % 10)
            candidate = f"{base}_{ts_new}{ext}"
            if not os.path.exists(os.path.join(folder, candidate)) and \
               not any(r[1] == candidate for r in renames):
                new_name = candidate
                break
        else:
            print(f"  !! Не нашлось свободного имени для {name} — пропускаю")
            continue

        renames.append((name, new_name))

    print(f"Найдено для переименования: {len(renames)}\n")
    for old, new in renames:
        print(f"  {old}\n    -> {new}")

    if not renames:
        print("Ничего делать не нужно.")
    elif apply:
        for old, new in renames:
            os.rename(os.path.join(folder, old), os.path.join(folder, new))
        print("\nГотово!")
    else:
        print("\nЭто предпросмотр. Вызови с apply=True, чтобы применить.")

    return renames
