import shutil
from collections import defaultdict
import io
import chess.pgn
import glob
import os

def get_game_files():
    return [f for f in glob.glob("data/*.pgn") + glob.glob("archive/*.pgn")
            if os.path.basename(f).startswith("game_")]

def fix_interrupted_games():
    fixed = 0

    for f in get_game_files():
        with open(f, encoding="utf-8") as pgn:
            text = pgn.read()

        if '[Result "*"]' not in text:
            continue

        text = text.replace('[Result "*"]', '[Result "1/2-1/2"]')
        with open(f, "w", encoding="utf-8") as out:
            out.write(text)

        fixed += 1
        print(f"Исправлено: {f}")

    print(f"\nВсего исправлено: {fixed}")

def dedupe(dry_run=True):
    groups = defaultdict(list)

    for f in get_game_files():
        with open(f, encoding="utf-8") as pgn:
            game = chess.pgn.read_game(pgn)
        if game is None:
            print(f"Не удалось распарсить (пропускаю): {f}")
            continue
        moves = " ".join(m.uci() for m in game.mainline_moves())
        key = (game.headers.get("White", ""), game.headers.get("Black", ""), moves)
        groups[key].append(f)

    moved = 0
    for group in groups.values():
        if len(group) < 2:
            continue
        keep = sorted(group)[0]
        for f in group[1:]:
            print(f"Дубль: {f} (остаётся: {keep})")
            if not dry_run:
                os.makedirs("duplicates", exist_ok=True)
                shutil.move(f, os.path.join("duplicates", os.path.basename(f)))
            moved += 1

    print(f"\nНайдено дублей: {moved}")

def list_games():
    games = defaultdict(list)
    scores = defaultdict(lambda: [0, 0, 0])
    
    for f in glob.glob("data/*.pgn") + glob.glob("archive/*.pgn"):
        name = os.path.basename(f)
        if not name.startswith("game_"):
            continue
            
        parts = name.split("_vs_")
        if len(parts) < 2:
            continue
            
        white = parts[0].replace("game_", "")
        black = parts[1][:-20]
      
        games[f"{white}_vs_{black}"].append(f)

        result = None
        with open(f) as pgn:
            for line in pgn:
                if line.startswith("[Result"):
                    result = line.split('"')[1]
                    break

        if result == "1-0":
            scores[f"{white}_vs_{black}"][0] += 1
        elif result == "1/2-1/2":
            scores[f"{white}_vs_{black}"][1] += 1
        elif result == "0-1":
            scores[f"{white}_vs_{black}"][2] += 1

    print("Кол-во партий")
    print({cat: len(files) for cat, files in games.items()})

    print("\nРезультаты")
    for match, (w, d, b) in sorted(scores.items()):
        pts_w, pts_b = w + d * 0.5, b + d * 0.5
        print(f"{match}: {w}-{d}-{b}  (белые {pts_w} : {pts_b} чёрные)")

    return games, scores

if __name__ == "__main__":
    fix_interrupted_games()
    dedupe(dry_run=False)
    list_games()