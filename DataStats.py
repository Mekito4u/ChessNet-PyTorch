from collections import defaultdict, Counter
import glob
import io
import os
import re
import chess
import chess.pgn

from Openings import ensure_loaded, BY_EPD


DATA_DIR = "data"
N_PLIES = 20


def read_header(path, tag):
    """
    Чтение заголовка .pgn по тегу
    """
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.startswith(f"[{tag}"):
                try:
                    return line.split('"')[1]
                except IndexError:
                    return None
    return None


def match_name(path):
    """game_v6_vs_v6_20261120_153000123456.pgn -> v6_vs_v6."""
    name = os.path.basename(path)
    name = re.sub(r"_\d{8}_\d{6}\d*\.pgn$", "", name)
    name = re.sub(r"^game_", "", name)
    return name if "_vs_" in name else None


def opening_key(game):
    """
    Определение названия дебюта для данной шахматной партии
    """
    uci_moves = []
    node = game
    while node.variations:
        node = node.variations[0]
        if node.move is None:
            break
        uci_moves.append(node.move.uci())

    board = game.board()
    best = None
    for uci in uci_moves[:N_PLIES]:
        move = chess.Move.from_uci(uci)
        if move not in board.legal_moves:
            break
        board.push(move)
        if board.epd() in BY_EPD:
            best = BY_EPD[board.epd()]
    if best:
        return f"{best[0]} {best[1]}"

    b, sans = game.board(), []
    for uci in uci_moves[:6]:
        m = chess.Move.from_uci(uci)
        if m not in b.legal_moves:
            break
        sans.append(b.san(m))
        b.push(m)
    return "Неизвестный дебют: " + (" ".join(sans) if sans else "пустая партия")


def list_games():
    """
    Сбор статистики по всем партиям из DATA_DIR.

    Печатает три отчёта:
      1. Счёт по матчам (победы-ничьи-поражения и очки сторон).
      2. Топ-30 дебютов по всей базе с раскладкой результатов.
      3. Топ-10 дебютов отдельно по каждому матчу.

    Возвращает: (games, scores, opening_counts).
    """

    ensure_loaded()

    games = defaultdict(list)
    scores = defaultdict(lambda: [0, 0, 0])
    opening_counts = Counter()
    opening_scores = defaultdict(lambda: [0, 0, 0])
    opening_by_match = defaultdict(Counter)
    result_map = {"1-0": 0, "1/2-1/2": 1, "0-1": 2}

    for f in glob.glob(os.path.join(DATA_DIR, "*.pgn")):
        name = os.path.basename(f)
        if not name.startswith("game_"):
            continue
        m = match_name(f)
        if m is None:
            continue

        
        with open(f, encoding="utf-8") as pgn_file:
            game = chess.pgn.read_game(pgn_file)
        if game is None:
            continue

        result = game.headers.get("Result", "*")

        games[m].append(f)
        if result in result_map:
            scores[m][result_map[result]] += 1

        
        opening = game.headers.get("Opening") or opening_key(game)

        opening_counts[opening] += 1
        if result in result_map:
            opening_scores[opening][result_map[result]] += 1
        opening_by_match[m][opening] += 1

    print("\nРезультаты")
    for match, (w, dr, bl) in sorted(scores.items()):
        pts_w = w + dr * 0.5
        pts_b = bl + dr * 0.5
        print(f"{match}: {w}-{dr}-{bl}  (белые {pts_w} : {pts_b} чёрные)")

    print(f"\nДебюты (до {N_PLIES} полуходов), всего партий: {sum(opening_counts.values())}")
    for opening, cnt in opening_counts.most_common(30):
        w, dr, bl = opening_scores[opening]
        print(f"  {cnt:4d}x  {opening:<60}  1-0: {w:3d}  ½: {dr:3d}  0-1: {bl:3d}")

    print("\nДебюты по матчам (топ-10 в каждом)")
    for match, cnts in sorted(opening_by_match.items()):
        print(f"\n{match}:")
        for opening, c in cnts.most_common(10):
            print(f"  {c:4d}x  {opening}")

    return games, scores, opening_counts
