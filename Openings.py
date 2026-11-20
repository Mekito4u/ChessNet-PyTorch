import glob
import io
import os
import random
import chess
import chess.pgn

BOOK_DIR = "chess-openings"

LINES = []
BY_EPD = {}
_loaded = False


def load_book():
    """
    Загрузка базы дебютов. Один проход по файлам на весь проект.
    """
    global _loaded
    files = sorted(glob.glob(os.path.join(BOOK_DIR, "[a-e].tsv")))
    if not files:
        print(f"[!] Файлы a-e.tsv не найдены в '{BOOK_DIR}'")
        return
    for path in files:
        with open(path, encoding="utf-8") as f:
            cols = f.readline().rstrip("\n").split("\t")
            i_eco, i_name, i_pgn = cols.index("eco"), cols.index("name"), cols.index("pgn")
            for line in f:
                p = line.rstrip("\n").split("\t")
                if len(p) <= i_pgn or p[i_pgn] == "*":
                    continue
                game = chess.pgn.read_game(io.StringIO(p[i_pgn]))
                if game is None or game.errors:
                    continue
                board = game.board()
                uci_moves = []
                for node in game.mainline():
                    if node.move is None or node.move not in board.legal_moves:
                        break
                    board.push(node.move)
                    uci_moves.append(node.move.uci())
                    # помечаем каждую позицию линии, а не только последнюю
                    BY_EPD.setdefault(board.epd(), (p[i_eco], p[i_name]))
                if uci_moves:
                    LINES.append((uci_moves, p[i_eco], p[i_name]))
    _loaded = True
    print(f"[i] Загружено {len(LINES)} линий, {len(BY_EPD)} позиций из книги дебютов")


def ensure_loaded():
    """
    Проверка на загрузку бд
    """
    if not _loaded:
        load_book()


def list_openings(substr=""):
    """
    Имена дебютов с подстрокой — для меню.
    """
    ensure_loaded()
    s = substr.lower()
    return sorted({name for _, _, name in LINES if s in name.lower()})


def sample_opening(name, n_plies=12):
    """
    Доска после n_plies полуходов случайной линии дебюта name.
    Возвращает (board, uci_moves, eco, name).
    Среди линий предпочитает глубокие (от 8 полуходов), чтобы не ловить
    короткие префиксы вроде голого 'Scotch Game' в 5 полуходов.
    """
    ensure_loaded()
    variants = [(m, e) for m, e, n in LINES if n == name]
    if not variants:
        raise ValueError(f"Дебют '{name}' не найден в книге")
    deep = [(m, e) for m, e in variants if len(m) >= 8]
    uci_moves, eco = random.choice(deep if deep else variants)
    uci_moves = uci_moves[:n_plies]
    board = chess.Board()
    for u in uci_moves:
        board.push(chess.Move.from_uci(u))
    return board, uci_moves, eco, name