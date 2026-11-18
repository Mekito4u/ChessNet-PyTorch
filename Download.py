import io
import os
import urllib.request

import chess.pgn


def download_lichess_games(username, max_games):
    os.makedirs("data", exist_ok=True)
    os.makedirs("archive", exist_ok=True)

    url = f"https://lichess.org/api/games/user/{username}?max={max_games}&clocks=false&evals=false&opening=true"
    req = urllib.request.Request(url, headers={'User-Agent': 'ChessNet-Trainer'})

    with urllib.request.urlopen(req) as response:
        pgn_text = response.read().decode('utf-8')

    pgn_io = io.StringIO(pgn_text)
    count_saved = 0
    count_skipped = 0
    while True:
        game = chess.pgn.read_game(pgn_io)
        if game is None: break

        utc_date = game.headers.get("UTCDate", "2000.01.01").replace(".", "")
        utc_time = game.headers.get("UTCTime", "000000").replace(":", "")
        base_name = f"game_human_vs_human_{utc_date}_{utc_time}.pgn"

        path_data = os.path.join("data", base_name)
        path_archive = os.path.join("archive", base_name)

        if not os.path.exists(path_data) and not os.path.exists(path_archive):
            with open(path_data, "w", encoding="utf-8") as f:
                print(game, file=f, end="\n\n")
            count_saved += 1
        else:
            count_skipped += 1

    print(f"Скачано: {count_saved} Пропущено: {count_skipped}")
