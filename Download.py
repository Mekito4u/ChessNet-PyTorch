import io
import os
import urllib.request

import chess.pgn


def download_lichess_games(username, max_games):
    os.makedirs("data", exist_ok=True)
    url = f"https://lichess.org/api/games/user/{username}?max={max_games}&clocks=false&evals=false&opening=true"
    req = urllib.request.Request(url, headers={'User-Agent': 'ChessNet-Trainer'})

    with urllib.request.urlopen(req) as response:
        pgn_text = response.read().decode('utf-8')

    pgn_io = io.StringIO(pgn_text)
    count = 0
    while True:
        game = chess.pgn.read_game(pgn_io)
        if game is None: break

        utc_date = game.headers.get("UTCDate", "2000.01.01").replace(".", "")
        utc_time = game.headers.get("UTCTime", "000000").replace(":", "")
        filename = f"data/game_human_vs_human_{utc_date}_{utc_time}.pgn"

        if not os.path.exists(filename):
            with open(filename, "w", encoding="utf-8") as f:
                print(game, file=f, end="\n\n")
            count += 1

    print(f"Скачано: {count} Пропущено: {max_games-count}")
