import os
import shutil

import torch

from ChessDataset import ChessDataset
from ChessNet import ChessNet
from Load import load_all_pgns_from_folder
from Train import train_core


def move_files_to_archive():
    # Переносим обученные файлы в архив
    os.makedirs("archive", exist_ok=True)
    files = [f for f in os.listdir("data") if f.endswith(".pgn")]
    for f in files:
        shutil.move(os.path.join("data", f), os.path.join("archive", f))
    print(f"Перемещено в архив: {len(files)} файлов.")


def finetune_model(base_model_path, epochs=5):
    data = load_all_pgns_from_folder("data")
    if not data:
        print("Нет новых данных!")
        return

    model = ChessNet()
    model.load_state_dict(torch.load(base_model_path))

    train_core(model, ChessDataset(data), prefix="finetuned", epochs=epochs, lr=0.0005)
    move_files_to_archive()
