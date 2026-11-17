import os

import torch

from ChessDataset import ChessDataset
from ChessNet import ChessNet
from Load import load_all_pgns_from_folder
from Train import train_core, move_files_to_archive


def finetune_model(base_model_path, epochs=5):
    data = load_all_pgns_from_folder("data")
    if not data:
        print("Нет новых данных!")
        return

    model = ChessNet()
    model.load_state_dict(torch.load(base_model_path))

    train_core(model, ChessDataset(data), prefix="finetuned", epochs=epochs, lr=0.0005)
    move_files_to_archive()
