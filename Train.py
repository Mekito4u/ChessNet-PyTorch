import os
import time

import torch.optim as optim
from sympy.printing.pytorch import torch
from torch import nn
from torch.utils.data import DataLoader

from ChessDataset import ChessDataset
from ChessNet import ChessNet
from Load import load_all_pgns_from_folder


def get_next_path(prefix, folder="models"):
    if not os.path.exists(folder):
        os.makedirs(folder)

    existing = [f for f in os.listdir(folder) if f.startswith(f"{prefix}_v") and f.endswith(".pth")]

    if not existing:
        return os.path.join(folder, f"{prefix}_v1.pth")

    versions = [int(f.split("_v")[1].split(".")[0]) for f in existing]
    next_version = max(versions) + 1

    return os.path.join(folder, f"{prefix}_v{next_version}.pth")


def train_core(model, dataset, prefix, epochs=20, batch_size=256, lr=0.001, resume=False):
    time0 = time.time()
    device = torch.device('cuda')

    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=2, 
        pin_memory=True
    )

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=lr, steps_per_epoch=len(dataloader), epochs=epochs)

    model.to(device)
    scaler = torch.amp.GradScaler('cuda')

    start_epoch = 0
    checkpoint_path = f"models/checkpoint_{prefix}.pth"
    if resume and os.path.exists(checkpoint_path):
        ckpt = torch.load(checkpoint_path)
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        scheduler.load_state_dict(ckpt['scheduler'])
        start_epoch = ckpt['epoch']

    policy_loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)
    value_loss_fn = nn.SmoothL1Loss()

    model.train()
    for epoch in range(start_epoch, epochs):
        total_loss = 0.0
        for boards, target_moves, target_values in dataloader:
            boards, target_moves, target_values = boards.to(device), target_moves.to(device), target_values.to(device)
            optimizer.zero_grad()

            with torch.amp.autocast('cuda'):
                pred_policies, pred_values = model(boards)
                loss_p = policy_loss_fn(pred_policies, target_moves)
                target_values = target_values.view(-1, 1)
                loss_v = value_loss_fn(pred_values, target_values)
                loss = loss_p + 2.0 * loss_v

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            scheduler.step()
            total_loss += loss.item()

        print(f"Эпоха {epoch+1}/{epochs}, Loss: {total_loss/len(dataloader):.4f}, LR: {optimizer.param_groups[0]['lr']:.6f}")
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
        torch.save({
            'epoch': epoch + 1, 
            'model': model.state_dict(), 
            'optimizer': optimizer.state_dict(), 
            'scheduler': scheduler.state_dict()
            }, checkpoint_path
        )

    save_path = get_next_path(prefix)
    torch.save(model.state_dict(), save_path)
    print(f"Модель сохранена в {save_path}. Время: {time.time() - time0:.1f} сек")
    if os.path.exists(checkpoint_path): os.remove(checkpoint_path)


def train_from_scratch(epochs=20, resume=False):
    data = load_all_pgns_from_folder("data") + load_all_pgns_from_folder("archive")
    if not data:
        print("Нет данных!")
        return
    train_core(ChessNet(), ChessDataset(data), prefix="scratch", epochs=epochs, lr=0.001, resume=resume)
