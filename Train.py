import os
import time
import shutil

import torch
import torch.optim as optim
from torch import nn
from torch.utils.data import DataLoader

from ChessDataset import ChessDataset
from ChessNet import ChessNet
from Load import load_all_pgns_from_folder

def move_files_to_archive():
    # Переносим обученные файлы в архив
    os.makedirs("archive", exist_ok=True)
    files = [f for f in os.listdir("data") if f.endswith(".pgn")]
    for f in files:
        shutil.move(os.path.join("data", f), os.path.join("archive", f))
    print(f"Перемещено в архив: {len(files)} файлов.")


def get_model_path(version):
    """Отдаёт путь к модели по номеру версии: models/v<N>.pth."""
    os.makedirs("models", exist_ok=True)
    return os.path.join("models", f"v{version}.pth")


def train_core(model, dataset, version, epochs=20, batch_size=256, lr=0.001, resume=False):
    """
    Обучение модели: policy — CrossEntropy с label smoothing, value — SmoothL1,
    вес value-лосса x2. AdamW + OneCycleLR + AMP. Каждый эпох сохраняет
    checkpoint для resume, в конце сохраняет models/v<version>.pth.
    """
    time0 = time.time()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
    scaler = torch.amp.GradScaler(device.type)

    start_epoch = 0
    checkpoint_path = f"models/checkpoint_v{version}.pth"
    if resume and os.path.exists(checkpoint_path):
        ckpt = torch.load(checkpoint_path, weights_only=True)
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        scheduler.load_state_dict(ckpt['scheduler'])
        start_epoch = ckpt['epoch']

    policy_loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)
    value_loss_fn = nn.SmoothL1Loss()

    model.train()
    for epoch in range(start_epoch, epochs):
        total_loss = 0.0
        for boards, target_moves, target_values, target_material in dataloader:
            boards = boards.to(device)
            target_moves = target_moves.to(device)
            target_values = target_values.to(device)
            target_material = target_material.to(device)
            optimizer.zero_grad()

            with torch.amp.autocast(device.type):
                pred_policies, pred_values, pred_material = model(boards)

                loss_p = policy_loss_fn(pred_policies, target_moves)
                target_values = target_values.view(-1, 1)
                loss_v = value_loss_fn(pred_values, target_values)

                target_material = target_material.view(-1, 1)
                loss_material = value_loss_fn(pred_material, target_material)

                loss = loss_p + 2.0 * loss_v + 0.5 * loss_material

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            scheduler.step()
            total_loss += loss.item()

        print(f"Эпоха {epoch+1}/{epochs}, Loss: {total_loss/len(dataloader):.4f}, LR: {optimizer.param_groups[0]['lr']:.6f}")
        torch.save({
            'epoch': epoch + 1,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict()
        }, checkpoint_path)

    save_path = get_model_path(version)
    torch.save(model.state_dict(), save_path)
    print(f"Модель сохранена в {save_path}. Время: {time.time() - time0:.1f} сек")
    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)


def train_from_scratch(version, epochs=20, resume=False):
    """
    Полное обучение с нуля на всех партиях из data/. Результат — models/v<version>.pth.
    """
    data = load_all_pgns_from_folder("data")
    if not data:
        print("Нет данных!")
        return
    train_core(ChessNet(), ChessDataset(data), version=version, epochs=epochs, lr=0.001, resume=resume)
