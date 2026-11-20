from torch.utils.data import Dataset

class ChessDataset(Dataset):
    """
    Простой датасет: хранит список примеров и отдаёт их по индексу.
    Вся конвертация (в тензоры, таргеты) делается снаружи или в collate_fn.
    """
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
