import torch.nn.functional as F
from torch import nn


class ChessNet(nn.Module):
    def __init__(self):
        super(ChessNet, self).__init__()
        # Свертка
        self.conv1 = nn.Conv2d(13, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(256)

        self.dropout = nn.Dropout(0.2)

        # Политика
        self.policy_conv = nn.Conv2d(256, 32, kernel_size=1)
        self.policy_fc = nn.Linear(32 * 8 * 8, 4096)

        # Оценка
        self.value_fc1 = nn.Linear(256 * 8 * 8, 256)
        self.value_fc2 = nn.Linear(256, 1)

        # Материал
        self.material_fc1 = nn.Linear(256 * 8 * 8, 128)
        self.material_fc2 = nn.Linear(128, 1)

    def forward(self, x):
        # Свертка
        x = F.silu(self.bn1(self.conv1(x)))
        x = F.silu(self.bn2(self.conv2(x)))
        x = F.silu(self.bn3(self.conv3(x)))

        x_flat = x.view(x.size(0), -1)
        x_flat = self.dropout(x_flat)

        # Политика
        p = F.silu(self.policy_conv(x))
        p = p.view(p.size(0), -1)
        policy = self.policy_fc(p)

        # Оценка
        value = F.silu(self.value_fc1(x_flat))
        value = self.value_fc2(value)

        # Материал
        material = F.silu(self.material_fc1(x_flat))
        material = self.material_fc2(material)

        return policy, value, material
