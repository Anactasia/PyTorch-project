import torch
import torch.nn as nn
import torch.nn.functional as F

# 3.1 Кастомные СЛОИ

class CustomConv(nn.Module):
    """Кастомный сверточный слой с усреднением по выходным каналам"""
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)

    def forward(self, x):
        out = self.conv(x)
        mean = out.mean(dim=1, keepdim=True)  # Доп. логика: усреднение по каналам
        return out + mean


class ChannelAttention(nn.Module):
    """Channel-wise Attention"""
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class Swish(nn.Module):
    """Кастомная функция активации: x * sigmoid(x)"""
    def forward(self, x):
        return x * torch.sigmoid(x)


class L2Pooling(nn.Module):
    """Кастомный pooling: корень из среднего квадрата"""
    def __init__(self, kernel_size=2, stride=2):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride

    def forward(self, x):
        x_squared = x ** 2
        pooled = F.avg_pool2d(x_squared, self.kernel_size, self.stride)
        return torch.sqrt(pooled + 1e-6)  # избегаем sqrt(0)


class BasicResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if in_channels != out_channels or stride != 1:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + self.shortcut(x))


class BottleneckBlock(nn.Module):
    def __init__(self, in_channels, bottleneck_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, bottleneck_channels, 1)
        self.conv2 = nn.Conv2d(bottleneck_channels, bottleneck_channels, 3, stride, padding=1)
        self.conv3 = nn.Conv2d(bottleneck_channels, bottleneck_channels * 4, 1)
        self.bn1 = nn.BatchNorm2d(bottleneck_channels)
        self.bn2 = nn.BatchNorm2d(bottleneck_channels)
        self.bn3 = nn.BatchNorm2d(bottleneck_channels * 4)

        self.shortcut = nn.Sequential()
        if in_channels != bottleneck_channels * 4 or stride != 1:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, bottleneck_channels * 4, 1, stride),
                nn.BatchNorm2d(bottleneck_channels * 4)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        return F.relu(out + self.shortcut(x))


class WideResidualBlock(nn.Module):
    def __init__(self, in_channels, widen_factor=2, stride=1):
        super().__init__()
        out_channels = in_channels * widen_factor
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if in_channels != out_channels or stride != 1:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + self.shortcut(x))