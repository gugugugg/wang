# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F


if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True


class Swish(nn.Module):
    """Swish/SiLU activation: f(x) = x * sigmoid(x)."""

    def forward(self, x):
        return x * torch.sigmoid(x)


class SEBlock(nn.Module):
    """Squeeze-and-Excitation channel attention block."""

    def __init__(self, channel, reduction=16):
        super().__init__()
        hidden = max(1, channel // reduction)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, hidden, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, channel, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class ResBlock(nn.Module):
    """Residual block with Swish activation and SE attention."""

    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.act1 = Swish()
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.act2 = Swish()
        self.se = SEBlock(channels)

    def forward(self, x):
        residual = x
        out = self.act1(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.se(out)
        out += residual
        return self.act2(out)


class GobangDualHead(nn.Module):
    """AlphaZero-style dual-head policy/value network for 15x15 Gomoku."""

    def __init__(self, num_filters=64, num_blocks=5, board_size=15):
        super().__init__()
        self.board_size = board_size
        self.num_filters = num_filters
        self.num_blocks = num_blocks

        self.conv_input = nn.Conv2d(3, num_filters, kernel_size=3, padding=1, bias=False)
        self.bn_input = nn.BatchNorm2d(num_filters)
        self.act_input = Swish()
        self.res_blocks = nn.ModuleList([ResBlock(num_filters) for _ in range(num_blocks)])

        self.conv_p = nn.Conv2d(num_filters, 4, kernel_size=1)
        self.bn_p = nn.BatchNorm2d(4)
        self.act_p = Swish()
        self.fc_p = nn.Linear(4 * board_size * board_size, board_size * board_size)

        self.conv_v = nn.Conv2d(num_filters, 2, kernel_size=1)
        self.bn_v = nn.BatchNorm2d(2)
        self.act_v = Swish()
        self.fc_v1 = nn.Linear(2 * board_size * board_size, 64)
        self.act_v_fc = Swish()
        self.fc_v2 = nn.Linear(64, 1)

    def forward(self, x):
        out = self.act_input(self.bn_input(self.conv_input(x)))
        for block in self.res_blocks:
            out = block(out)

        p = self.act_p(self.bn_p(self.conv_p(out)))
        p = p.view(p.size(0), -1)
        p = F.log_softmax(self.fc_p(p), dim=1)

        v = self.act_v(self.bn_v(self.conv_v(out)))
        v = v.view(v.size(0), -1)
        v = self.act_v_fc(self.fc_v1(v))
        v = torch.tanh(self.fc_v2(v))
        return p, v
