# -*- coding: utf-8 -*-
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SpikeFn(torch.autograd.Function):
    """Binary spike with a fast-sigmoid surrogate gradient."""

    @staticmethod
    def forward(ctx, x: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(x)
        return (x > 0).to(x.dtype)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        (x,) = ctx.saved_tensors
        grad = 1.0 / (1.0 + x.abs()).pow(2)
        return grad_output * grad


spike_fn = SpikeFn.apply


class LIFLayer(nn.Module):
    """Leaky integrate-and-fire wrapper around a feed-forward module."""

    def __init__(self, module: nn.Module, tau: float = 0.25, threshold: float = 1.0):
        super().__init__()
        self.module = module
        self.tau = tau
        self.threshold = threshold

    def forward(self, x: torch.Tensor, membrane: torch.Tensor | None) -> tuple[torch.Tensor, torch.Tensor]:
        current = self.module(x)
        if membrane is None:
            membrane = torch.zeros_like(current)
        membrane = membrane * self.tau + current
        spike = spike_fn(membrane - self.threshold)
        membrane = membrane * (1.0 - spike.detach())
        return spike, membrane


class SpikingResBlock(nn.Module):
    """Residual block using LIF activations across simulation steps."""

    def __init__(self, channels: int, tau: float = 0.25, threshold: float = 1.0):
        super().__init__()
        self.lif1 = LIFLayer(
            nn.Sequential(
                nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(channels),
            ),
            tau=tau,
            threshold=threshold,
        )
        self.lif2 = LIFLayer(
            nn.Sequential(
                nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(channels),
            ),
            tau=tau,
            threshold=threshold,
        )

    def forward(
        self,
        x: torch.Tensor,
        mem1: torch.Tensor | None,
        mem2: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        out, mem1 = self.lif1(x, mem1)
        out, mem2 = self.lif2(out, mem2)
        return spike_fn(out + x - 0.5), mem1, mem2


class GobangSNNDualHead(nn.Module):
    """LIF-based dual-head SNN for 15x15 Gomoku policy and value prediction."""

    def __init__(
        self,
        num_filters: int = 64,
        num_blocks: int = 3,
        board_size: int = 15,
        time_steps: int = 6,
        tau: float = 0.25,
        threshold: float = 1.0,
    ):
        super().__init__()
        self.board_size = board_size
        self.num_filters = num_filters
        self.num_blocks = num_blocks
        self.time_steps = time_steps
        self.tau = tau
        self.threshold = threshold

        self.input_lif = LIFLayer(
            nn.Sequential(
                nn.Conv2d(3, num_filters, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(num_filters),
            ),
            tau=tau,
            threshold=threshold,
        )
        self.res_blocks = nn.ModuleList([SpikingResBlock(num_filters, tau=tau, threshold=threshold) for _ in range(num_blocks)])

        self.conv_p = nn.Conv2d(num_filters, 4, kernel_size=1)
        self.bn_p = nn.BatchNorm2d(4)
        self.fc_p = nn.Linear(4 * board_size * board_size, board_size * board_size)

        self.conv_v = nn.Conv2d(num_filters, 2, kernel_size=1)
        self.bn_v = nn.BatchNorm2d(2)
        self.fc_v1 = nn.Linear(2 * board_size * board_size, 64)
        self.fc_v2 = nn.Linear(64, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        input_mem = None
        block_mems: list[tuple[torch.Tensor | None, torch.Tensor | None]] = [(None, None) for _ in self.res_blocks]
        spike_acc = None

        for _ in range(self.time_steps):
            out, input_mem = self.input_lif(x, input_mem)
            next_mems = []
            for block, (mem1, mem2) in zip(self.res_blocks, block_mems):
                out, mem1, mem2 = block(out, mem1, mem2)
                next_mems.append((mem1, mem2))
            block_mems = next_mems
            spike_acc = out if spike_acc is None else spike_acc + out

        features = spike_acc / float(self.time_steps)

        p = spike_fn(self.bn_p(self.conv_p(features)) - 0.5)
        p = p.view(p.size(0), -1)
        p = F.log_softmax(self.fc_p(p), dim=1)

        v = spike_fn(self.bn_v(self.conv_v(features)) - 0.5)
        v = v.view(v.size(0), -1)
        v = F.silu(self.fc_v1(v))
        v = torch.tanh(self.fc_v2(v))
        return p, v
