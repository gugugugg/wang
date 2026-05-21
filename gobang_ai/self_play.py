# -*- coding: utf-8 -*-
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import csv
import random
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from gobang_ai.env import GobangEnv
from gobang_ai.features import ACTION_SIZE, BOARD_SIZE, augment_state_policy, policy_one_hot, state_to_feature
from gobang_ai.model import GobangDualHead
from gobang_ai.snn_model import GobangSNNDualHead
from gobang_ai.training import get_device


@dataclass
class SelfPlaySample:
    state: np.ndarray
    pi: np.ndarray
    player: int
    outcome: float = 0.0


class SelfPlayDataset(Dataset):
    def __init__(self, samples: list[SelfPlaySample], augment: bool = True, board_size: int = BOARD_SIZE):
        self.samples = samples
        self.augment = augment
        self.board_size = board_size

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        state = sample.state
        pi = sample.pi
        if self.augment:
            mode = np.random.randint(0, 8)
            state, pi = augment_state_policy(state, pi, mode, self.board_size)
        return (
            torch.from_numpy(state.copy()),
            torch.from_numpy(pi.copy()),
            torch.tensor(sample.outcome, dtype=torch.float32),
        )


def load_model_from_checkpoint(checkpoint_path: str | Path, device: torch.device):
    checkpoint_path = Path(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint.get("model_config", {})
    architecture = config.get("architecture", "dual_head_resnet")
    filters = int(config.get("num_filters", 64))
    blocks = int(config.get("num_blocks", 5))
    board_size = int(config.get("board_size", BOARD_SIZE))
    if architecture == "snn_lif":
        model = GobangSNNDualHead(
            num_filters=filters,
            num_blocks=blocks,
            board_size=board_size,
            time_steps=int(config.get("time_steps", 6)),
            tau=float(config.get("tau", 0.25)),
            threshold=float(config.get("threshold", 1.0)),
        ).to(device)
    else:
        model = GobangDualHead(num_filters=filters, num_blocks=blocks, board_size=board_size).to(device)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    return model, checkpoint, dict(config) | {"num_filters": filters, "num_blocks": blocks, "board_size": board_size}


def active_valid_moves(board: np.ndarray, distance: int = 2) -> np.ndarray:
    rows, cols = np.where(board != 0)
    if len(rows) == 0:
        center = board.shape[0] // 2
        return np.array([center * board.shape[0] + center], dtype=np.int64)

    active: set[int] = set()
    size = board.shape[0]
    for r, c in zip(rows, cols):
        for dr in range(-distance, distance + 1):
            for dc in range(-distance, distance + 1):
                nr, nc = int(r + dr), int(c + dc)
                if 0 <= nr < size and 0 <= nc < size and board[nr, nc] == 0:
                    active.add(nr * size + nc)
    if active:
        return np.array(sorted(active), dtype=np.int64)
    return np.flatnonzero(board == 0).astype(np.int64)


def sample_policy_action(
    model: GobangDualHead,
    board: np.ndarray,
    current_player: int,
    device: torch.device,
    temperature: float = 1.0,
) -> tuple[int, np.ndarray]:
    valid_moves = active_valid_moves(board)
    state = state_to_feature(board, current_player)
    tensor = torch.from_numpy(state).unsqueeze(0).to(device)
    with torch.no_grad():
        log_policy, _ = model(tensor)
    logits = log_policy[0].detach().cpu().numpy()
    masked = logits[valid_moves]
    if temperature <= 1e-6:
        action = int(valid_moves[int(np.argmax(masked))])
    else:
        scaled = masked / temperature
        scaled = scaled - np.max(scaled)
        probs = np.exp(scaled)
        probs = probs / probs.sum()
        action = int(np.random.choice(valid_moves, p=probs))
    return action, policy_one_hot(action, ACTION_SIZE)


def generate_self_play_game(
    model: GobangDualHead,
    device: torch.device,
    temperature: float = 1.0,
    max_moves: int = ACTION_SIZE,
) -> tuple[list[SelfPlaySample], int]:
    env = GobangEnv()
    board = env.reset(randomize_opening=True)
    samples: list[SelfPlaySample] = []
    winner = 0

    for _ in range(max_moves):
        player = env.current_player
        state = state_to_feature(board, player)
        action, pi = sample_policy_action(model, board, player, device, temperature=temperature)
        samples.append(SelfPlaySample(state=state, pi=pi, player=player))
        board, reward, done = env.step(action)
        if done:
            if reward > 0:
                winner = player
            break

    for sample in samples:
        if winner == 0:
            sample.outcome = 0.0
        else:
            sample.outcome = 1.0 if sample.player == winner else -1.0
    return samples, winner


def train_on_self_play_samples(
    model: GobangDualHead,
    samples: list[SelfPlaySample],
    device: torch.device,
    batch_size: int,
    lr: float,
    weight_decay: float,
    epochs: int,
    num_workers: int = 0,
):
    dataset = SelfPlayDataset(samples, augment=True)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=device.type == "cuda")
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    history = []

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        total_policy = 0.0
        total_value = 0.0
        total_seen = 0
        for state, pi, z in loader:
            state = state.to(device)
            pi = pi.to(device)
            z = z.to(device)
            optimizer.zero_grad(set_to_none=True)
            log_policy, value = model(state)
            loss_p = -torch.mean(torch.sum(pi * log_policy, dim=1))
            loss_v = F.mse_loss(value.view(-1), z)
            loss = loss_p + loss_v
            loss.backward()
            optimizer.step()

            batch = state.size(0)
            total_seen += batch
            total_loss += loss.item() * batch
            total_policy += loss_p.item() * batch
            total_value += loss_v.item() * batch

        history.append(
            {
                "epoch": epoch,
                "loss": total_loss / total_seen,
                "policy_loss": total_policy / total_seen,
                "value_loss": total_value / total_seen,
                "samples": total_seen,
            }
        )
    return history


def train_self_play(
    checkpoint_path,
    output_dir,
    cycles=10,
    games_per_cycle=20,
    train_epochs_per_cycle=1,
    batch_size=128,
    lr=1e-4,
    weight_decay=1e-4,
    temperature=1.0,
    seed=42,
    num_workers=0,
):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = get_device()
    if device.type == "cuda":
        print(f"device=cuda name={torch.cuda.get_device_name(0)}")
    else:
        print("device=cpu")

    checkpoint_path = Path(checkpoint_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model, source_checkpoint, model_config = load_model_from_checkpoint(checkpoint_path, device)

    log_path = output_dir / "self_play_log.csv"
    write_header = not log_path.exists()
    with log_path.open("a", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(
            fp,
            fieldnames=[
                "cycle",
                "games",
                "samples",
                "black_wins",
                "white_wins",
                "draws",
                "loss",
                "policy_loss",
                "value_loss",
                "checkpoint",
                "duration_seconds",
            ],
        )
        if write_header:
            writer.writeheader()

        all_history = []
        for cycle in range(1, cycles + 1):
            start = time.time()
            model.eval()
            samples: list[SelfPlaySample] = []
            winners = []
            for game_idx in range(1, games_per_cycle + 1):
                game_samples, winner = generate_self_play_game(model, device, temperature=temperature)
                samples.extend(game_samples)
                winners.append(winner)
                print(f"cycle={cycle} game={game_idx}/{games_per_cycle} moves={len(game_samples)} winner={winner}")

            train_history = train_on_self_play_samples(
                model,
                samples,
                device,
                batch_size=batch_size,
                lr=lr,
                weight_decay=weight_decay,
                epochs=train_epochs_per_cycle,
                num_workers=num_workers,
            )
            last_train = train_history[-1]
            checkpoint_name = f"{checkpoint_path.stem}_SP_C{cycle:03d}.pth"
            checkpoint_out = output_dir / checkpoint_name
            cycle_record = {
                "cycle": cycle,
                "games": games_per_cycle,
                "samples": len(samples),
                "black_wins": sum(1 for w in winners if w == 1),
                "white_wins": sum(1 for w in winners if w == -1),
                "draws": sum(1 for w in winners if w == 0),
                "loss": last_train["loss"],
                "policy_loss": last_train["policy_loss"],
                "value_loss": last_train["value_loss"],
                "checkpoint": str(checkpoint_out),
                "duration_seconds": time.time() - start,
            }
            all_history.append(cycle_record)
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "model_config": {
                        **model_config,
                        "source": "policy_self_play",
                    },
                    "training_config": {
                        "source_checkpoint": str(checkpoint_path),
                        "cycles": cycles,
                        "games_per_cycle": games_per_cycle,
                        "train_epochs_per_cycle": train_epochs_per_cycle,
                        "batch_size": batch_size,
                        "lr": lr,
                        "weight_decay": weight_decay,
                        "temperature": temperature,
                        "seed": seed,
                    },
                    "source_history": source_checkpoint.get("history", []),
                    "self_play_history": all_history,
                },
                checkpoint_out,
            )
            writer.writerow(cycle_record)
            fp.flush()
            print(
                f"cycle={cycle} samples={len(samples)} loss={last_train['loss']:.4f} "
                f"saved={checkpoint_out}"
            )

    return all_history
