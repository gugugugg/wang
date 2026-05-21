# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
from pathlib import Path

from gobang_ai.self_play import train_self_play


def default_output_dir(checkpoint: str) -> str:
    path = Path(checkpoint)
    parent = path.parent.name if path.parent.name else "model"
    return str(Path("artifacts") / "self_play" / parent / path.stem)


def main():
    parser = argparse.ArgumentParser(description="Continue a checkpoint with lightweight policy self-play training.")
    parser.add_argument("--checkpoint", required=True, help="Source checkpoint path.")
    parser.add_argument("--output-dir", default=None, help="Directory for self-play checkpoints and CSV log.")
    parser.add_argument("--cycles", type=int, default=10, help="Number of self-play training cycles.")
    parser.add_argument("--games-per-cycle", type=int, default=20, help="Self-play games generated before each update.")
    parser.add_argument("--train-epochs-per-cycle", type=int, default=1, help="Training epochs per cycle.")
    parser.add_argument("--batch-size", type=int, default=128, help="Self-play training batch size.")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate.")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay.")
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature; use 0 for greedy.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader workers.")
    args = parser.parse_args()

    output_dir = args.output_dir or default_output_dir(args.checkpoint)
    train_self_play(
        checkpoint_path=args.checkpoint,
        output_dir=output_dir,
        cycles=args.cycles,
        games_per_cycle=args.games_per_cycle,
        train_epochs_per_cycle=args.train_epochs_per_cycle,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        temperature=args.temperature,
        seed=args.seed,
        num_workers=args.num_workers,
    )


if __name__ == "__main__":
    main()
