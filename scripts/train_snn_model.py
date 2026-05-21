# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse

from gobang_ai.snn_training import train_snn_model


def main():
    parser = argparse.ArgumentParser(description="Train a LIF-SNN expert-distilled Gomoku model.")
    parser.add_argument("--data", default="data/human_games.txt", help="Path to moves|result expert dataset.")
    parser.add_argument("--output", default="artifacts/snn_models/SNN_M64_T6_E01.pth", help="Checkpoint output path.")
    parser.add_argument("--resume", default=None, help="Optional SNN checkpoint to continue training from.")
    parser.add_argument("--filters", type=int, default=64, help="SNN channel width.")
    parser.add_argument("--blocks", type=int, default=3, help="Spiking residual block count.")
    parser.add_argument("--time-steps", type=int, default=6, help="LIF simulation steps per board state.")
    parser.add_argument("--tau", type=float, default=0.25, help="Membrane decay coefficient.")
    parser.add_argument("--threshold", type=float, default=1.0, help="Spike threshold.")
    parser.add_argument("--epochs", type=int, default=3, help="Training epochs.")
    parser.add_argument("--start-epoch", type=int, default=0, help="Epoch number already completed in the resume checkpoint.")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size.")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate.")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay.")
    parser.add_argument("--max-positions", type=int, default=None, help="Optional cap for quick tests.")
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader workers.")
    args = parser.parse_args()

    train_snn_model(
        dataset_path=args.data,
        output_path=args.output,
        resume_path=args.resume,
        filters=args.filters,
        blocks=args.blocks,
        time_steps=args.time_steps,
        tau=args.tau,
        threshold=args.threshold,
        epochs=args.epochs,
        start_epoch=args.start_epoch,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        max_positions=args.max_positions,
        num_workers=args.num_workers,
    )


if __name__ == "__main__":
    main()
