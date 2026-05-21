# -*- coding: utf-8 -*-
import argparse

from gobang_ai.training import train_base_model


def main():
    parser = argparse.ArgumentParser(description="Train a clean expert-distilled base model.")
    parser.add_argument("--data", default="data/human_games.txt", help="Path to moves|result expert dataset.")
    parser.add_argument("--output", default="artifacts/base_models/base_M64.pth", help="Checkpoint output path.")
    parser.add_argument("--resume", default=None, help="Optional checkpoint to continue training from.")
    parser.add_argument("--filters", type=int, default=64, help="Model width.")
    parser.add_argument("--blocks", type=int, default=5, help="Residual block count.")
    parser.add_argument("--epochs", type=int, default=3, help="Training epochs.")
    parser.add_argument("--start-epoch", type=int, default=0, help="Epoch number already completed in the resume checkpoint.")
    parser.add_argument("--batch-size", type=int, default=256, help="Batch size.")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate.")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay.")
    parser.add_argument("--max-positions", type=int, default=None, help="Optional cap for quick tests.")
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader workers.")
    args = parser.parse_args()

    train_base_model(
        dataset_path=args.data,
        output_path=args.output,
        resume_path=args.resume,
        filters=args.filters,
        blocks=args.blocks,
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
