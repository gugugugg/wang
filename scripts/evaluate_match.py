# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
from pathlib import Path

from gobang_ai.evaluation import run_match_evaluation


def default_name(path: str) -> str:
    if path.lower() == "rule":
        return "RuleBasedAI"
    return Path(path).stem


def main():
    parser = argparse.ArgumentParser(description="Evaluate Gomoku checkpoints with greedy legal-move matches.")
    parser.add_argument("--model-a", required=True, help="Checkpoint path for model A, or 'rule'.")
    parser.add_argument("--model-b", required=True, help="Checkpoint path for model B, or 'rule'.")
    parser.add_argument("--model-a-name", default=None, help="Display name for model A.")
    parser.add_argument("--model-b-name", default=None, help="Display name for model B.")
    parser.add_argument("--games", type=int, default=20, help="Number of games to play, or opening count with --paired-openings.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--experiment-id", default="minimal_validation", help="Experiment id written to CSV.")
    parser.add_argument("--run-id", default=None, help="Optional run id written to CSV.")
    parser.add_argument("--output", required=True, help="Output CSV path.")
    parser.add_argument("--no-swap-sides", action="store_true", help="Disable alternating black/white sides.")
    parser.add_argument("--min-opening-moves", type=int, default=0, help="Minimum generated opening length.")
    parser.add_argument("--max-opening-moves", type=int, default=0, help="Optional random opening length cap.")
    parser.add_argument("--paired-openings", action="store_true", help="Run each generated opening twice with sides swapped.")
    args = parser.parse_args()

    model_a_name = args.model_a_name or default_name(args.model_a)
    model_b_name = args.model_b_name or default_name(args.model_b)
    rows, summary = run_match_evaluation(
        model_a_path=args.model_a,
        model_b_path=args.model_b,
        output_csv=args.output,
        games=args.games,
        model_a_name=model_a_name,
        model_b_name=model_b_name,
        seed=args.seed,
        experiment_id=args.experiment_id,
        run_id=args.run_id,
        swap_sides=not args.no_swap_sides,
        max_opening_moves=args.max_opening_moves,
        min_opening_moves=args.min_opening_moves,
        paired_openings=args.paired_openings,
    )

    print(f"output={args.output}")
    print(
        f"games={summary['games']} model_a_wins={summary['model_a_wins']} "
        f"losses={summary['model_a_losses']} draws={summary['draws']} "
        f"win_rate={summary['model_a_win_rate']:.3f} avg_moves={summary['avg_moves']:.1f} "
        f"illegal_moves={summary['illegal_moves']}"
    )
    if rows:
        print(f"decision_mode={rows[0]['decision_mode']} device={rows[0]['device']} gpu={rows[0]['gpu_name']}")


if __name__ == "__main__":
    main()
