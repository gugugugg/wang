# GobangAI Self-Play Experiment Plan

Last updated: 2026-05-14

## Goal

Use the clean expert-distilled checkpoints as initial models, then compare self-play training under controlled budgets.

Separate two claims in the paper:

1. Model-scale comparison: under the same self-play budget, compare S16, M64, and L128.
2. Training-budget comparison: under the same model scale, compare different self-play cycle counts.

Validation Top-1 from expert distillation can be reported as an auxiliary metric, but playing strength must be supported by win-rate or Elo-style evaluation.

## Initial Checkpoints

Use the E10 distilled checkpoints as the default initializers.

| Scale | Filters | Initial checkpoint |
| --- | ---: | --- |
| S16 | 16 | `artifacts/base_models/S16/base_S16_E10.pth` |
| M64 | 64 | `artifacts/base_models/M64/base_M64_E10.pth` |
| L128 | 128 | `artifacts/base_models/L128/base_L128_E10.pth` |

Keep E01, E03, and E05 for the supervised pretraining ablation, not for the main self-play scaling experiment.

## Pilot Plan

Run this first to verify stability, GPU utilization, and log format.

| Group | Model | Cycles | Games per cycle | Total games |
| --- | --- | ---: | ---: | ---: |
| P1 | S16_E10 | 3 | 10 | 30 |
| P2 | M64_E10 | 3 | 10 | 30 |
| P3 | L128_E10 | 3 | 10 | 30 |

Purpose:

- Confirm that all three model sizes can complete self-play training.
- Check whether losses are finite and checkpoints are saved every cycle.
- Run a small evaluation against the rule-based baseline before committing to long runs.

## Main Experiment A: Same Budget, Different Model Scale

Use the same self-play budget for each scale.

| Group | Model | Cycles | Games per cycle | Total games |
| --- | --- | ---: | ---: | ---: |
| A1 | S16_E10 | 30 | 20 | 600 |
| A2 | M64_E10 | 30 | 20 | 600 |
| A3 | L128_E10 | 30 | 20 | 600 |

If time is sufficient, extend the same groups to 50 cycles and 20 games per cycle, for 1000 total games per model.

Controlled variables:

- Same initial expert dataset.
- Same expert-distillation epoch: E10.
- Same self-play cycles and games per cycle.
- Same sampling temperature, learning rate, batch size, and evaluation protocol.

## Main Experiment B: Same Model, Different Training Budget

Use M64_E10 as the main budget-scaling model because it has good supervised accuracy and moderate training cost.

| Group | Model | Cycles | Games per cycle | Total games |
| --- | --- | ---: | ---: | ---: |
| B0 | M64_E10 | 0 | 0 | 0 |
| B1 | M64_E10 | 10 | 20 | 200 |
| B2 | M64_E10 | 30 | 20 | 600 |
| B3 | M64_E10 | 50 | 20 | 1000 |
| B4 | M64_E10 | 100 | 20 | 2000 |

If compute is tight, stop at B3 and report B4 as future work.

## Recommended Commands

```powershell
$env:UV_CACHE_DIR = ".uv-cache"
uv run train-self-play --checkpoint artifacts/base_models/M64/base_M64_E10.pth --output-dir artifacts/self_play/M64/base_M64_E10 --cycles 30 --games-per-cycle 20
```

The dashboard can launch the same command from the UI:

```powershell
$env:UV_CACHE_DIR = ".uv-cache"
uv run gobang-dashboard
```

In the UI:

- Select a checkpoint with the model selector.
- Select `Cycles`.
- Select `Games / cycle`.
- Click `Start Training` or press Enter.

## Evaluation Plan

After each important checkpoint, run evaluations in three layers:

1. Rule-based baseline: measure win rate as black and white.
2. Round-robin between S16, M64, and L128 at the same cycle count.
3. Round-robin between M64 checkpoints at C0, C10, C30, C50, and C100.

Recommended report metrics:

- Win rate.
- First-player and second-player win rate.
- Average move count.
- Draw rate.
- Invalid move rate, expected to be zero.
- Training loss, policy loss, value loss.
- GPU model and software environment.

## Paper Argument Structure

Use the experiment results to support restrained claims:

- Expert distillation gives a legal and structured initial policy.
- With the same self-play budget, larger models may improve evaluation strength, but the gain should be verified by win rate rather than only supervised Top-1.
- With the same model size, additional self-play cycles may improve playing strength until saturation or instability appears.
- If a larger model has better supervised Top-1 but worse self-play evaluation, discuss overfitting, exploration, and compute budget as possible causes.
