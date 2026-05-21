# SNN Brain-Inspired Gomoku Experiment Plan

Last updated: 2026-05-20

## Working Title

SNN 训练的类脑模型相比于传统训练方法的模型的智能提升程度研究

## Core Question

Under the same Gomoku expert dataset and comparable training budgets, does a LIF-SNN brain-inspired model produce measurable intelligence improvement over a traditional CNN/ResNet policy-value model?

The word "intelligence" should be operationalized as game-playing strength, not only supervised accuracy.

## Compared Methods

### Traditional Baseline

- Model: `GobangDualHead`
- Architecture: CNN input stem + residual blocks + policy/value heads
- Training command: `train-base-model`
- Existing checkpoints:
  - `artifacts/base_models/S16/base_S16_E10.pth`
  - `artifacts/base_models/M64/base_M64_E10.pth`
  - `artifacts/base_models/L128/base_L128_E10.pth`

### Brain-Inspired SNN

- Model: `GobangSNNDualHead`
- Architecture: leaky integrate-and-fire layers, surrogate-gradient spikes, spiking residual blocks, policy/value heads
- Training command: `train-snn-model`
- Default checkpoint folder: `artifacts/snn_models/`

## Initial Pilot

Run small samples first to verify convergence and runtime.

| Group | Model | Filters | Blocks | Time steps | Epochs | Max positions |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| T-Pilot | Traditional CNN | 64 | 5 | - | 1 | 8192 |
| S-Pilot | LIF-SNN | 64 | 3 | 6 | 1 | 8192 |

Recommended commands:

```powershell
$env:UV_CACHE_DIR = ".uv-cache"
uv run train-base-model --data data/human_games.txt --output artifacts/dry_runs/traditional_M64_E01_8192.pth --filters 64 --blocks 5 --epochs 1 --max-positions 8192
uv run train-snn-model --data data/human_games.txt --output artifacts/dry_runs/snn_M64_B3_T6_E01_8192.pth --filters 64 --blocks 3 --time-steps 6 --epochs 1 --max-positions 8192
```

## Main Comparison A: Same Dataset, Same Epochs

| Group | Method | Filters | Blocks | Time steps | Epochs |
| --- | --- | ---: | ---: | ---: | ---: |
| A1 | Traditional CNN | 64 | 5 | - | 10 |
| A2 | LIF-SNN | 64 | 3 | 6 | 10 |

This comparison tests whether the SNN produces stronger policy/value behavior under the same expert dataset and epoch count.

## Main Comparison B: SNN Time-Step Ablation

| Group | Method | Filters | Blocks | Time steps | Epochs |
| --- | --- | ---: | ---: | ---: | ---: |
| B1 | LIF-SNN | 64 | 3 | 2 | 5 |
| B2 | LIF-SNN | 64 | 3 | 4 | 5 |
| B3 | LIF-SNN | 64 | 3 | 6 | 5 |
| B4 | LIF-SNN | 64 | 3 | 8 | 5 |

This comparison tests whether more spiking simulation steps improve intelligent behavior or only increase compute cost.

## Main Comparison C: Game-Strength Evaluation

Evaluate every important checkpoint by:

1. Rule-based baseline games as black and white.
2. Traditional-vs-SNN head-to-head matches.
3. Round-robin between checkpoints trained with different SNN time steps.

Report:

- supervised validation Top-1
- win rate against rule baseline
- traditional-vs-SNN head-to-head win rate
- first-player and second-player win rates
- draw rate
- average move count
- training time per epoch
- inference time per move

## Paper Claim Discipline

Only claim "intelligence improvement" if SNN checkpoints improve game-based metrics, preferably head-to-head win rate or Elo-style score. If SNN improves Top-1 but not win rate, describe it as representation or imitation improvement, not playing intelligence improvement.
