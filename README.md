# GobangAI PyTorch

Gomoku AI research project for comparing LIF-SNN brain-inspired models against traditional CNN/ResNet training.

## Environment

This project is managed with `uv`.

```powershell
$env:UV_CACHE_DIR = ".uv-cache"
uv sync
```

If you already have a CUDA-specific PyTorch build installed in a local environment, keep that environment and use the same entry commands below.

## Required Local Files

These files are used locally and are intentionally ignored by Git:

- `data/human_games.txt`: expert dataset in `moves|result` format.
- `assets/engines/pbrain-Yixin2018.exe`: optional Yixin engine binary for generating more expert games.
- `models_scale_16/`, `models_scale_64/`, `models_scale_128/`: generated checkpoints.
- `raw_sgf_data/`: temporary SGF output from expert game generation.

## Common Commands

Run the GUI:

```powershell
$env:UV_CACHE_DIR = ".uv-cache"
uv run gobang-dashboard
```

The dashboard shows the S/M/L distillation checkpoints, validation curves, environment status, and checkpoint inventory.
It also includes controls for selecting a checkpoint, choosing self-play cycles and games per cycle, and launching self-play training in a background process.

Run the diagnostic suite:

```powershell
$env:UV_CACHE_DIR = ".uv-cache"
uv run python test_all.py
```

Compile-check the core modules:

```powershell
$env:UV_CACHE_DIR = ".uv-cache"
uv run python -m py_compile worker.py model.py game_env.py mcts_engine.py task_train.py task_gen.py task_convert.py
```

Train a clean base model from the existing expert dataset:

```powershell
$env:UV_CACHE_DIR = ".uv-cache"
uv run train-base-model --data data/human_games.txt --output artifacts/base_models/base_M64.pth --filters 64 --epochs 3
```

Train a LIF-SNN brain-inspired model on the same expert dataset:

```powershell
$env:UV_CACHE_DIR = ".uv-cache"
uv run train-snn-model --data data/human_games.txt --output artifacts/snn_models/SNN_M64_T6_E01.pth --filters 64 --blocks 3 --time-steps 6 --epochs 1
```

Continue a checkpoint with lightweight self-play training:

```powershell
$env:UV_CACHE_DIR = ".uv-cache"
uv run train-self-play --checkpoint artifacts/base_models/M64/base_M64_E10.pth --output-dir artifacts/self_play/M64/base_M64_E10 --cycles 30 --games-per-cycle 20
```

Quick dry-run with a small sample:

```powershell
$env:UV_CACHE_DIR = ".uv-cache"
uv run train-base-model --data data/human_games.txt --output artifacts/base_models/dry_run_M16.pth --filters 16 --epochs 1 --max-positions 2048
```

## Research Plan

Working paper title:

```text
SNN 训练的类脑模型相比于传统训练方法的模型的智能提升程度研究
```

The intended experiment is now:

1. Train traditional CNN/ResNet and LIF-SNN models from the same `human_games.txt`.
2. Keep model width, data budget, and evaluation protocol aligned where possible.
3. Compare supervised policy accuracy, game win rate, Elo-style score, first/second-player strength, and training cost.
4. Report whether the SNN class-brain model improves intelligence metrics over traditional training, and under which budgets.

Before long training runs, verify that expert-label construction and inference feature planes are correct.

Detailed self-play planning is stored in:

```text
docs/SELF_PLAY_EXPERIMENT_PLAN.md
```

## Source Layout

- `gobang_ai/`: package code for model, features, environment, data loading, training, and evaluation.
- `gobang_ai/snn_model.py`: LIF-SNN dual-head model with surrogate gradients.
- `gobang_ai/ui/`: Pygame research dashboard.
- `scripts/`: command-line entry scripts.
- `docs/`: project plans and experiment design notes.
- `data/`: local datasets.
- `assets/`: images and external engine binaries.
- `artifacts/`: generated checkpoints, dry runs, and archived old checkpoints.
- `legacy/`: old IDE/build files or deprecated source retained for reference.

## Project Memory Skill

Project-specific context is stored in:

```text
skills/gobangai-project-memory/
```

Use this skill when continuing the project so prior decisions, known bugs, uv setup, dataset status, and training plans stay synchronized.
