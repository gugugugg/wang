# GobangAI Project Status

Last updated: 2026-05-20 Asia/Shanghai

## Active Workspace

- Active project copy: `D:\Users\mjc74\Desktop\GobangAI_PyTorch`
- Original Win project source: `D:\c\xiangmu\GobangAI_PyTorch`
- Linux/server run archive: `D:\Users\mjc74\Desktop\wang`
- Earlier linux-named copy inspected: `D:\c\xiangmu\GobangAI_PyTorch - linux`

## Dataset

- Clean expert dataset copied into active project:
  `D:\Users\mjc74\Desktop\GobangAI_PyTorch\data\human_games.txt`
- Verified statistics from `D:\Users\mjc74\Desktop\wang\human_games.txt`:
  - 34,198 games
  - 8,724,970 bytes
  - no duplicate moves detected
  - no out-of-range moves detected
  - complete `moves|result` labels
  - results: black wins 17,974, white wins 16,224

## Model Findings

- Old 100-cycle models in `D:\Users\mjc74\Desktop\wang` were trained with a wrong expert label.
- Diagnosis on 5,000 samples:
  - S latest: true current move Top1 2.22%, next move Top1 72.50%
  - M latest: true current move Top1 1.50%, next move Top1 75.50%
  - L latest: true current move Top1 1.14%, next move Top1 79.98%, next move Top5 94.04%
- Conclusion: old models learned the next move (`moves[i + 1]`) instead of the current legal move. Do not use them as base models.

## Known Bugs To Fix Before Training

1. Expert-data label bug in `worker.py`:
   - current code still uses `pi[moves[i + 1]] = 1.0`
   - correct target is the current move `m`
   - add illegal-game filtering while fixing
2. Self-play value bug in `worker.py`:
   - terminal reward is from the player who just moved
   - after `env.step`, `env.current_player` has already switched
   - compute winner as `-env.current_player` when reward is positive
3. Inference feature mismatch:
   - `search_engine.py` and `evaluator.py` use 1-plane tensors in old code
   - model requires `[current stones, opponent stones, color plane]`
4. Old checkpoints in `models_scale_*` are suspect; avoid continuing from them.

## Work Completed

- Copied Win project to desktop:
  `D:\Users\mjc74\Desktop\GobangAI_PyTorch`
- Copied clean dataset from:
  `D:\Users\mjc74\Desktop\wang\human_games.txt`
- Cleaned mojibake comments/docstrings in core files:
  - `worker.py`
  - `model.py`
  - `game_env.py`
  - `mcts_engine.py`
  - `task_train.py`
  - `task_gen.py`
  - `task_convert.py`
- Added uv project files:
  - `pyproject.toml`
  - `.python-version`
  - updated `.gitignore`
  - rewritten `README.md`
- `uv.lock` was generated and should be kept for reproducible installs.
- Ran `uv sync` successfully with project-local cache:
  `$env:UV_CACHE_DIR = ".uv-cache"; uv sync`
- uv installed:
  - Python environment created at `.venv`
- installed `torch==2.11.0`, `numpy==2.2.6`, `pygame==2.6.1`, `numba==0.65.1`, and dependencies
- Added project-local skill:
  `skills/gobangai-project-memory`
- Reorganized project root:
  - core package: `gobang_ai/`
  - command scripts: `scripts/`
  - dataset: `data/human_games.txt`
  - resources: `assets/`
  - generated checkpoints: `artifacts/`
  - old root modules and IDE/build files: `legacy/`
- Configured uv to use PyTorch CUDA 12.8 wheels.
- Verified GPU environment:
  - `torch==2.11.0+cu128`
  - CUDA available: true
  - GPU: NVIDIA GeForce RTX 5060 Laptop GPU
- Trained clean 1-epoch expert-distilled base models:
  - `artifacts/base_models/base_S16_clean_e1.pth`
    - train top1 0.6387, val top1 0.7101
  - `artifacts/base_models/base_M64_clean_e1.pth`
    - train top1 0.7372, val top1 0.8039
  - `artifacts/base_models/base_L128_clean_e1.pth`
    - train top1 0.7704, val top1 0.8200
- Recorded base model results in:
  `artifacts/base_models/TRAINING_RESULTS.md`
- Continued distillation training from E01 checkpoints:
  - S16 complete: E01/E03/E05/E10
  - M64 complete: E01/E03/E05/E10
  - L128 complete: E01/E03/E05/E10
- L128 intermediate checkpoints also exist: E07/E08/E09.
- Updated `artifacts/base_models/TRAINING_RESULTS.md` with full comparison table.
- Replaced the root `main.py` with a clean dashboard entry point.
- Added `gobang_ai/ui/dashboard.py`, a Pygame research dashboard showing:
  - S/M/L validation Top-1 curves across E01/E03/E05/E10
  - checkpoint inventory
  - CUDA/GPU and dataset status
  - quick controls: R reload, O open results, Esc quit
- Added uv entry command:
  `uv run gobang-dashboard`
- Added lightweight self-play training module:
  - `gobang_ai/self_play.py`
  - `scripts/train_self_play.py`
  - uv entry command: `uv run train-self-play`
- Extended dashboard with self-play controls:
  - model/checkpoint selector
  - self-play cycle selector
  - games-per-cycle selector
  - generated command preview
  - background launch button
- Verified self-play dry run on CUDA:
  - `torch==2.11.0+cu128`
  - GPU: NVIDIA GeForce RTX 5060 Laptop GPU
  - dry-run output: `artifacts/self_play/dry_run/S16_E10/base_S16_E10_SP_C001.pth`
- Added detailed self-play experiment plan:
  `docs/SELF_PLAY_EXPERIMENT_PLAN.md`
- Fixed dashboard startup crash caused by `pygame.font.SysFont("Segoe UI", ...)` failing on Windows font registry data:
  - `gobang_ai/ui/dashboard.py` now loads `segoeui.ttf` / `consola.ttf` directly from `C:\Windows\Fonts`.
  - no-window dashboard initialization passes.
- Reintroduced small launcher packaging:
  - `scripts/launcher.py`
  - `wang.spec`
  - `dist/wang.exe`
  - root copy: `wang.exe`
- Launcher still calls the live source environment:
  `.venv\Scripts\pythonw.exe -m gobang_ai.ui.dashboard`
- Packaged exe self-test passes:
  `.\wang.exe --self-test`
- Runtime logs are stored under:
  `C:\Users\mjc74\AppData\Local\GobangAI_PyTorch\logs`
- Preferred launch options:
  - double-click `D:\Users\mjc74\Desktop\GobangAI_PyTorch\wang.exe`
  - or source launch: `uv run gobang-dashboard`
- Added SNN brain-inspired training path:
  - `gobang_ai/snn_model.py`
  - `gobang_ai/snn_training.py`
  - `scripts/train_snn_model.py`
  - uv entry command: `uv run train-snn-model`
- SNN model:
  - `GobangSNNDualHead`
  - LIF layers with surrogate-gradient binary spikes
  - spiking residual blocks
  - policy/value heads compatible with existing Gomoku data and action space
- Updated self-play checkpoint loading so `architecture=snn_lif` checkpoints load with `GobangSNNDualHead`.
- Added SNN experiment plan:
  `docs/SNN_BRAIN_INSPIRED_EXPERIMENT_PLAN.md`
- Verified SNN smoke training on CUDA:
  - command uses 512 sampled positions, filters 8, 1 block, 2 time steps, 1 epoch
  - output: `artifacts/dry_runs/snn_smoke_M8_B1_T2_E01_512.pth`
  - checkpoint loads through self-play loader and produces policy/value tensors
- Completed formal SNN E01 expert distillation only:
  - checkpoint: `artifacts/snn_models/SNN_M64_B3_T6_E01.pth`
  - model: filters 64, 3 spiking residual blocks, 6 LIF time steps
  - train loss 2.7374, train Top-1 0.5602
  - val loss 2.1357, val Top-1 0.6551
  - results recorded in `artifacts/snn_models/TRAINING_RESULTS.md`
- User later requested E03, then E05, an E07 transition checkpoint, and E10 if it could complete before noon.
- Completed SNN checkpoints:
  - E01: train Top-1 0.5602, val Top-1 0.6551
  - E03: train Top-1 0.7157, val Top-1 0.7253
  - E05: train Top-1 0.7440, val Top-1 0.7514
  - E07: train Top-1 0.7581, val Top-1 0.7620
  - E10: train Top-1 0.7713, val Top-1 0.7719
- Final SNN E10 checkpoint:
  `artifacts/snn_models/SNN_M64_B3_T6_E10.pth`
- Verified E10 loads through the self-play checkpoint loader and produces policy/value tensors.
- Drafted opening report:
  `docs/SNN_OPENING_REPORT.md`
- Opening report covers:
  - research background and significance
  - domestic/foreign research status
  - technical route
  - traditional M64 vs SNN M64-B3-T6 comparison plan
  - completed distillation metrics
  - follow-up game-strength evaluation plan
  - schedule, expected outcomes, and references

## Intended Research Direction

Goal: compare SNN-trained brain-inspired Gomoku models against traditional CNN/ResNet models for a meeting/conference paper.

Working title:

`SNN 训练的类脑模型相比于传统训练方法的模型的智能提升程度研究`

Current framing:

- train traditional CNN/ResNet and LIF-SNN models on the same clean expert dataset
- compare under aligned width/data/epoch budgets where possible
- evaluate with win-rate matrices or Elo, not only supervised Top-1/loss
- include training time and inference time because SNN time steps increase compute
- avoid using old wrong-label checkpoints

Recommended experiment framing:

- measurable intelligence improvement of SNN brain-inspired training over traditional training
- variables: architecture type, SNN time steps, model width, training budget
- metrics: Elo/win rate, head-to-head win rate, first/second-player win rates, average move count, invalid move rate, loss, supervised Top-1, runtime

## uv Notes

- `uv` default cache at `D:\AI_Caches\uv_cache` had access errors.
- Use project-local cache for commands:
  ```powershell
  $env:UV_CACHE_DIR = ".uv-cache"
uv sync
uv run gobang-dashboard
uv run train-base-model --data data/human_games.txt --output artifacts/base_models/base_M64_clean_e1.pth --filters 64 --epochs 1
uv run train-snn-model --data data/human_games.txt --output artifacts/snn_models/SNN_M64_T6_E01.pth --filters 64 --blocks 3 --time-steps 6 --epochs 1
uv run train-self-play --checkpoint artifacts/base_models/M64/base_M64_E10.pth --output-dir artifacts/self_play/M64/base_M64_E10 --cycles 30 --games-per-cycle 20
uv run python -m py_compile worker.py model.py game_env.py mcts_engine.py task_train.py task_gen.py task_convert.py
```

## Next Planned Steps

1. Run pilot SNN vs traditional training on matched 8,192-position samples.
2. Train full M64 traditional baseline and M64/T6 SNN comparison checkpoints.
3. Add/finish a checkpoint-vs-checkpoint evaluation script for win-rate matrices.
4. Evaluate traditional-vs-SNN head-to-head games as black and white.
5. Record supervised metrics, win-rate metrics, and runtime metrics for paper tables.

## 2026-05-21 Report Update

- Rewrote `docs/SNN_OPENING_REPORT.md` with more rigorous academic language.
- Updated the title to focus on a LIF-SNN Gomoku policy-value model and comparison with traditional CNN/ResNet baselines.
- Strengthened research framing:
  - SNN improvement is treated as a hypothesis to be tested, not a preset conclusion.
  - Traditional and SNN models are described as same-task, same-data, similar-width comparisons, not strictly parameter-equivalent comparisons.
  - Supervised Top-1 is explicitly separated from game-playing strength.
  - Head-to-head evaluation, rule-baseline evaluation, stability metrics, and efficiency metrics are emphasized as necessary follow-up evidence.
- Verified the report reads correctly with UTF-8 encoding.
