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

## 2026-06-03 Evaluation Data Plan

- CSV template added:
  `docs/EVALUATION_MATCH_TEMPLATE.csv`
- New standing rule: every evaluation/training step must be recorded in project memory, including command purpose, command pattern, checkpoint paths, result paths, game count, seed, decision mode, opponent setup, key metrics, and whether it is a minimal validation or formal result.
- Real evaluation outputs should be written under:
  `artifacts/evaluation/`
- The first evaluation script should be added as:
  `scripts/evaluate_match.py`
- Shared match logic should be added to:
  `gobang_ai/evaluation.py`
- Keep `sample_policy_action` for later self-play training; evaluation should default to greedy argmax over legal moves.
- Do not add SNN spike-rate statistics in the first evaluation version. It is not needed for the current paper conclusion.
- First script version should record one CSV row per game with:
  experiment id, run id, match type, model names/paths, black/white player, winner, result for model A, move count, illegal move count, black/white inference time, game duration, decision mode, seed, opening fields, device, torch version, GPU name, and notes.
- Minimum validation experiments:
  1. `base_M64_E10` vs `RuleBasedAI`, 20 games.
  2. `SNN_M64_B3_T6_E10` vs `RuleBasedAI`, 20 games.
  3. `base_M64_E10` vs `SNN_M64_B3_T6_E10`, 20 games.
- Formal experiment order:
  1. Traditional and SNN models vs `RuleBasedAI` as an anchor baseline.
  2. Final SNN vs final traditional model head-to-head.
  3. Same-width, same-epoch comparisons:
     `E01`, `E03`, `E05`, and `E10`.
  4. Increase game count and repeat with multiple seeds to reduce accidental variance.
- Formal reporting metrics:
  win/loss/draw, black/white win rates, average moves, illegal moves, per-step inference time, per-game time, and later Wilson confidence intervals / relative Elo if needed.

## 2026-06-03 Minimal Evaluation Implementation

- Added reusable greedy match evaluation logic to:
  `gobang_ai/evaluation.py`
- Added CLI script:
  `scripts/evaluate_match.py`
- The evaluator supports:
  - checkpoint vs `RuleBasedAI`;
  - checkpoint vs checkpoint;
  - SNN and traditional checkpoint loading through `load_model_from_checkpoint`;
  - greedy argmax over legal moves;
  - alternating black/white sides by default;
  - one CSV row per game using the template fields;
  - model warmup before timing to reduce CUDA cold-start bias.
- Compile check passed:
  ```powershell
  $env:UV_CACHE_DIR='.uv-cache'; uv run python -m py_compile gobang_ai\evaluation.py scripts\evaluate_match.py
  ```
- Minimal validation outputs were written to:
  `artifacts/evaluation/minimal_validation/`
- Minimal validation commands and results:
  1. Purpose: traditional final model vs weak rule baseline.
     Command pattern:
     ```powershell
     $env:UV_CACHE_DIR='.uv-cache'; uv run python -m scripts.evaluate_match --model-a artifacts\base_models\M64\base_M64_E10.pth --model-b rule --model-a-name base_M64_E10 --model-b-name RuleBasedAI --games 20 --seed 42 --experiment-id minimal_base_vs_rule --output artifacts\evaluation\minimal_validation\base_m64_e10_vs_rule_20.csv
     ```
     Result: `base_M64_E10` won 20/20, draw 0, illegal moves 0, average moves 16.8.
  2. Purpose: SNN final model vs weak rule baseline.
     Command pattern:
     ```powershell
     $env:UV_CACHE_DIR='.uv-cache'; uv run python -m scripts.evaluate_match --model-a artifacts\snn_models\SNN_M64_B3_T6_E10.pth --model-b rule --model-a-name SNN_M64_B3_T6_E10 --model-b-name RuleBasedAI --games 20 --seed 42 --experiment-id minimal_snn_vs_rule --output artifacts\evaluation\minimal_validation\snn_m64_b3_t6_e10_vs_rule_20.csv
     ```
     Result: `SNN_M64_B3_T6_E10` won 15/20, lost 5/20, draw 0, illegal moves 0, average moves 51.6.
  3. Purpose: direct final traditional vs final SNN pipeline validation.
     Command pattern:
     ```powershell
     $env:UV_CACHE_DIR='.uv-cache'; uv run python -m scripts.evaluate_match --model-a artifacts\base_models\M64\base_M64_E10.pth --model-b artifacts\snn_models\SNN_M64_B3_T6_E10.pth --model-a-name base_M64_E10 --model-b-name SNN_M64_B3_T6_E10 --games 20 --seed 42 --experiment-id minimal_base_vs_snn --output artifacts\evaluation\minimal_validation\base_m64_e10_vs_snn_m64_b3_t6_e10_20.csv
     ```
     Result: `base_M64_E10` won 20/20, draw 0, illegal moves 0, average moves 66.5.
- Interpretation: these 20-game runs validate the evaluation pipeline only. They are not formal paper evidence. Next formal step is larger RuleBasedAI anchor experiments, then larger SNN-vs-traditional head-to-head, then same-width/same-epoch comparisons.

## 2026-06-03 100-Game Trial Evaluation

- Purpose: extend the 20-game minimal validation to 100 games per matchup before formal large-scale experiments.
- Status: trial / extended validation, not final paper evidence.
- Shared settings:
  - decision mode: greedy argmax over legal moves
  - seed: 42
  - side control: alternating black/white sides
  - device: CUDA, NVIDIA GeForce RTX 5060 Laptop GPU
  - output directory: `artifacts/evaluation/trial_100/`
- Commands and results:
  1. Traditional final model vs weak rule baseline.
     ```powershell
     $env:UV_CACHE_DIR='.uv-cache'; uv run python -m scripts.evaluate_match --model-a artifacts\base_models\M64\base_M64_E10.pth --model-b rule --model-a-name base_M64_E10 --model-b-name RuleBasedAI --games 100 --seed 42 --experiment-id trial100_base_vs_rule --output artifacts\evaluation\trial_100\base_m64_e10_vs_rule_100.csv
     ```
     Result: `base_M64_E10` won 100/100, lost 0, draw 0, illegal moves 0, average moves 15.6.
     Side split: black 50/50, white 50/50.
     Average inference time: `base_M64_E10` 2.356 ms/step, `RuleBasedAI` 0.405 ms/step.
  2. SNN final model vs weak rule baseline.
     ```powershell
     $env:UV_CACHE_DIR='.uv-cache'; uv run python -m scripts.evaluate_match --model-a artifacts\snn_models\SNN_M64_B3_T6_E10.pth --model-b rule --model-a-name SNN_M64_B3_T6_E10 --model-b-name RuleBasedAI --games 100 --seed 42 --experiment-id trial100_snn_vs_rule --output artifacts\evaluation\trial_100\snn_m64_b3_t6_e10_vs_rule_100.csv
     ```
     Result: `SNN_M64_B3_T6_E10` won 76/100, lost 24, draw 0, illegal moves 0, average moves 48.3.
     Side split: black 35/50, white 41/50.
     Average inference time: `SNN_M64_B3_T6_E10` 11.235 ms/step, `RuleBasedAI` 0.800 ms/step.
  3. Direct final traditional vs final SNN trial.
     ```powershell
     $env:UV_CACHE_DIR='.uv-cache'; uv run python -m scripts.evaluate_match --model-a artifacts\base_models\M64\base_M64_E10.pth --model-b artifacts\snn_models\SNN_M64_B3_T6_E10.pth --model-a-name base_M64_E10 --model-b-name SNN_M64_B3_T6_E10 --games 100 --seed 42 --experiment-id trial100_base_vs_snn --output artifacts\evaluation\trial_100\base_m64_e10_vs_snn_m64_b3_t6_e10_100.csv
     ```
     Result: `base_M64_E10` won 100/100, `SNN_M64_B3_T6_E10` won 0/100, draw 0, illegal moves 0, average moves 66.5.
     Side split for `base_M64_E10`: black 50/50, white 50/50.
     Average inference time: `base_M64_E10` 2.713 ms/step, `SNN_M64_B3_T6_E10` 10.996 ms/step.
- Interpretation:
  - The 100-game trial still shows the traditional M64 E10 model is much stronger than the current SNN M64-B3-T6 E10 under greedy evaluation.
  - SNN remains stronger than the weak rule baseline but significantly weaker than the traditional M64 final model.
  - SNN inference is roughly 4x slower than traditional M64 on this GPU because of LIF time steps.
- Recommended next step:
  - If continuing toward formal results, run the same-width/same-epoch matrix (`E01`, `E03`, `E05`, `E10`) or rerun final head-to-head with larger game count / multiple seeds before writing final paper claims.

## 2026-06-03 Same-Epoch 100-Game Matrix

- Purpose: compare traditional M64 and SNN M64-B3-T6 checkpoints at matched epochs.
- Status: trial matrix, not final paper evidence.
- Shared settings:
  - matchups: `base_M64_E01/E03/E05/E10` vs `SNN_M64_B3_T6_E01/E03/E05/E10`
  - games: 100 per matchup
  - seed: 42
  - decision mode: greedy argmax over legal moves
  - side control: alternating black/white sides
  - start board: empty board, no opening perturbation
  - output directory: `artifacts/evaluation/epoch_matrix_100/`
- Commands:
  ```powershell
  $env:UV_CACHE_DIR='.uv-cache'; uv run python -m scripts.evaluate_match --model-a artifacts\base_models\M64\base_M64_E01.pth --model-b artifacts\snn_models\SNN_M64_B3_T6_E01.pth --model-a-name base_M64_E01 --model-b-name SNN_M64_B3_T6_E01 --games 100 --seed 42 --experiment-id matrix100_base_vs_snn_e01 --output artifacts\evaluation\epoch_matrix_100\base_m64_e01_vs_snn_m64_b3_t6_e01_100.csv
  $env:UV_CACHE_DIR='.uv-cache'; uv run python -m scripts.evaluate_match --model-a artifacts\base_models\M64\base_M64_E03.pth --model-b artifacts\snn_models\SNN_M64_B3_T6_E03.pth --model-a-name base_M64_E03 --model-b-name SNN_M64_B3_T6_E03 --games 100 --seed 42 --experiment-id matrix100_base_vs_snn_e03 --output artifacts\evaluation\epoch_matrix_100\base_m64_e03_vs_snn_m64_b3_t6_e03_100.csv
  $env:UV_CACHE_DIR='.uv-cache'; uv run python -m scripts.evaluate_match --model-a artifacts\base_models\M64\base_M64_E05.pth --model-b artifacts\snn_models\SNN_M64_B3_T6_E05.pth --model-a-name base_M64_E05 --model-b-name SNN_M64_B3_T6_E05 --games 100 --seed 42 --experiment-id matrix100_base_vs_snn_e05 --output artifacts\evaluation\epoch_matrix_100\base_m64_e05_vs_snn_m64_b3_t6_e05_100.csv
  $env:UV_CACHE_DIR='.uv-cache'; uv run python -m scripts.evaluate_match --model-a artifacts\base_models\M64\base_M64_E10.pth --model-b artifacts\snn_models\SNN_M64_B3_T6_E10.pth --model-a-name base_M64_E10 --model-b-name SNN_M64_B3_T6_E10 --games 100 --seed 42 --experiment-id matrix100_base_vs_snn_e10 --output artifacts\evaluation\epoch_matrix_100\base_m64_e10_vs_snn_m64_b3_t6_e10_100.csv
  ```
- Results:
  - E01: `base_M64_E01` won 50/100, `SNN_M64_B3_T6_E01` won 50/100, draw 0, illegal 0, average moves 62.0. Black won 100/100. Base side split: black 50/50, white 0/50. Average inference: base 2.743 ms/step, SNN 11.171 ms/step.
  - E03: `base_M64_E03` won 50/100, `SNN_M64_B3_T6_E03` won 50/100, draw 0, illegal 0, average moves 64.0. White won 100/100. Base side split: black 0/50, white 50/50. Average inference: base 2.712 ms/step, SNN 11.012 ms/step.
  - E05: `base_M64_E05` won 50/100, `SNN_M64_B3_T6_E05` won 50/100, draw 0, illegal 0, average moves 101.0. Black won 100/100. Base side split: black 50/50, white 0/50. Average inference: base 2.744 ms/step, SNN 11.049 ms/step.
  - E10: `base_M64_E10` won 100/100, `SNN_M64_B3_T6_E10` won 0/100, draw 0, illegal 0, average moves 66.5. Black won 50/100, white won 50/100. Base side split: black 50/50, white 50/50. Average inference: base 2.671 ms/step, SNN 10.764 ms/step.
- Interpretation:
  - E01/E03/E05 50/50 results are not evidence of equal strength. They are deterministic greedy empty-board patterns dominated by color/winner-color effects.
  - E10 remains clearly favorable to the traditional M64 model under this evaluation setup.
  - Repeating the same empty-board deterministic game 100 or 200 times gives limited extra information.
- Recommended next step:
  - Add paired-opening evaluation support: generate fixed legal opening sequences, run each opening twice with sides swapped, and report paired results.
  - After paired openings are implemented, rerun final E10 and same-epoch matrix as formal evidence.

## 2026-05-21 Report Update

- Rewrote `docs/SNN_OPENING_REPORT.md` with more rigorous academic language.
- Updated the title to focus on a LIF-SNN Gomoku policy-value model and comparison with traditional CNN/ResNet baselines.
- Strengthened research framing:
  - SNN improvement is treated as a hypothesis to be tested, not a preset conclusion.
  - Traditional and SNN models are described as same-task, same-data, similar-width comparisons, not strictly parameter-equivalent comparisons.
  - Supervised Top-1 is explicitly separated from game-playing strength.
  - Head-to-head evaluation, rule-baseline evaluation, stability metrics, and efficiency metrics are emphasized as necessary follow-up evidence.
- Verified the report reads correctly with UTF-8 encoding.
- Further revised the report to avoid vague "intelligence level" wording and use measurable terms:
  strategy fitting ability, game-playing performance, decision quality, and efficiency-performance tradeoff.

## 2026-05-21 GitHub Upload

- Initialized a local Git repository in `D:\Users\mjc74\Desktop\GobangAI_PyTorch`.
- Cleaned `.gitattributes` to avoid stale `.venv` and build-output LFS entries.
- Confirmed `.gitignore` excludes local/generated files:
  `.venv/`, `.uv-cache/`, `data/human_games.txt`, `*.pth`, `*.exe`, `build/`, `dist/`, and `artifacts/self_play/`.
- Created initial commit:
  `fc96da4 Initial GobangAI PyTorch project`
- Added remote:
  `https://github.com/gugugugg/wang.git`
- Pushed `main` successfully to GitHub.
