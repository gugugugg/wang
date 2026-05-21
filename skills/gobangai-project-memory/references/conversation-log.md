# Conversation Log

Last updated: 2026-05-20 Asia/Shanghai

## 2026-05-13

- User asked to inspect `D:\c\xiangmu\GobangAI_PyTorch - linux`.
- Project identified as a Python/PyTorch Gomoku AI with AlphaZero-style dual-head model, MCTS, expert distillation, self-play, and Pygame UI.
- Initial issue: large-scale distilled models still played randomly.
- Investigation found the small root `human_games.txt` in one project copy was corrupted, but later the actual server data in `D:\Users\mjc74\Desktop\wang` was clean.
- User provided `D:\Users\mjc74\Desktop\wang` as the real uploaded/server run result.
- Dataset in `wang` was verified clean: 34,198 games, complete labels, no duplicate/out-of-range moves.
- Model diagnosis showed old trained models predicted `moves[i + 1]` with very high Top1 accuracy but barely predicted the correct current move.
- Root cause identified: `worker.py` expert loader used `pi[moves[i + 1]] = 1.0`, teaching the next move instead of the current move.
- User confirmed understanding: generated data is correct, training is wrong.
- User decided to use Win version `D:\c\xiangmu\GobangAI_PyTorch` for local training.
- Project copied to `D:\Users\mjc74\Desktop\GobangAI_PyTorch`.
- Clean `human_games.txt` copied from `D:\Users\mjc74\Desktop\wang`.
- User requested fixing comments first, before logic changes.
- Mojibake comments/docstrings cleaned in core training files.
- User requested next steps: project normalization and uv management.
- Added `pyproject.toml`, `.python-version`, updated `.gitignore`, rewrote `README.md`.
- `uv sync` initially timed out during PyTorch download, then completed successfully using `.uv-cache`.
- User requested conversation and project state be stored as a project skill for next use.
- Created project-local skill: `skills/gobangai-project-memory`.
- Verified `uv run python -m py_compile ...` works in the new `.venv`.
- `uv.lock` was generated and retained for reproducibility.
- User objected that root project folder was still messy.
- Project files were reorganized:
  - data moved to `data/`
  - images and Yixin engine moved to `assets/`
  - core code placed in `gobang_ai/`
  - old root modules moved to `legacy/root_modules/`
  - old checkpoints moved to `artifacts/old_checkpoints/`
- PyTorch was changed from CPU build to CUDA build via uv:
  `torch==2.11.0+cu128`.
- GPU was verified as NVIDIA GeForce RTX 5060 Laptop GPU.
- Clean base models trained with CUDA for S/M/L widths, 1 epoch each:
  - S16: val top1 0.7101
  - M64: val top1 0.8039
  - L128: val top1 0.8200
- User asked to record training results.
- Added `artifacts/base_models/TRAINING_RESULTS.md` with environment, commands, metrics, and checkpoint paths.
- User requested organizing current three models into per-parameter folders and continuing E3/E5/E10 training.
- Moved E01 checkpoints into `artifacts/base_models/S16`, `M64`, and `L128`.
- Moved dry-run checkpoints into `artifacts/dry_runs`.
- Added `--resume` and `--start-epoch` support to `train-base-model`.
- Completed distillation checkpoints:
  - S16: E01/E03/E05/E10
  - M64: E01/E03/E05/E10
  - L128: E01/E03/E05/E10
- L128 E10 required splitting into E07, E08, E09, E10 because one long E07->E10 command hit the tool timeout.
- User requested a more beautiful UI.
- Implemented a new Pygame research dashboard in `gobang_ai/ui/dashboard.py`.
- Replaced `main.py` with the dashboard entry point and added uv script `gobang-dashboard`.
- Verified dashboard parsing and no-window smoke rendering.

## Standing User Intent

- Build a clean base model from the existing expert dataset.
- Then run controlled experiments:
  - same training cycles with different model sizes
  - same model size with different training cycles
- Use these experiments as the basis for a conference-style AI paper.

## 2026-05-14

- User requested UI controls for selecting a model, self-play cycles, and games per cycle.
- Added a lightweight self-play training path:
  - `gobang_ai/self_play.py`
  - `scripts/train_self_play.py`
  - uv entry command: `train-self-play`
- Dashboard now supports:
  - selecting available checkpoints under `artifacts/base_models/*/*.pth`
  - changing `Cycles`
  - changing `Games / cycle`
  - previewing the generated self-play command
  - launching training in the background from the UI
- Verified dashboard no-window rendering.
- Verified CUDA availability:
  - `torch==2.11.0+cu128`
  - `NVIDIA GeForce RTX 5060 Laptop GPU`
- Ran self-play dry run:
  - checkpoint: `artifacts/base_models/S16/base_S16_E10.pth`
  - cycles: 1
  - games per cycle: 1
  - output: `artifacts/self_play/dry_run/S16_E10/base_S16_E10_SP_C001.pth`
- Added `docs/SELF_PLAY_EXPERIMENT_PLAN.md` with pilot, scale-comparison, and budget-comparison plans.
- User asked for an exe packaging method where code changes do not require repackaging.
- Decided to package only a small launcher exe, not the full project source.
- Added `scripts/launcher.py`.
- Packaged with icon `assets/images/logo.ico` and exe name `wang`.
- Launcher calls the live source environment:
  `.venv\Scripts\pythonw.exe -m gobang_ai.ui.dashboard`
- Therefore code updates take effect by closing and reopening `wang.exe`; repackaging is only needed if the launcher itself changes.
- User reported double-click startup flash-exit.
- Root cause found with a console debug build:
  the launcher tried to write logs into the project root and received `PermissionError: [Errno 13] Permission denied`.
- Fixed launcher logging:
  - runtime/error logs now go to `%LOCALAPPDATA%\GobangAI_PyTorch\logs`
  - logging failures no longer stop the UI from launching
  - launcher searches upward from exe location to find `.venv` and `gobang_ai`
- Rebuilt and copied `wang.exe` to:
  `D:\Users\mjc74\Desktop\GobangAI_PyTorch\wang.exe`
- If startup fails later, inspect:
  `C:\Users\mjc74\AppData\Local\GobangAI_PyTorch\logs`
- User reported the packaged exe still had problems and decided not to package.
- Removed exe packaging-related files:
  - `build/`
  - `dist/`
  - `wang.exe`
  - `wang.spec`
  - `wang_debug.spec`
  - `scripts/launcher.py`
- Removed the PyInstaller optional dependency from `pyproject.toml`.
- Current dashboard launch path is source-based:
  `$env:UV_CACHE_DIR = ".uv-cache"; uv run gobang-dashboard`

## Important Caution

- Do not train from the old wrong-label checkpoints.
- Do not use supervised validation Top-1 alone as playing-strength evidence; report win-rate or Elo-style results.

## 2026-05-21

- User clarified that `docs/SNN_OPENING_REPORT.md` should be read with UTF-8 encoding; the file itself was not corrupted.
- Re-read the report with `-Encoding UTF8` and confirmed the Chinese content is normal.
- User requested more rigorous language for the opening report.
- Rewrote `docs/SNN_OPENING_REPORT.md` into a more formal academic opening-report draft:
  - changed the title to emphasize LIF-SNN policy-value modeling and comparison with traditional CNN/ResNet models;
  - removed language that implied SNN improvement was already proven;
  - clarified that SNN improvement is an experimental hypothesis;
  - stated that Traditional M64 and SNN M64-B3-T6 are same-task, same-data, similar-width comparisons, not strictly parameter-equivalent;
  - emphasized game-based evaluation and efficiency metrics as necessary evidence beyond supervised Top-1.

## 2026-05-20

- User reported the packaged exe could not open.
- Inspected existing launcher logs in:
  `C:\Users\mjc74\AppData\Local\GobangAI_PyTorch\logs`
- Found current crash root cause:
  `pygame.font.SysFont("Segoe UI", ...)` failed inside Pygame Windows font discovery with `TypeError: expected str, bytes or os.PathLike object, not int`.
- Fixed `gobang_ai/ui/dashboard.py` to load Windows font files directly from `C:\Windows\Fonts`, avoiding Pygame `SysFont` registry parsing.
- Verified:
  - `uv run python -m py_compile gobang_ai\ui\dashboard.py`
  - dummy-video dashboard initialization
  - direct font load smoke test
- Recreated `scripts/launcher.py` as a small source launcher with `--self-test`.
- Rebuilt `wang.exe` using PyInstaller onefile/noconsole with `assets/images/logo.ico`.
- Copied the new exe to:
  `D:\Users\mjc74\Desktop\GobangAI_PyTorch\wang.exe`
- Verified packaged exe self-test:
  `.\wang.exe --self-test`
- User requested changing the training logic to try SNN and changing the paper topic to:
  `SNN 训练的类脑模型相比于传统训练方法的模型的智能提升程度研究`
- Added SNN model/training implementation:
  - `gobang_ai/snn_model.py`
  - `gobang_ai/snn_training.py`
  - `scripts/train_snn_model.py`
  - `train-snn-model` uv entry point
- SNN implementation uses LIF layers, surrogate-gradient spikes, spiking residual blocks, and dual policy/value heads.
- Updated `gobang_ai/self_play.py` so checkpoints with `model_config.architecture=snn_lif` load correctly.
- Updated README with the SNN paper direction and training command.
- Added:
  `docs/SNN_BRAIN_INSPIRED_EXPERIMENT_PLAN.md`
- Verified:
  - py_compile for SNN modules and loader
  - SNN forward pass tensor shapes
  - CUDA smoke training on 512 sampled positions
  - SNN checkpoint loading through the self-play loader
- Smoke checkpoint:
  `artifacts/dry_runs/snn_smoke_M8_B1_T2_E01_512.pth`
- User asked to start distillation, then clarified to run only E01 and wait for further instruction.
- Started formal SNN expert distillation:
  `uv run train-snn-model --data data\human_games.txt --output artifacts\snn_models\SNN_M64_B3_T6_E01.pth --filters 64 --blocks 3 --time-steps 6 --epochs 1 --batch-size 128`
- Training completed and saved:
  `artifacts/snn_models/SNN_M64_B3_T6_E01.pth`
- E01 metrics:
  - train loss 2.7374
  - train Top-1 0.5602
  - val loss 2.1357
  - val Top-1 0.6551
- Verified the SNN E01 checkpoint loads through the self-play loader and produces policy/value tensors.
- Added:
  `artifacts/snn_models/TRAINING_RESULTS.md`
- User requested E03, then requested E05 and an intermediate checkpoint between E05 and E10 to reduce risk.
- Completed SNN continuation checkpoints:
  - `artifacts/snn_models/SNN_M64_B3_T6_E03.pth`
  - `artifacts/snn_models/SNN_M64_B3_T6_E05.pth`
  - `artifacts/snn_models/SNN_M64_B3_T6_E07.pth`
  - `artifacts/snn_models/SNN_M64_B3_T6_E10.pth`
- E10 completed at about 2026-05-21 04:05 Asia/Shanghai, before the user's noon deadline.
- Final recorded SNN metrics:
  - E01 val Top-1 0.6551
  - E03 val Top-1 0.7253
  - E05 val Top-1 0.7514
  - E07 val Top-1 0.7620
  - E10 val Top-1 0.7719
- Updated `artifacts/snn_models/TRAINING_RESULTS.md` with all SNN checkpoints.
- User asked to write an opening report.
- Added:
  `docs/SNN_OPENING_REPORT.md`
- Report is a formal Chinese opening-report draft for:
  `SNN 训练的类脑模型相比于传统训练方法的模型的智能提升程度研究`
- It uses the completed traditional M64 and SNN M64-B3-T6 distillation metrics, while leaving game-strength conclusions as future experimental validation.
