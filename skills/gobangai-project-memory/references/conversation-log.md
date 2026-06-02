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

## 2026-06-03

- User decided the evaluation workflow should start with a minimal validation experiment, then scale up to reduce accidental variance.
- User requested that every step be written down in project memory before starting the minimal validation work.
- Added this standing rule to `skills/gobangai-project-memory/SKILL.md`: record every evaluation/training step with command purpose, command pattern, checkpoint paths, result paths, game count, seed, decision mode, opponent setup, key metrics, and whether it is minimal validation or formal result.
- Decided to keep `sample_policy_action` for later self-play training and use greedy argmax over legal moves for evaluation.
- Decided SNN spike-rate statistics are not needed for the current paper because the core conclusion depends on strategy fitting, game performance, decision quality, and runtime efficiency.
- Added CSV game-detail template:
  `docs/EVALUATION_MATCH_TEMPLATE.csv`
- Recorded intended implementation locations:
  - match/evaluation logic: `gobang_ai/evaluation.py`
  - CLI script: `scripts/evaluate_match.py`
  - generated results: `artifacts/evaluation/`
- Planned experiment order:
  1. minimal validation: traditional vs RuleBasedAI, SNN vs RuleBasedAI, traditional vs SNN;
  2. formal RuleBasedAI anchor experiments;
  3. formal final SNN vs traditional head-to-head;
  4. same-width, same-epoch SNN vs traditional comparisons;
  5. larger game counts and multiple seeds.
- Implemented the first greedy match evaluator:
  - shared logic in `gobang_ai/evaluation.py`;
  - CLI entry in `scripts/evaluate_match.py`;
  - output location `artifacts/evaluation/minimal_validation/`.
- Added model warmup before timing to reduce CUDA cold-start bias in per-step inference metrics.
- Compile check passed for:
  `gobang_ai/evaluation.py` and `scripts/evaluate_match.py`.
- Completed minimal validation runs with seed 42, 20 games each, greedy argmax over legal moves, alternating sides:
  - `base_M64_E10` vs `RuleBasedAI`: 20 wins, 0 losses, 0 draws, 0 illegal moves, average moves 16.8.
  - `SNN_M64_B3_T6_E10` vs `RuleBasedAI`: 15 wins, 5 losses, 0 draws, 0 illegal moves, average moves 51.6.
  - `base_M64_E10` vs `SNN_M64_B3_T6_E10`: `base_M64_E10` won 20/20, 0 draws, 0 illegal moves, average moves 66.5.
- These results are recorded as pipeline validation only, not formal paper evidence.
- User requested a 100-game trial before larger formal experiments.
- Completed 100-game trial runs with seed 42, greedy argmax over legal moves, alternating sides:
  - `base_M64_E10` vs `RuleBasedAI`: 100 wins, 0 losses, 0 draws, illegal moves 0, average moves 15.6; side split black 50/50, white 50/50; average inference time 2.356 ms/step for base model.
  - `SNN_M64_B3_T6_E10` vs `RuleBasedAI`: 76 wins, 24 losses, 0 draws, illegal moves 0, average moves 48.3; side split black 35/50, white 41/50; average inference time 11.235 ms/step for SNN.
  - `base_M64_E10` vs `SNN_M64_B3_T6_E10`: base won 100/100, SNN won 0/100, 0 draws, illegal moves 0, average moves 66.5; base side split black 50/50, white 50/50; average inference time base 2.713 ms/step, SNN 10.996 ms/step.
- Trial CSV outputs were written under:
  `artifacts/evaluation/trial_100/`
- Interpretation recorded: the current SNN model is stronger than the weak rule baseline but much weaker and slower than the traditional M64 E10 model under greedy evaluation. Treat this as trial evidence before formal multi-seed or same-epoch matrix experiments.
- User asked to continue, and the same-epoch 100-game matrix was run for E01/E03/E05/E10.
- Outputs were written under:
  `artifacts/evaluation/epoch_matrix_100/`
- Same-epoch matrix summary:
  - E01: base 50/100, SNN 50/100, average moves 62.0, illegal 0; black won 100/100, so this is a deterministic color-pattern result, not equal-strength evidence.
  - E03: base 50/100, SNN 50/100, average moves 64.0, illegal 0; white won 100/100, also a deterministic color-pattern result.
  - E05: base 50/100, SNN 50/100, average moves 101.0, illegal 0; black won 100/100, also a deterministic color-pattern result.
  - E10: base 100/100, SNN 0/100, average moves 66.5, illegal 0; base won as both black and white.
- Average inference times in the matrix were about 2.7 ms/step for traditional M64 and 10.8-11.2 ms/step for SNN.
- Important conclusion: repeating deterministic greedy empty-board games has limited value after this point. Next evaluator improvement should be paired openings: fixed opening sequences, each replayed with sides swapped.
- User agreed to implement paired openings.
- Added paired-opening support:
  - `--paired-openings` treats `--games` as opening count and runs two side-swapped games per opening.
  - `--min-opening-moves` and `--max-opening-moves` control generated opening length.
  - CSV rows now record shared `opening_id` and `start_moves` for paired games.
- Compile check passed after paired-opening changes.
- Ran paired-opening validation for E10 with 10 fixed 4-move openings, 20 games total:
  - output: `artifacts/evaluation/paired_opening_validation/base_m64_e10_vs_snn_m64_b3_t6_e10_10openings_20games.csv`
  - valid paired openings: 10/10
  - base won 18/20, SNN won 2/20, draws 0, illegal moves 0, average moves 47.9.
- Next recommended run: paired-opening E10 with 50 or 100 openings, then same-epoch paired-opening matrix.
- User requested to continue, so the paired-opening E10 evaluation was scaled to 50 openings / 100 games.
- Output:
  `artifacts/evaluation/paired_opening_e10_50/base_m64_e10_vs_snn_m64_b3_t6_e10_50openings_100games.csv`
- Settings: seed 42, fixed 4-move openings, greedy argmax, two side-swapped games per opening.
- Result:
  - 50/50 valid opening pairs;
  - base won 93/100, SNN won 7/100, draws 0, illegal moves 0, average moves 44.6;
  - base side split: black 49/50, white 44/50;
  - paired outcomes: base 2-0 on 43 openings, split 1-1 on 7 openings, SNN 2-0 on 0 openings;
  - average inference: base 2.834 ms/step, SNN 11.583 ms/step.
- Interpretation: paired openings reduce the earlier deterministic color-pattern artifact; E10 still strongly favors the traditional M64 model.
- User requested to continue, so the paired-opening same-epoch matrix was run with 50 openings / 100 games per epoch.
- Output directory:
  `artifacts/evaluation/paired_opening_epoch_matrix_50/`
- Shared settings: seed 42, fixed 4-move openings, greedy argmax, two side-swapped games per opening.
- Matrix results:
  - E01: base 99/100, SNN 1/100, average moves 30.6, illegal 0; base 2-0 on 49 openings, split 1-1 on 1 opening, SNN 2-0 on 0 openings.
  - E03: base 98/100, SNN 2/100, average moves 35.7, illegal 0; base 2-0 on 48 openings, split 1-1 on 2 openings, SNN 2-0 on 0 openings.
  - E05: base 97/100, SNN 3/100, average moves 39.9, illegal 0; base 2-0 on 47 openings, split 1-1 on 3 openings, SNN 2-0 on 0 openings.
  - E10: base 93/100, SNN 7/100, average moves 45.0, illegal 0; base 2-0 on 43 openings, split 1-1 on 7 openings, SNN 2-0 on 0 openings.
- Average inference stayed near 2.7-2.8 ms/step for traditional M64 and 11.0-11.5 ms/step for SNN.
- Interpretation: paired-opening results are the strongest current game-strength evidence and consistently favor the traditional M64 model at all matched epochs.
- User reflected that the current chess-model/SNN direction has limited value beyond a本科毕业设计.
- Assistant agreed:
  - it is enough for an undergraduate thesis because the project has a complete engineering and experimental loop;
  - it is not strong enough as a higher-level research direction because Gomoku AI is mature, direct CNN-to-SNN replacement is weak novelty, and the result is a negative comparison without deeper theory or hardware evidence.
- User concluded the direction seems problematic but still acceptable for本科毕业设计.
- Recorded new recommended framing:
  - do not present the work as proving SNN improves intelligence;
  - present it as an applicability evaluation / comparison study of LIF-SNN and CNN/ResNet for Gomoku strategy modeling.
- Recorded practical conclusion:
  - SNN is not completely unable to play because it beats RuleBasedAI and occasionally wins paired-opening games;
  - however, under current architecture/training/evaluation it is clearly weaker than traditional M64 and slower on GPU.
- Recommended thesis titles:
  - `LIF-SNN 在五子棋策略建模中的适用性评估`
  - `基于 LIF-SNN 与 CNN/ResNet 的五子棋策略模型对比研究`
  - `基于配对开局评测的 LIF-SNN 与传统 CNN/ResNet 五子棋策略模型对比分析`
- Recommended future direction if continuing SNN research: leave Gomoku and move to event-driven/time-series/low-power sensing tasks where SNN has a more natural advantage.

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
- User pointed out that "智能程度" was still too vague.
- Revised `docs/SNN_OPENING_REPORT.md` again to avoid vague ability-level wording, replacing it with measurable terms such as strategy fitting ability, game-playing performance, decision quality, and running efficiency.
- User requested uploading the project to GitHub and provided:
  `https://github.com/gugugugg/wang`
- Initialized a local Git repository, cleaned stale `.gitattributes` entries, verified ignored large/local files, created the initial commit, added the GitHub remote, and pushed `main` successfully.

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
