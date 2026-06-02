---
name: gobangai-project-memory
description: Use when working in this GobangAI_PyTorch project to recall and update project context, decisions, known bugs, training plans, uv setup, dataset/model status, and conversation history. Trigger when the user asks to continue this project, train/evaluate models, refactor the project, rewrite UI, use uv, update notes, or remember prior discussion.
---

# GobangAI Project Memory

Use this skill as the project-local memory for `D:\Users\mjc74\Desktop\GobangAI_PyTorch`.

## Required Workflow

1. Read `references/project-status.md` before making project decisions.
2. Read `references/conversation-log.md` when the user asks to continue prior work or asks what has already been decided.
3. Update both references after meaningful changes:
   - `project-status.md`: current files, known issues, environment, training/data/model state.
   - `conversation-log.md`: concise dated summary of user decisions and completed actions.
4. For evaluation/training experiments, write down every executed step:
   - command purpose and exact command pattern;
   - checkpoint paths and dataset/result paths;
   - game count, seed, decision mode, and opponent setup;
   - key metrics and whether the run is only a minimal validation or a formal result.
5. Keep updates factual and concise. Do not paste long command logs.
6. Treat old checkpoints as suspect unless they were trained after the expert-label bug was fixed.

## Project Rules

- Work in the desktop copy unless the user explicitly points elsewhere:
  `D:\Users\mjc74\Desktop\GobangAI_PyTorch`
- Do not overwrite the clean dataset unless explicitly requested:
  `human_games.txt`
- Do not continue training from old wrong-label models.
- Evaluation output should be written under `artifacts/evaluation/`.
- Evaluation should default to greedy argmax over legal moves; keep `sample_policy_action` for later self-play training.
- When doing minimal validation or formal experiments, update project memory after each completed step so the sequence of actions is recoverable.
- Current paper direction should be framed as an applicability/comparison study, not as proof that SNN improves intelligence or chess-playing strength.
- Before long training, verify:
  - expert labels use the current move, not `moves[i + 1]`;
  - inference uses 3 feature planes;
  - the uv environment works;
  - old checkpoints are isolated or output names are new.

## References

- `references/project-status.md`: current technical status.
- `references/conversation-log.md`: summarized dialogue and decisions.
