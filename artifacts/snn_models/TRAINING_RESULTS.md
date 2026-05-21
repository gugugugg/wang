# SNN Model Training Results

Recorded: 2026-05-21 Asia/Shanghai

## Environment

- Project: `D:\Users\mjc74\Desktop\GobangAI_PyTorch`
- Dataset: `data/human_games.txt`
- Training source: clean expert distillation
- Architecture: LIF-SNN dual-head policy/value model
- PyTorch: project uv environment
- GPU: NVIDIA GeForce RTX 5060 Laptop GPU

## Working Paper Title

SNN 训练的类脑模型相比于传统训练方法的模型的智能提升程度研究

## Command Pattern

```powershell
$env:UV_CACHE_DIR = ".uv-cache"
uv run train-snn-model --data data/human_games.txt --output artifacts/snn_models/SNN_M64_B3_T6_E01.pth --filters 64 --blocks 3 --time-steps 6 --epochs 1 --batch-size 128
```

Continuation pattern:

```powershell
$env:UV_CACHE_DIR = ".uv-cache"
uv run train-snn-model --data data/human_games.txt --resume artifacts/snn_models/SNN_M64_B3_T6_E07.pth --output artifacts/snn_models/SNN_M64_B3_T6_E10.pth --filters 64 --blocks 3 --time-steps 6 --epochs 3 --start-epoch 7 --batch-size 128
```

## Main SNN Checkpoints

| Model | Filters | Blocks | Time Steps | Epoch | Checkpoint | Train Loss | Train Top1 | Val Loss | Val Top1 |
|---|---:|---:|---:|---:|---|---:|---:|---:|---:|
| SNN-M64-B3-T6 | 64 | 3 | 6 | 1 | `SNN_M64_B3_T6_E01.pth` | 2.7374 | 0.5602 | 2.1357 | 0.6551 |
| SNN-M64-B3-T6 | 64 | 3 | 6 | 3 | `SNN_M64_B3_T6_E03.pth` | 1.7603 | 0.7157 | 1.7158 | 0.7253 |
| SNN-M64-B3-T6 | 64 | 3 | 6 | 5 | `SNN_M64_B3_T6_E05.pth` | 1.5998 | 0.7440 | 1.5856 | 0.7514 |
| SNN-M64-B3-T6 | 64 | 3 | 6 | 7 | `SNN_M64_B3_T6_E07.pth` | 1.5225 | 0.7581 | 1.5105 | 0.7620 |
| SNN-M64-B3-T6 | 64 | 3 | 6 | 10 | `SNN_M64_B3_T6_E10.pth` | 1.4489 | 0.7713 | 1.4525 | 0.7719 |

## Notes

- E07 was added as an intermediate transition checkpoint before E10 to reduce progress-loss risk during long runs.
- This checkpoint is for SNN-vs-traditional comparison and should be evaluated with game-based metrics before making intelligence-improvement claims.
