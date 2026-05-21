# Base Model Training Results

Recorded: 2026-05-13 21:57 Asia/Shanghai

## Environment

- Project: `D:\Users\mjc74\Desktop\GobangAI_PyTorch`
- Dataset: `data/human_games.txt`
- Dataset size: 34,198 expert games
- PyTorch: `2.11.0+cu128`
- CUDA runtime: `12.8`
- GPU: `NVIDIA GeForce RTX 5060 Laptop GPU`
- Model blocks: 5 residual blocks
- Board size: 15x15
- Training source: clean expert distillation

## Command Pattern

```powershell
$env:UV_CACHE_DIR = ".uv-cache"
uv run train-base-model --data data/human_games.txt --resume <previous_checkpoint> --output <next_checkpoint> --filters <filters> --epochs <delta_epochs> --start-epoch <previous_epoch> --batch-size 256
```

## Main Comparison Checkpoints

| Scale | Filters | Epoch | Checkpoint | Train Loss | Train Top1 | Val Loss | Val Top1 |
|---|---:|---:|---|---:|---:|---:|---:|
| S | 16 | 1 | `S16/base_S16_E01.pth` | 2.2819 | 0.6387 | 1.8476 | 0.7101 |
| S | 16 | 3 | `S16/base_S16_E03.pth` | 1.5820 | 0.7519 | 1.5382 | 0.7615 |
| S | 16 | 5 | `S16/base_S16_E05.pth` | 1.4351 | 0.7765 | 1.4166 | 0.7817 |
| S | 16 | 10 | `S16/base_S16_E10.pth` | 1.2524 | 0.8089 | 1.2441 | 0.8114 |
| M | 64 | 1 | `M64/base_M64_E01.pth` | 1.6856 | 0.7372 | 1.2905 | 0.8039 |
| M | 64 | 3 | `M64/base_M64_E03.pth` | 1.1097 | 0.8355 | 1.0892 | 0.8402 |
| M | 64 | 5 | `M64/base_M64_E05.pth` | 1.0586 | 0.8447 | 1.0464 | 0.8471 |
| M | 64 | 10 | `M64/base_M64_E10.pth` | 0.9645 | 0.8629 | 0.9676 | 0.8626 |
| L | 128 | 1 | `L128/base_L128_E01.pth` | 1.4942 | 0.7704 | 1.1918 | 0.8200 |
| L | 128 | 3 | `L128/base_L128_E03.pth` | 1.0276 | 0.8489 | 1.0131 | 0.8521 |
| L | 128 | 5 | `L128/base_L128_E05.pth` | 0.9824 | 0.8570 | 0.9719 | 0.8598 |
| L | 128 | 10 | `L128/base_L128_E10.pth` | 0.9817 | 0.8566 | 0.9864 | 0.8554 |

## Intermediate L128 Checkpoints

| Scale | Filters | Epoch | Checkpoint | Train Loss | Train Top1 | Val Loss | Val Top1 |
|---|---:|---:|---|---:|---:|---:|---:|
| L | 128 | 7 | `L128/base_L128_E07.pth` | 0.9572 | 0.8620 | 0.9541 | 0.8622 |
| L | 128 | 8 | `L128/base_L128_E08.pth` | 0.9965 | 0.8539 | 0.9973 | 0.8556 |
| L | 128 | 9 | `L128/base_L128_E09.pth` | 0.9899 | 0.8552 | 0.9902 | 0.8571 |

## Notes

- Old wrong-label checkpoints are archived under `artifacts/old_checkpoints/` and should not be used.
- `S16`, `M64`, and `L128` now all have E01/E03/E05/E10 checkpoints for the first distillation scaling comparison.
- L128 E10 was completed by splitting the long run into E07, E08, E09, and E10 checkpoints to avoid tool timeout loss.
- Top1 is supervised policy accuracy on the held-out validation split, not direct playing strength. Next step should evaluate these checkpoints in games.
