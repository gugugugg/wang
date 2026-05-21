# -*- coding: utf-8 -*-
from pathlib import Path
import time

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split

from gobang_ai.data import ExpertMoveDataset, load_expert_games
from gobang_ai.model import GobangDualHead


def get_device(prefer_cuda=True):
    if prefer_cuda and torch.cuda.is_available():
        try:
            torch.zeros(1).cuda()
            return torch.device("cuda")
        except Exception:
            pass
    return torch.device("cpu")


def train_base_model(
    dataset_path,
    output_path,
    resume_path=None,
    filters=64,
    blocks=5,
    epochs=3,
    start_epoch=0,
    batch_size=256,
    lr=2e-4,
    weight_decay=1e-4,
    max_positions=None,
    val_fraction=0.05,
    seed=42,
    num_workers=0,
):
    dataset_path = Path(dataset_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    games, skipped = load_expert_games(dataset_path)
    dataset = ExpertMoveDataset(games, augment=True, max_positions=max_positions, seed=seed)
    if len(dataset) == 0:
        raise RuntimeError(f"No valid training samples loaded from {dataset_path}")

    val_size = int(len(dataset) * val_fraction)
    train_size = len(dataset) - val_size
    generator = torch.Generator().manual_seed(seed)
    if val_size > 0:
        train_set, val_set = random_split(dataset, [train_size, val_size], generator=generator)
    else:
        train_set, val_set = dataset, None

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = None
    if val_set is not None:
        val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    device = get_device()
    if device.type == "cuda":
        print(f"device=cuda name={torch.cuda.get_device_name(0)}")
    else:
        print("device=cpu")
    model = GobangDualHead(num_filters=filters, num_blocks=blocks).to(device)
    loaded_history = []
    if resume_path:
        resume_path = Path(resume_path)
        checkpoint = torch.load(resume_path, map_location=device)
        model.load_state_dict(checkpoint["model_state"], strict=True)
        loaded_history = list(checkpoint.get("history", []))
        cfg = checkpoint.get("model_config", {})
        if cfg.get("num_filters") and int(cfg["num_filters"]) != int(filters):
            raise ValueError(f"Resume checkpoint filters={cfg['num_filters']} does not match requested filters={filters}")
        if cfg.get("num_blocks") and int(cfg["num_blocks"]) != int(blocks):
            raise ValueError(f"Resume checkpoint blocks={cfg['num_blocks']} does not match requested blocks={blocks}")
        print(f"resumed={resume_path}")

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs), eta_min=1e-6)

    history = loaded_history
    start_time = time.time()
    for local_epoch in range(1, epochs + 1):
        epoch = start_epoch + local_epoch
        model.train()
        total_loss = 0.0
        total_policy = 0.0
        total_value = 0.0
        total_top1 = 0
        total_seen = 0

        for state, pi, z in train_loader:
            state = state.to(device)
            pi = pi.to(device)
            z = z.to(device)

            optimizer.zero_grad(set_to_none=True)
            log_policy, value = model(state)
            loss_p = -torch.mean(torch.sum(pi * log_policy, dim=1))
            loss_v = F.mse_loss(value.view(-1), z)
            loss = loss_p + loss_v
            loss.backward()
            optimizer.step()

            batch = state.size(0)
            target = pi.argmax(dim=1)
            pred = log_policy.argmax(dim=1)
            total_top1 += (pred == target).sum().item()
            total_seen += batch
            total_loss += loss.item() * batch
            total_policy += loss_p.item() * batch
            total_value += loss_v.item() * batch

        scheduler.step()
        row = {
            "epoch": epoch,
            "local_epoch": local_epoch,
            "train_loss": total_loss / total_seen,
            "train_policy_loss": total_policy / total_seen,
            "train_value_loss": total_value / total_seen,
            "train_top1": total_top1 / total_seen,
        }

        if val_loader is not None:
            row.update(evaluate_policy_value(model, val_loader, device))

        history.append(row)
        print(
            f"epoch={epoch} loss={row['train_loss']:.4f} "
            f"policy={row['train_policy_loss']:.4f} value={row['train_value_loss']:.4f} "
            f"top1={row['train_top1']:.4f}"
        )
        if "val_loss" in row:
            print(f"  val_loss={row['val_loss']:.4f} val_top1={row['val_top1']:.4f}")

    checkpoint = {
        "model_state": model.state_dict(),
        "model_config": {
            "num_filters": filters,
            "num_blocks": blocks,
            "board_size": 15,
            "source": "clean_expert_distillation",
        },
        "training_config": {
            "dataset_path": str(dataset_path),
            "resume_path": str(resume_path) if resume_path else None,
            "games": len(games),
            "skipped_games": skipped,
            "samples": len(dataset),
            "epochs_this_run": epochs,
            "start_epoch": start_epoch,
            "total_epoch": start_epoch + epochs,
            "batch_size": batch_size,
            "lr": lr,
            "weight_decay": weight_decay,
            "max_positions": max_positions,
            "duration_seconds": time.time() - start_time,
        },
        "history": history,
    }
    torch.save(checkpoint, output_path)
    print(f"saved={output_path}")
    return checkpoint


def evaluate_policy_value(model, loader, device):
    model.eval()
    total_loss = 0.0
    total_policy = 0.0
    total_value = 0.0
    total_top1 = 0
    total_seen = 0

    with torch.no_grad():
        for state, pi, z in loader:
            state = state.to(device)
            pi = pi.to(device)
            z = z.to(device)
            log_policy, value = model(state)
            loss_p = -torch.mean(torch.sum(pi * log_policy, dim=1))
            loss_v = F.mse_loss(value.view(-1), z)
            loss = loss_p + loss_v

            batch = state.size(0)
            total_loss += loss.item() * batch
            total_policy += loss_p.item() * batch
            total_value += loss_v.item() * batch
            total_top1 += (log_policy.argmax(dim=1) == pi.argmax(dim=1)).sum().item()
            total_seen += batch

    return {
        "val_loss": total_loss / total_seen,
        "val_policy_loss": total_policy / total_seen,
        "val_value_loss": total_value / total_seen,
        "val_top1": total_top1 / total_seen,
    }
