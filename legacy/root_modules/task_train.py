# -*- coding: utf-8 -*-
import os
import time
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from model import GobangDualHead


BATCH_SIZE = 256
START_LR = 2e-4
WEIGHT_DECAY = 1e-4
BOARD_SIZE = 15
LOG_FILE = "system_log.txt"


def log_to_file(msg):
    t_str = time.strftime('%H:%M:%S')
    print(f"[Train] {msg}")
    try:
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(f"[{t_str}] [Train] {msg}\n")
    except:
        pass


def get_safe_device():
    if torch.cuda.is_available():
        try:
            torch.zeros(1).cuda()
            return torch.device("cuda")
        except:
            pass
    return torch.device("cpu")


class GobangMCTSDataset(Dataset):
    """Dataset wrapper with optional 8-way board symmetry augmentation."""

    def __init__(self, data_list, use_augmentation=True):
        self.data = data_list
        self.use_augmentation = use_augmentation

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        state, pi, z = self.data[idx]

        state = state.astype(np.float32)
        pi = pi.astype(np.float32)
        z = float(z)

        if self.use_augmentation:
            mode = np.random.randint(0, 8)
            state, pi = self.augment(state, pi, mode)

        return torch.from_numpy(state), torch.from_numpy(pi), torch.tensor(z, dtype=torch.float32)

    def augment(self, state, pi, mode):
        pi_board = pi.reshape(15, 15)

        state_aug = state.copy()
        pi_aug = pi_board.copy()

        k = mode % 4
        if k > 0:
            state_aug = np.rot90(state_aug, k, axes=(1, 2))
            pi_aug = np.rot90(pi_aug, k)

        if mode >= 4:
            state_aug = np.flip(state_aug, axis=2)
            pi_aug = np.flip(pi_aug, axis=1)

        return state_aug.copy(), pi_aug.flatten().copy()


def run_train_task(model_name, filters_str, current_loop, dir_name, training_data):
    """Run one training pass and save a checkpoint."""
    if not training_data:
        return None

    device = get_safe_device()
    save_path = os.path.join(dir_name, f"models_scale_{filters_str}", model_name)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    num_filters = int(filters_str)
    model = GobangDualHead(num_filters=num_filters, board_size=BOARD_SIZE).to(device)

    if os.path.exists(save_path):
        try:
            checkpoint = torch.load(save_path, map_location=device)
            model.load_state_dict(checkpoint['model_state'], strict=False)
            log_to_file(f"Loaded existing model: {model_name}")
        except:
            log_to_file(f"Created new model: {model_name}")

    model.train()
    optimizer = optim.Adam(model.parameters(), lr=START_LR, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100, eta_min=1e-6)

    dataset = GobangMCTSDataset(training_data, use_augmentation=True)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)

    log_to_file(f"Training on {len(dataset)} samples (8-way augmentation sampled online).")

    total_loss = 0
    total_steps = 0
    start_time = time.time()

    epochs = 1
    for epoch in range(epochs):
        for state, pi, z in dataloader:
            state, pi, z = state.to(device), pi.to(device), z.to(device)

            optimizer.zero_grad()
            p_logits, v = model(state)

            loss_v = F.mse_loss(v.view(-1), z)
            loss_p = -torch.mean(torch.sum(pi * F.log_softmax(p_logits, dim=1), dim=1))
            loss = loss_v + loss_p

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_steps += 1

        scheduler.step()

    avg_loss = total_loss / total_steps if total_steps > 0 else 0
    duration = time.time() - start_time

    save_dict = {
        'model_state': model.state_dict(),
        'model_config': {
            'num_filters': num_filters,
            'board_size': BOARD_SIZE,
            'loop_index': current_loop,
            'loss_avg': avg_loss
        }
    }
    torch.save(save_dict, save_path)

    del model, optimizer, scheduler, dataloader, dataset
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return f"Loss: {avg_loss:.4f} ({duration:.1f}s)"
