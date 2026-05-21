# -*- coding: utf-8 -*-
from dataclasses import dataclass
from pathlib import Path
import random

import numpy as np
import torch
from torch.utils.data import Dataset

from gobang_ai.features import ACTION_SIZE, BOARD_SIZE, augment_state_policy, policy_one_hot, state_to_feature


@dataclass(frozen=True)
class ExpertGame:
    moves: tuple[int, ...]
    result: float


def load_expert_games(path, board_size=BOARD_SIZE):
    """Load and validate expert games from a moves|result text file."""
    games = []
    skipped = 0
    path = Path(path)
    for raw in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if "|" not in raw:
            skipped += 1
            continue

        moves_text, result_text = raw.split("|", 1)
        try:
            moves = tuple(int(x) for x in moves_text.split())
            result = float(result_text)
        except ValueError:
            skipped += 1
            continue

        if result not in (-1.0, 1.0):
            skipped += 1
            continue

        seen = set()
        valid = True
        for move in moves:
            if move < 0 or move >= board_size * board_size or move in seen:
                valid = False
                break
            seen.add(move)

        if valid and len(moves) > 5:
            games.append(ExpertGame(moves=moves, result=result))
        else:
            skipped += 1

    return games, skipped


class ExpertMoveDataset(Dataset):
    """
    Lazy expert-move dataset.

    Each sample is a board state before the current move, the one-hot current
    move target, and the final result from the side-to-move perspective.
    """

    def __init__(self, games, board_size=BOARD_SIZE, augment=True, max_positions=None, seed=0):
        self.games = list(games)
        self.board_size = board_size
        self.augment = augment
        self.positions = []

        for game_idx, game in enumerate(self.games):
            for move_idx in range(len(game.moves)):
                self.positions.append((game_idx, move_idx))

        if max_positions is not None and max_positions < len(self.positions):
            rng = random.Random(seed)
            self.positions = rng.sample(self.positions, max_positions)

    def __len__(self):
        return len(self.positions)

    def __getitem__(self, idx):
        game_idx, move_idx = self.positions[idx]
        game = self.games[game_idx]

        board = np.zeros((self.board_size, self.board_size), dtype=np.int8)
        current_player = 1

        for move in game.moves[:move_idx]:
            r, c = divmod(move, self.board_size)
            board[r, c] = current_player
            current_player = -current_player

        move = game.moves[move_idx]
        state = state_to_feature(board, current_player, self.board_size)
        pi = policy_one_hot(move, ACTION_SIZE)
        z = np.float32(game.result * current_player)

        if self.augment:
            mode = np.random.randint(0, 8)
            state, pi = augment_state_policy(state, pi, mode, self.board_size)

        return torch.from_numpy(state), torch.from_numpy(pi), torch.tensor(z, dtype=torch.float32)
