# -*- coding: utf-8 -*-
import random

import numpy as np
import torch

from gobang_ai.env import GobangEnv
from gobang_ai.features import state_to_feature


class RuleBasedAI:
    def __init__(self, board_size=15):
        self.size = board_size

    def get_action(self, board, player_color):
        valid = self.get_active_empty_cells(board)
        if not valid:
            return -1

        for m in valid:
            r, c = divmod(m, self.size)
            board[r, c] = player_color
            if self.check_win(board, r, c, player_color):
                board[r, c] = 0
                return m
            board[r, c] = 0

        opponent = -player_color
        for m in valid:
            r, c = divmod(m, self.size)
            board[r, c] = opponent
            if self.check_win(board, r, c, opponent):
                board[r, c] = 0
                return m
            board[r, c] = 0

        return random.choice(valid)

    def get_active_empty_cells(self, board):
        active = set()
        rows, cols = np.where(board != 0)
        if len(rows) == 0:
            return [self.size ** 2 // 2]
        for r, c in zip(rows, cols):
            for dr in range(-2, 3):
                for dc in range(-2, 3):
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < self.size and 0 <= nc < self.size and board[nr, nc] == 0:
                        active.add(nr * self.size + nc)
        return list(active) if active else np.flatnonzero(board == 0).tolist()

    def check_win(self, board, r, c, color):
        for dr, dc in [(0, 1), (1, 0), (1, 1), (1, -1)]:
            count = 1
            for i in range(1, 5):
                nr, nc = r + dr * i, c + dc * i
                if 0 <= nr < self.size and 0 <= nc < self.size and board[nr, nc] == color:
                    count += 1
                else:
                    break
            for i in range(1, 5):
                nr, nc = r - dr * i, c - dc * i
                if 0 <= nr < self.size and 0 <= nc < self.size and board[nr, nc] == color:
                    count += 1
                else:
                    break
            if count >= 5:
                return True
        return False


class Arena:
    def __init__(self, net, device="cpu"):
        self.net = net
        self.device = device
        self.env = GobangEnv()
        self.rule_ai = RuleBasedAI(15)

    def run_evaluation(self, num_games, log_func=print, should_stop_func=None, on_progress=None):
        wins = 0
        for i in range(num_games):
            if should_stop_func and should_stop_func():
                break
            ai_color = 1 if (i % 2 == 0) else -1
            state = self.env.reset(randomize_opening=False)
            done = False
            while not done:
                is_ai_turn = self.env.current_player == ai_color
                valid_moves = self.env.get_valid_moves()
                if is_ai_turn:
                    feat = state_to_feature(state, ai_color)
                    t = torch.from_numpy(feat).unsqueeze(0).to(self.device)
                    with torch.no_grad():
                        log_policy, _ = self.net(t)
                        q = log_policy[0].detach().cpu().numpy()
                    action = int(valid_moves[np.argmax(q[valid_moves])])
                else:
                    action = self.rule_ai.get_action(state, self.env.current_player)

                state, reward, done = self.env.step(action)
                if done and reward > 0 and is_ai_turn:
                    wins += 1

            win_rate = wins / (i + 1)
            if on_progress:
                on_progress(win_rate)
            if log_func:
                log_func(f"eval_game={i + 1} win_rate={win_rate:.3f}")

        return wins / num_games if num_games > 0 else 0
