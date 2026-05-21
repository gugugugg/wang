# -*- coding: utf-8 -*-
import numpy as np


class GobangEnv:
    def __init__(self, board_size=15):
        self.board_size = board_size
        self.board = np.zeros((board_size, board_size), dtype=int)
        self.current_player = 1
        self.steps = 0
        self.last_move = None

    def reset(self, randomize_opening=True):
        self.board.fill(0)
        self.current_player = 1
        self.steps = 0
        self.last_move = None

        if randomize_opening:
            num_initial = np.random.randint(0, 4)
            center = self.board_size // 2
            for _ in range(num_initial):
                for _attempt in range(100):
                    r = np.random.randint(center - 2, center + 3)
                    c = np.random.randint(center - 2, center + 3)
                    if self.board[r, c] == 0:
                        self.board[r, c] = self.current_player
                        self.current_player = -self.current_player
                        self.steps += 1
                        break
        return self.board.copy()

    def get_valid_moves(self):
        return np.flatnonzero(self.board == 0)

    def step(self, action):
        r, c = divmod(action, self.board_size)
        if self.board[r, c] != 0:
            raise ValueError(f"Invalid move: {r}, {c}")

        self.board[r, c] = self.current_player
        self.steps += 1
        self.last_move = (r, c)

        done = False
        reward = 0.0
        if self.check_win(r, c):
            done = True
            reward = 1.0
        elif self.steps == self.board_size ** 2:
            done = True

        self.current_player = -self.current_player
        return self.board.copy(), reward, done

    def check_win(self, r, c):
        color = self.board[r, c]
        if color == 0:
            return False

        bs = self.board_size
        for dr, dc in ((0, 1), (1, 0), (1, 1), (1, -1)):
            count = 1
            for i in range(1, 5):
                nr, nc = r + dr * i, c + dc * i
                if 0 <= nr < bs and 0 <= nc < bs and self.board[nr, nc] == color:
                    count += 1
                else:
                    break
            for i in range(1, 5):
                nr, nc = r - dr * i, c - dc * i
                if 0 <= nr < bs and 0 <= nc < bs and self.board[nr, nc] == color:
                    count += 1
                else:
                    break
            if count >= 5:
                return True
        return False
