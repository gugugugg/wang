# -*- coding: utf-8 -*-
import numpy as np


BOARD_SIZE = 15
ACTION_SIZE = BOARD_SIZE * BOARD_SIZE


def state_to_feature(board, current_player, board_size=BOARD_SIZE):
    """Build feature planes: current stones, opponent stones, and side-to-move."""
    f = np.zeros((3, board_size, board_size), dtype=np.float32)
    f[0][board == current_player] = 1.0
    f[1][board == -current_player] = 1.0
    if current_player == 1:
        f[2][:, :] = 1.0
    return f


def policy_one_hot(move, action_size=ACTION_SIZE):
    pi = np.zeros(action_size, dtype=np.float32)
    pi[move] = 1.0
    return pi


def augment_state_policy(state, pi, mode, board_size=BOARD_SIZE):
    """Apply one of 8 square-board symmetries to a feature tensor and policy."""
    pi_board = pi.reshape(board_size, board_size)
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
