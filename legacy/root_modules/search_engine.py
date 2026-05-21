# -*- coding: utf-8 -*-
import time
import numpy as np
import random
import torch
from gobang_ai.features import state_to_feature

try:
    from numba import jit
    print("Numba acceleration: Enabled")
except ImportError:
    print("Numba acceleration: Disabled")
    def jit(nopython=True):
        def decorator(func): return func
        return decorator

@jit(nopython=True)
def fast_check_win(board, r, c, color, size):
    directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
    for dr, dc in directions:
        cnt = 1
        for k in range(1, 5):
            nr, nc = r + dr * k, c + dc * k
            if 0 <= nr < size and 0 <= nc < size and board[nr, nc] == color: cnt += 1
            else: break
        for k in range(1, 5):
            nr, nc = r - dr * k, c - dc * k
            if 0 <= nr < size and 0 <= nc < size and board[nr, nc] == color: cnt += 1
            else: break
        if cnt >= 5: return True
    return False

class AlphaBetaOptimizer:
    def __init__(self, model, device, board_size=15):
        self.model = model 
        self.device = device
        self.board_size = board_size
        self.tt = {} 
        # [学术增强] 初始化 Zobrist 表: 15x15x2 (0:黑, 1:白)
        self.zobrist_table = np.random.randint(1, 2**63, size=(board_size, board_size, 2), dtype=np.uint64)

    def _to_tensor(self, board, color):
        feat = state_to_feature(board, color, self.board_size)
        return torch.from_numpy(feat).unsqueeze(0).to(self.device)

    def _get_board_hash(self, board):
        """全盘计算初始哈希值"""
        h = np.uint64(0)
        for r in range(self.board_size):
            for c in range(self.board_size):
                if board[r, c] != 0:
                    p_idx = 0 if board[r, c] == 1 else 1
                    h ^= self.zobrist_table[r, c, p_idx]
        return h

    def get_move(self, board, player_color, time_limit=1.0):
        start_time = time.time()
        urgent = self.find_urgent_move(board, player_color)
        if urgent != -1: return urgent
        urgent = self.find_urgent_move(board, -player_color)
        if urgent != -1: return urgent

        candidates = self.get_candidates(board, player_color, top_k=8)
        # [逻辑修复] 使用 size 判断 NumPy 数组
        if hasattr(candidates, 'size'):
            if candidates.size == 0: return random.choice(np.flatnonzero(board == 0))
        
        root_hash = self._get_board_hash(board)
        best_move = candidates[0]
        depth = 1
        try:
            while time.time() - start_time < time_limit * 0.9:
                move, score = self.search_root(board, player_color, candidates, depth, root_hash)
                best_move = move
                if score > 0.95: break
                depth += 1
        except Exception: pass
        return best_move

    def search_root(self, board, color, candidates, depth, current_hash):
        alpha, beta = -float('inf'), float('inf')
        best_val, best_move = -float('inf'), candidates[0]
        for move in candidates:
            r, c = divmod(move, self.board_size)
            p_idx = 0 if color == 1 else 1
            new_hash = current_hash ^ self.zobrist_table[r, c, p_idx]
            
            board[r, c] = color
            val = -self.minimax(board, depth - 1, -beta, -alpha, -color, new_hash)
            board[r, c] = 0
            if val > best_val: best_val, best_move = val, move
            alpha = max(alpha, val)
        return best_move, best_val

    def minimax(self, board, depth, alpha, beta, color, current_hash):
        state_key = int(current_hash) 
        if state_key in self.tt:
            e = self.tt[state_key]
            if e['depth'] >= depth:
                if e['flag'] == 'EXACT': return e['val']
                elif e['flag'] == 'LOWER': alpha = max(alpha, e['val'])
                elif e['flag'] == 'UPPER': beta = min(beta, e['val'])
                if alpha >= beta: return e['val']

        if depth == 0: return self.evaluate_network(board, color)
        
        best_val = -float('inf')
        valid = np.flatnonzero(board == 0)
        for m in valid:
            r, c = divmod(m, self.board_size)
            p_idx = 0 if color == 1 else 1
            new_hash = current_hash ^ self.zobrist_table[r, c, p_idx]
            
            board[r, c] = color
            val = -self.minimax(board, depth - 1, -beta, -alpha, -color, new_hash)
            board[r, c] = 0
            best_val = max(best_val, val); alpha = max(alpha, val)
            if alpha >= beta: break
        
        flag = 'EXACT'
        if best_val <= alpha: flag = 'UPPER'
        elif best_val >= beta: flag = 'LOWER'
        self.tt[state_key] = {'val': best_val, 'depth': depth, 'flag': flag}
        return best_val

    def find_urgent_move(self, board, color):
        empty = np.flatnonzero(board == 0)
        for m in empty:
            r, c = divmod(m, self.board_size)
            board[r, c] = color
            if fast_check_win(board, r, c, color, self.board_size):
                board[r, c] = 0; return m
            board[r, c] = 0
        return -1

    def get_candidates(self, board, color, top_k=8):
        valid = np.flatnonzero(board == 0)
        if len(valid) == 0: return np.array([], dtype=np.int64)
        t = self._to_tensor(board, color)
        with torch.no_grad():
            # [关键修复] 双头网络返回 (logits, value)，取 logits 用于落子候选
            policy_logits, _ = self.model(t)
            q = policy_logits.squeeze(0).cpu().numpy()
        vq = q[valid]
        num = min(len(valid), top_k)
        idx = np.argpartition(vq, -num)[-num:]
        return valid[idx[np.argsort(-vq[idx])]]

    def evaluate_network(self, board, color):
        t = self._to_tensor(board, color)
        with torch.no_grad():
            # [关键修复] 取 Value 头作为局面评分
            _, value = self.model(t)
            return value.item()
