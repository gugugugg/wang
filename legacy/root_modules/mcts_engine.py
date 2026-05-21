# -*- coding: utf-8 -*-
import copy
import numpy as np
import torch
from numba import jit


@jit(nopython=True)
def calculate_ucb(q, p, n, total, c):
    """Compute the UCB score used by PUCT selection."""
    return q + c * p * np.sqrt(total) / (1.0 + n)


class TreeNode:
    def __init__(self, parent, prior):
        self.parent = parent
        self.children = {}
        self.n = 0
        self.Q = 0.0
        self.P = prior

    def expand(self, priors):
        """Create child nodes from policy priors."""
        for act, prob in priors:
            if act not in self.children:
                self.children[act] = TreeNode(self, prob)

    def select(self, c):
        """Select the child with the highest UCB score."""
        items = list(self.children.items())
        acts = np.array([k for k, v in items])
        nodes = [v for k, v in items]

        scores = calculate_ucb(
            np.array([n.Q for n in nodes]),
            np.array([n.P for n in nodes]),
            np.array([n.n for n in nodes]),
            self.n,
            c
        )
        idx = np.argmax(scores)
        return acts[idx], nodes[idx]

    def update(self, val):
        """Update visit count and mean value."""
        self.n += 1
        self.Q += (val - self.Q) / self.n

    def is_leaf(self):
        return len(self.children) == 0


class MCTS:
    def __init__(self, net, device, c=5, n_playout=400):
        self.net = net
        self.device = device
        self.c = c
        self.n_playout = n_playout

    def get_move_probs(self, state, temp=1e-3, is_self_play=False):
        """Run MCTS and return legal actions plus a 225-position policy vector."""
        self.root = TreeNode(None, 1.0)

        feat = self._to_feat(state)
        with torch.no_grad():
            p_log, v = self.net(feat)

        probs = torch.exp(p_log).cpu().numpy().flatten()
        valid_moves = state.get_valid_moves()

        if is_self_play:
            noise = np.random.dirichlet([0.03] * len(valid_moves))
            p_sum = np.sum(probs[valid_moves])
            priors = []
            for i, act in enumerate(valid_moves):
                p_val = 0.75 * (probs[act] / p_sum) + 0.25 * noise[i]
                priors.append((act, p_val))
        else:
            p_sum = np.sum(probs[valid_moves])
            priors = [(act, probs[act] / p_sum) for act in valid_moves]

        self.root.expand(priors)

        for _ in range(self.n_playout):
            node = self.root
            sim_state = copy.deepcopy(state)

            while not node.is_leaf():
                act, node = node.select(self.c)
                sim_state.step(act)

            feat_leaf = self._to_feat(sim_state)
            with torch.no_grad():
                p_log_leaf, v_leaf = self.net(feat_leaf)

            p_leaf = torch.exp(p_log_leaf).cpu().numpy().flatten()
            v_val = v_leaf.item()

            if not sim_state.check_win(*sim_state.last_move) if sim_state.last_move else False:
                valid_leaf = sim_state.get_valid_moves()
                if len(valid_leaf) > 0:
                    ps = p_leaf[valid_leaf]
                    node.expand(zip(valid_leaf, ps / np.sum(ps)))
            else:
                v_val = 1.0

            while node:
                node.update(v_val)
                v_val = -v_val
                node = node.parent

        acts = list(self.root.children.keys())
        counts = np.array([n.n for n in self.root.children.values()])

        if temp == 1e-3:
            best_idx = np.argmax(counts)
            pi = np.zeros(225)
            pi[acts[best_idx]] = 1.0
        else:
            counts_temp = counts ** (1.0 / temp)
            probs_final = counts_temp / np.sum(counts_temp)
            pi = np.zeros(225)
            pi[acts] = probs_final

        return acts, pi

    def _to_feat(self, state):
        """Build 3 feature planes: current stones, opponent stones, color plane."""
        f = np.zeros((3, 15, 15), dtype=np.float32)
        f[0][state.board == state.current_player] = 1.0
        f[1][state.board == -state.current_player] = 1.0
        if state.current_player == 1:
            f[2][:, :] = 1.0
        return torch.from_numpy(f).unsqueeze(0).to(self.device)
