# -*- coding: utf-8 -*-
import threading
import os
import sys
import time
import gc
import torch
import numpy as np
import task_gen
import task_convert
import task_train
from model import GobangDualHead
from mcts_engine import MCTS
from game_env import GobangEnv
from gobang_ai.features import state_to_feature


class WorkerThread(threading.Thread):
    def __init__(self, task_type, stats, config_data):
        super().__init__()
        self.task_type = task_type
        self.stats = stats
        self.config_data = config_data
        self.running = True
        self.daemon = True
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def run(self):
        try:
            if self.task_type == 'DISTILL':
                self.distill()
            elif self.task_type == 'TRAIN':
                self.train_rl()
        except Exception as e:
            self.stats.add_log(f"Runtime error: {str(e)}")
        finally:
            self.stats.set_device_status("Idle")

    def distill(self):
        """Run the expert distillation pipeline with progress and snapshots."""
        cycles = int(self.config_data.get('distill_cycles', 100))
        games_per_cycle = int(self.config_data.get('distill_games', 500))
        total_target = cycles * games_per_cycle
        save_prefix = self.config_data.get('save_name', 'AAAI_Model')
        start_t = time.time()

        # Use the current working directory as the project output directory.
        work_dir = os.getcwd()

        for c in range(1, cycles + 1):
            if not self.running:
                break

            # 1. Generate expert games.
            task_gen.run_generation_task(str(games_per_cycle))

            # 2. Estimate generation speed and ETA.
            current_count = len([f for f in os.listdir(task_gen.SAVE_DIR) if f.endswith('.sgf')]) if os.path.exists(task_gen.SAVE_DIR) else 0
            elapsed = time.time() - start_t
            if elapsed > 10:
                rate = (current_count / elapsed) * 3600
                eta = (total_target - current_count) / rate if rate > 0 else 0
                info = f"Speed: {int(rate)} g/h | ETA: {eta:.1f}h"
            else:
                info = "Analyzing..."
            self.stats.set_progress(current_count / total_target, info)

            # 3. Convert data and train the selected model scale.
            if task_convert.run_convert_task() > 0:
                data = self._load_expert_data("human_games.txt")
                if data:
                    filters = {'S': 16, 'M': 64, 'L': 128}.get(self.config_data.get('scale', 'M'), 64)

                    # Save the latest checkpoint by overwriting the same file.
                    task_train.run_train_task(f"{save_prefix}_latest.pth", str(filters), c, work_dir, data)

                    # Save periodic snapshots for scaling-law plots.
                    if c % 50 == 0:
                        snap = f"{save_prefix}_snapshot_cycle_{c}.pth"
                        task_train.run_train_task(snap, str(filters), c, work_dir, data)
                        self.stats.add_log(f"Saved research snapshot: {snap}")
                    del data

            gc.collect()
            torch.cuda.empty_cache()

    def train_rl(self):
        """Run self-play reinforcement learning after distillation."""
        loops = int(self.config_data.get('loops', 500))
        filters = int(self.config_data.get('filters', 64))
        save_prefix = self.config_data.get('save_name', 'RL_Model')
        work_dir = os.getcwd()

        for i in range(1, loops + 1):
            if not self.running:
                break

            # Load the latest checkpoint before the next self-play loop.
            model = GobangDualHead(num_filters=filters).to(self.device)
            self._load_latest_weights(model, f"{save_prefix}_latest.pth", filters)
            model.eval()

            # Generate self-play data with MCTS.
            mcts = MCTS(model, self.device, n_playout=400)
            samples = self._generate_self_play_data(mcts, games=20)

            # Train on the newly generated self-play samples.
            task_train.run_train_task(f"{save_prefix}_latest.pth", str(filters), i, work_dir, samples)

            # Save reinforcement-learning snapshots.
            if i % 50 == 0:
                torch.save(model.state_dict(), os.path.join(work_dir, f"{save_prefix}_RL_snap_{i}.pth"))

            del model, mcts, samples
            gc.collect()
            torch.cuda.empty_cache()

    def _generate_self_play_data(self, mcts, games):
        """Generate self-play samples with exploration noise."""
        samples = []
        for _ in range(games):
            env = GobangEnv()
            env.reset(randomize_opening=True)
            history = []
            while True:
                # Use a higher temperature in the opening for exploration.
                temp = 1.0 if env.steps < 30 else 1e-3
                acts, pi = mcts.get_move_probs(env, temp=temp, is_self_play=True)
                history.append((self._state_to_feat(env.board, env.current_player), pi, env.current_player))

                action = np.random.choice(acts, p=pi[acts])
                _, reward, done = env.step(action)
                if done:
                    winner = -env.current_player if reward > 0 else 0
                    for f, p, player in history:
                        z = 0.0 if winner == 0 else (1.0 if player == winner else -1.0)
                        samples.append((f, p, z))
                    break
        return samples

    def _state_to_feat(self, board, curr):
        return state_to_feature(board, curr)

    def _load_latest_weights(self, model, filename, f_size):
        path = os.path.join(os.getcwd(), f"models_scale_{f_size}", filename)
        if os.path.exists(path):
            try:
                ckpt = torch.load(path, map_location=self.device)
                model.load_state_dict(ckpt['model_state'], strict=False)
            except:
                pass

    def _load_expert_data(self, file_path):
        """Load expert records stored as 'moves|result' lines."""
        data = []
        if not os.path.exists(file_path):
            return data
        skipped = 0
        with open(file_path, 'r') as f:
            for line in f.readlines()[-30000:]:
                if '|' not in line:
                    continue
                try:
                    m_str, r_str = line.strip().split('|')
                    moves, res = [int(x) for x in m_str.split()], float(r_str)
                except ValueError:
                    skipped += 1
                    continue

                b = np.zeros((15, 15), dtype=np.int8)
                curr = 1
                game_data = []
                valid_game = True
                for m in moves:
                    if m < 0 or m >= 225:
                        valid_game = False
                        break
                    r, c = divmod(m, 15)
                    if b[r, c] != 0:
                        valid_game = False
                        break

                    pi = np.zeros(225, dtype=np.float32)
                    pi[m] = 1.0
                    game_data.append((self._state_to_feat(b, curr), pi, res * curr))
                    b[r, c] = curr
                    curr = -curr

                if valid_game:
                    data.extend(game_data)
                else:
                    skipped += 1

        if skipped:
            self.stats.add_log(f"Skipped invalid expert games: {skipped}")
        return data
