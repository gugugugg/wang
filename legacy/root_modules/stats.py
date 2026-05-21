# -*- coding: utf-8 -*-
import threading, time, os, numpy as np
from collections import deque

class TrainingStats:
    def __init__(self):
        self.lock = threading.Lock()
        self.losses = deque(maxlen=500)
        self.train_win_rates = deque(maxlen=2000)
        self.recent_results = deque(maxlen=100)
        self.eval_scores = []
        self.board_state = None
        self.gui_logs = deque(maxlen=30) 
        self.log_file = "system_log.txt"
        self.progress = 0.0
        self.eta = "--:--"
        self.device_info = "System Ready"
        self.running = True
        
        # 初始化清空日志
        with open(self.log_file, "w", encoding="utf-8") as f:
            f.write(f"=== Session Started: {time.ctime()} ===\n")

        self.log_thread = threading.Thread(target=self._monitor_log_file, daemon=True)
        self.log_thread.start()

    def _monitor_log_file(self):
        last_pos = 0
        while self.running:
            if os.path.exists(self.log_file):
                try:
                    # [学术增强] 定期检查文件大小，防止长周期训练导致 I/O 锁死
                    if os.path.getsize(self.log_file) > 5 * 1024 * 1024: # 5MB 阈值
                        with self.lock:
                            with open(self.log_file, "w", encoding="utf-8") as f:
                                f.write(f"--- Log Rotated at {time.ctime()} ---\n")
                        last_pos = 0
                        continue

                    with open(self.log_file, "r", encoding="utf-8", errors='replace') as f:
                        f.seek(last_pos)
                        new_lines = f.readlines()
                        last_pos = f.tell()
                        if new_lines:
                            with self.lock:
                                for line in new_lines:
                                    if line.strip(): self.gui_logs.append(line.strip())
                except: pass
            time.sleep(0.5)

    def record_step(self, loss):
        with self.lock: self.losses.append(loss)

    def record_eval(self, checkpoint, score):
        with self.lock: self.eval_scores.append(score)

    def set_device_status(self, status):
        with self.lock: self.device_info = status

    def add_log(self, message):
        t_str = time.strftime("[%H:%M:%S]")
        try:
            with open(self.log_file, "a", encoding="utf-8") as f:
                f.write(f"{t_str} {message}\n")
        except: pass

    def set_progress(self, value, eta_str):
        with self.lock:
            self.progress = max(0.0, min(1.0, value))
            self.eta = eta_str

    def update_board(self, board):
        with self.lock: self.board_state = board.copy()

    def clear_board(self):
        with self.lock: self.board_state = np.zeros((15, 15), dtype=int)