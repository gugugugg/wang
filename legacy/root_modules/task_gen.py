# -*- coding: utf-8 -*-
import os
import sys
import time
import subprocess
import threading
import queue
import numpy as np
from multiprocessing import Pool, freeze_support
from functools import partial


def get_resource_path(relative_path):
    """Resolve a project resource path, including PyInstaller bundles."""
    if hasattr(sys, '_MEIPASS'):
        return os.path.join(sys._MEIPASS, relative_path)
    return os.path.join(os.path.abspath("."), relative_path)


YIXIN_EXE_PATH = get_resource_path(os.path.join("assets", "engines", "pbrain-Yixin2018.exe"))
SAVE_DIR = os.path.join(os.getcwd(), "raw_sgf_data")


class EngineWrapper:
    """Persistent wrapper for the Yixin pbrain engine protocol."""

    def __init__(self, exe_path):
        self.output_queue = queue.Queue()
        self.is_running = True
        flags = 0x08000000 if os.name == 'nt' else 0
        try:
            self.proc = subprocess.Popen(
                exe_path,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
                creationflags=flags,
                encoding='utf-8',
                errors='ignore'
            )
            threading.Thread(target=self._read, daemon=True).start()
        except Exception as e:
            print(f"Engine start failed: {e}")
            self.proc = None

    def _read(self):
        while self.is_running and self.proc and self.proc.stdout:
            try:
                line = self.proc.stdout.readline()
                if line:
                    self.output_queue.put(line.strip())
                else:
                    break
            except:
                break

    def send(self, cmd):
        if self.proc and self.proc.stdin:
            try:
                self.proc.stdin.write(f"{cmd}\n")
                self.proc.stdin.flush()
            except:
                pass

    def get_move(self, timeout=15.0):
        start = time.time()
        while time.time() - start < timeout:
            try:
                line = self.output_queue.get(timeout=1.0)
                if "," in line and line[0].isdigit():
                    return line
            except queue.Empty:
                continue
        return None

    def terminate(self):
        self.is_running = False
        if self.proc:
            try:
                self.proc.stdin.close()
                self.proc.terminate()
                self.proc.wait(timeout=2)
            except:
                try:
                    self.proc.kill()
                except:
                    pass


def check_winner(board):
    """Return the winning color, or 0 if the game has not ended."""
    for x in range(15):
        for y in range(15):
            if board[x][y] == 0:
                continue
            color = board[x][y]
            for dx, dy in [(1, 0), (0, 1), (1, 1), (1, -1)]:
                count = 1
                for k in range(1, 5):
                    nx, ny = x + k * dx, y + k * dy
                    if 0 <= nx < 15 and 0 <= ny < 15 and board[nx][ny] == color:
                        count += 1
                    else:
                        break
                if count >= 5:
                    return color
    return 0


def run_one_game(game_id, timeout_ms=500):
    """Generate one expert game by matching two Yixin processes."""
    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR, exist_ok=True)

    black = EngineWrapper(YIXIN_EXE_PATH)
    white = EngineWrapper(YIXIN_EXE_PATH)

    if not black.proc or not white.proc:
        return

    board = np.zeros((15, 15), dtype=int)
    moves, winner = [], 0

    try:
        for p in [black, white]:
            p.send("START 15")
            p.send(f"INFO timeout_turn {timeout_ms}")

        black.send("BEGIN")
        curr, other, curr_color = black, white, 1

        for step in range(225):
            move = curr.get_move(timeout=25.0 if step < 2 else 12.0)
            if not move:
                break

            try:
                x, y = map(int, move.split(','))
                if board[x][y] != 0:
                    break
                board[x][y] = curr_color
                moves.append(move)
                winner = check_winner(board)
                if winner != 0:
                    break
            except:
                break

            other.send(f"TURN {move}")
            curr, other, curr_color = other, curr, -curr_color

        if len(moves) >= 5 and winner != 0:
            res = "B+R" if winner == 1 else "W+R"
            sgf = f"(;SZ[15]RE[{res}]"
            for i, m in enumerate(moves):
                x, y = map(int, m.split(','))
                sgf += f";{'B' if i % 2 == 0 else 'W'}[{chr(97 + x)}{chr(97 + y)}]"
            sgf += ")"
            with open(os.path.join(SAVE_DIR, f"G{game_id}_{int(time.time())}.sgf"), "w") as f:
                f.write(sgf)
    finally:
        black.terminate()
        white.terminate()


def run_generation_task(count_str):
    """Generate expert games in parallel."""
    freeze_support()
    count = int(count_str)
    with Pool(processes=8) as pool:
        pool.map(partial(run_one_game, timeout_ms=500), range(count))
