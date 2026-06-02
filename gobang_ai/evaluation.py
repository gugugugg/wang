# -*- coding: utf-8 -*-
from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
import random
import time

import numpy as np
import torch

from gobang_ai.env import GobangEnv
from gobang_ai.features import state_to_feature
from gobang_ai.self_play import load_model_from_checkpoint
from gobang_ai.training import get_device


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


@dataclass(frozen=True)
class MatchSide:
    name: str
    kind: str
    path: str = ""
    model: torch.nn.Module | None = None
    rule_ai: RuleBasedAI | None = None


def load_match_side(name: str, checkpoint_path: str | None, device: torch.device) -> MatchSide:
    if checkpoint_path is None or checkpoint_path.lower() == "rule":
        return MatchSide(name=name or "RuleBasedAI", kind="rule", rule_ai=RuleBasedAI(15))

    model, _checkpoint, _config = load_model_from_checkpoint(checkpoint_path, device)
    model.eval()
    return MatchSide(name=name or Path(checkpoint_path).stem, kind="model", path=str(checkpoint_path), model=model)


def get_evaluation_runtime_info(device: torch.device) -> dict[str, str]:
    gpu_name = ""
    if device.type == "cuda":
        gpu_name = torch.cuda.get_device_name(0)
    return {
        "device": str(device),
        "torch_version": torch.__version__,
        "gpu_name": gpu_name,
    }


def choose_greedy_model_action(
    model: torch.nn.Module,
    board: np.ndarray,
    player_color: int,
    valid_moves: np.ndarray,
    device: torch.device,
) -> tuple[int, float]:
    feat = state_to_feature(board, player_color)
    tensor = torch.from_numpy(feat).unsqueeze(0).to(device)
    if device.type == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()
    with torch.no_grad():
        log_policy, _value = model(tensor)
        scores = log_policy[0].detach().cpu().numpy()
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    action = int(valid_moves[np.argmax(scores[valid_moves])])
    return action, elapsed_ms


def choose_side_action(
    side: MatchSide,
    board: np.ndarray,
    player_color: int,
    valid_moves: np.ndarray,
    device: torch.device,
) -> tuple[int, float]:
    if side.kind == "model":
        if side.model is None:
            raise RuntimeError(f"Model side {side.name} has no loaded model")
        return choose_greedy_model_action(side.model, board, player_color, valid_moves, device)

    if side.rule_ai is None:
        raise RuntimeError(f"Rule side {side.name} has no rule AI")
    start = time.perf_counter()
    action = int(side.rule_ai.get_action(board, player_color))
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    return action, elapsed_ms


def warmup_match_side(side: MatchSide, device: torch.device, steps: int = 3) -> None:
    if side.kind != "model" or side.model is None or steps <= 0:
        return
    board = np.zeros((15, 15), dtype=np.int8)
    tensor = torch.from_numpy(state_to_feature(board, 1)).unsqueeze(0).to(device)
    side.model.eval()
    with torch.no_grad():
        for _ in range(steps):
            side.model(tensor)
    if device.type == "cuda":
        torch.cuda.synchronize()


def apply_random_opening(env: GobangEnv, max_opening_moves: int, rng: random.Random) -> list[int]:
    start_moves: list[int] = []
    if max_opening_moves <= 0:
        return start_moves

    center = env.board_size // 2
    count = rng.randint(0, max_opening_moves)
    for _ in range(count):
        candidates = []
        for dr in range(-2, 3):
            for dc in range(-2, 3):
                r = center + dr
                c = center + dc
                if 0 <= r < env.board_size and 0 <= c < env.board_size and env.board[r, c] == 0:
                    candidates.append(r * env.board_size + c)
        if not candidates:
            break
        move = int(rng.choice(candidates))
        _board, _reward, done = env.step(move)
        start_moves.append(move)
        if done:
            break
    return start_moves


def play_single_match_game(
    model_a: MatchSide,
    model_b: MatchSide,
    black_side: MatchSide,
    white_side: MatchSide,
    game_index: int,
    total_games: int,
    experiment_id: str,
    run_id: str,
    seed: int,
    device: torch.device,
    max_opening_moves: int = 0,
) -> dict[str, object]:
    rng = random.Random(seed + game_index)
    env = GobangEnv()
    board = env.reset(randomize_opening=False)
    start_moves = apply_random_opening(env, max_opening_moves=max_opening_moves, rng=rng)
    board = env.board.copy()

    black_total_ms = 0.0
    white_total_ms = 0.0
    black_decisions = 0
    white_decisions = 0
    illegal_moves = 0
    winner = "draw"
    winner_model = ""
    winner_type = "draw"
    start_time = time.perf_counter()

    while True:
        current_side = black_side if env.current_player == 1 else white_side
        valid_moves = env.get_valid_moves()
        if len(valid_moves) == 0:
            winner = "draw"
            winner_type = "draw"
            break

        action, elapsed_ms = choose_side_action(current_side, board, env.current_player, valid_moves, device)
        if env.current_player == 1:
            black_total_ms += elapsed_ms
            black_decisions += 1
        else:
            white_total_ms += elapsed_ms
            white_decisions += 1

        if action not in set(int(x) for x in valid_moves):
            illegal_moves += 1
            winner = "white" if env.current_player == 1 else "black"
            winner_model = white_side.name if env.current_player == 1 else black_side.name
            winner_type = "illegal"
            break

        board, reward, done = env.step(action)
        if done:
            if reward > 0:
                winner = "black" if env.current_player == -1 else "white"
                winner_model = black_side.name if winner == "black" else white_side.name
                winner_type = "five_in_row"
            else:
                winner = "draw"
                winner_type = "draw"
            break

    if winner == "draw":
        result_for_model_a = 0.0
    elif winner_model == model_a.name:
        result_for_model_a = 1.0
    else:
        result_for_model_a = -1.0

    runtime = get_evaluation_runtime_info(device)
    return {
        "experiment_id": experiment_id,
        "run_id": run_id,
        "match_type": "model_vs_rule" if model_a.kind == "rule" or model_b.kind == "rule" else "model_vs_model",
        "game_index": game_index,
        "total_games": total_games,
        "model_a_name": model_a.name,
        "model_a_path": model_a.path,
        "model_b_name": model_b.name,
        "model_b_path": model_b.path,
        "black_player": black_side.name,
        "white_player": white_side.name,
        "winner": winner,
        "winner_model": winner_model,
        "result_for_model_a": result_for_model_a,
        "winner_type": winner_type,
        "moves": env.steps,
        "illegal_moves": illegal_moves,
        "black_infer_ms_avg": black_total_ms / black_decisions if black_decisions else 0.0,
        "white_infer_ms_avg": white_total_ms / white_decisions if white_decisions else 0.0,
        "black_total_infer_ms": black_total_ms,
        "white_total_infer_ms": white_total_ms,
        "game_seconds": time.perf_counter() - start_time,
        "decision_mode": "greedy_argmax",
        "seed": seed,
        "opening_id": f"seed_{seed + game_index}" if max_opening_moves > 0 else "",
        "start_moves": " ".join(str(m) for m in start_moves),
        "device": runtime["device"],
        "torch_version": runtime["torch_version"],
        "gpu_name": runtime["gpu_name"],
        "notes": "minimal_validation" if total_games <= 20 else "",
    }


MATCH_CSV_FIELDS = [
    "experiment_id",
    "run_id",
    "match_type",
    "game_index",
    "total_games",
    "model_a_name",
    "model_a_path",
    "model_b_name",
    "model_b_path",
    "black_player",
    "white_player",
    "winner",
    "winner_model",
    "result_for_model_a",
    "winner_type",
    "moves",
    "illegal_moves",
    "black_infer_ms_avg",
    "white_infer_ms_avg",
    "black_total_infer_ms",
    "white_total_infer_ms",
    "game_seconds",
    "decision_mode",
    "seed",
    "opening_id",
    "start_moves",
    "device",
    "torch_version",
    "gpu_name",
    "notes",
]


def summarize_match_rows(rows: list[dict[str, object]], model_a_name: str) -> dict[str, object]:
    games = len(rows)
    model_a_wins = sum(1 for row in rows if row["winner_model"] == model_a_name)
    draws = sum(1 for row in rows if row["winner"] == "draw")
    losses = games - model_a_wins - draws
    illegal = sum(int(row["illegal_moves"]) for row in rows)
    avg_moves = sum(float(row["moves"]) for row in rows) / games if games else 0.0
    avg_seconds = sum(float(row["game_seconds"]) for row in rows) / games if games else 0.0
    return {
        "games": games,
        "model_a_wins": model_a_wins,
        "model_a_losses": losses,
        "draws": draws,
        "model_a_win_rate": model_a_wins / games if games else 0.0,
        "draw_rate": draws / games if games else 0.0,
        "illegal_moves": illegal,
        "avg_moves": avg_moves,
        "avg_game_seconds": avg_seconds,
    }


def run_match_evaluation(
    model_a_path: str,
    model_b_path: str,
    output_csv: str,
    games: int,
    model_a_name: str | None = None,
    model_b_name: str | None = None,
    seed: int = 42,
    experiment_id: str = "match_eval",
    run_id: str | None = None,
    swap_sides: bool = True,
    max_opening_moves: int = 0,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = get_device()

    model_a = load_match_side(model_a_name or Path(model_a_path).stem, model_a_path, device)
    model_b = load_match_side(model_b_name or ("RuleBasedAI" if model_b_path.lower() == "rule" else Path(model_b_path).stem), model_b_path, device)
    warmup_match_side(model_a, device)
    warmup_match_side(model_b, device)
    run_id = run_id or time.strftime("%Y%m%d_%H%M%S")
    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    for game_index in range(1, games + 1):
        if swap_sides and game_index % 2 == 0:
            black_side, white_side = model_b, model_a
        else:
            black_side, white_side = model_a, model_b
        row = play_single_match_game(
            model_a=model_a,
            model_b=model_b,
            black_side=black_side,
            white_side=white_side,
            game_index=game_index,
            total_games=games,
            experiment_id=experiment_id,
            run_id=run_id,
            seed=seed,
            device=device,
            max_opening_moves=max_opening_moves,
        )
        rows.append(row)

    with output_path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=MATCH_CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    return rows, summarize_match_rows(rows, model_a.name)
