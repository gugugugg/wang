# -*- coding: utf-8 -*-
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os
import subprocess
import sys

import pygame
import torch


ROOT = Path(__file__).resolve().parents[2]
RESULTS_PATH = ROOT / "artifacts" / "base_models" / "TRAINING_RESULTS.md"
BASE_DIR = ROOT / "artifacts" / "base_models"
DATASET_PATH = ROOT / "data" / "human_games.txt"
SELF_PLAY_DIR = ROOT / "artifacts" / "self_play"
WINDOWS_FONTS = Path(os.environ.get("WINDIR", "C:\\Windows")) / "Fonts"


@dataclass
class MetricPoint:
    scale: str
    filters: int
    epoch: int
    checkpoint: str
    train_loss: float
    train_top1: float
    val_loss: float
    val_top1: float


PALETTE = {
    "bg": (15, 18, 24),
    "panel": (25, 30, 38),
    "panel_2": (31, 38, 48),
    "line": (68, 76, 90),
    "text": (232, 236, 243),
    "muted": (150, 160, 174),
    "accent": (72, 161, 255),
    "green": (87, 196, 128),
    "amber": (241, 176, 72),
    "red": (232, 92, 92),
    "s16": (87, 196, 128),
    "m64": (72, 161, 255),
    "l128": (241, 176, 72),
}


def parse_results(path: Path = RESULTS_PATH) -> list[MetricPoint]:
    points: list[MetricPoint] = []
    if not path.exists():
        return points
    for line in path.read_text(encoding="utf-8").splitlines():
        if "`" not in line or "base_" not in line:
            continue
        cols = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cols) != 8 or cols[0] in {"Scale", "---"}:
            continue
        try:
            scale = cols[0]
            filters = int(cols[1])
            epoch = int(cols[2])
            checkpoint = cols[3].replace("`", "")
            train_loss = float(cols[4])
            train_top1 = float(cols[5])
            val_loss = float(cols[6])
            val_top1 = float(cols[7])
        except ValueError:
            continue
        if epoch in {1, 3, 5, 10}:
            points.append(MetricPoint(scale, filters, epoch, checkpoint, train_loss, train_top1, val_loss, val_top1))
    return points


def load_runtime_info() -> dict[str, str]:
    info = {
        "torch": torch.__version__,
        "cuda": "yes" if torch.cuda.is_available() else "no",
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
        "dataset": f"{DATASET_PATH.stat().st_size / 1024 / 1024:.1f} MB" if DATASET_PATH.exists() else "missing",
    }
    return info


def load_font(name: str, size: int, bold: bool = False) -> pygame.font.Font:
    """Load a Windows font file directly, avoiding pygame SysFont registry parsing bugs."""
    candidates = {
        "ui": ["segoeuib.ttf" if bold else "segoeui.ttf", "arialbd.ttf" if bold else "arial.ttf"],
        "mono": ["consolab.ttf" if bold else "consola.ttf", "courbd.ttf" if bold else "cour.ttf"],
    }.get(name, [])

    for filename in candidates:
        path = WINDOWS_FONTS / filename
        if path.exists():
            return pygame.font.Font(str(path), size)
    return pygame.font.Font(None, size)


class DashboardApp:
    def __init__(self):
        pygame.init()
        pygame.display.set_caption("GobangAI Research Dashboard")
        self.screen = pygame.display.set_mode((1280, 780), pygame.RESIZABLE)
        self.clock = pygame.time.Clock()
        self.points = parse_results()
        self.info = load_runtime_info()
        self.checkpoints = self.find_checkpoints()
        self.selected_checkpoint = self.default_checkpoint_index()
        self.cycles_options = [1, 3, 5, 10, 30, 50, 100]
        self.games_options = [5, 10, 20, 50, 100]
        self.cycles_index = 3
        self.games_index = 2
        self.controls: dict[str, pygame.Rect] = {}
        self.train_process = None
        self.train_log_path: Path | None = None
        self.status = "Ready"
        self.font = load_font("ui", 18)
        self.font_small = load_font("ui", 14)
        self.font_title = load_font("ui", 28, bold=True)
        self.font_mono = load_font("mono", 15)

    def run(self):
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        pygame.quit()
                        return
                    if event.key == pygame.K_r:
                        self.points = parse_results()
                        self.info = load_runtime_info()
                        self.checkpoints = self.find_checkpoints()
                        self.status = "Reloaded results"
                    if event.key == pygame.K_o:
                        self.open_results()
                    if event.key == pygame.K_RETURN:
                        self.start_self_play()
                if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    self.handle_click(event.pos)

            self.poll_training()
            self.draw()
            pygame.display.flip()
            self.clock.tick(60)

    def find_checkpoints(self):
        if not BASE_DIR.exists():
            return []
        return sorted(BASE_DIR.glob("*/*.pth"), key=lambda p: (p.parent.name, p.name))

    def default_checkpoint_index(self):
        for i, path in enumerate(self.checkpoints):
            if path.name.endswith("E10.pth") and path.parent.name == "M64":
                return i
        return max(0, len(self.checkpoints) - 1)

    def selected_checkpoint_path(self):
        if not self.checkpoints:
            return None
        self.selected_checkpoint = max(0, min(self.selected_checkpoint, len(self.checkpoints) - 1))
        return self.checkpoints[self.selected_checkpoint]

    def selected_cycles(self):
        return self.cycles_options[self.cycles_index]

    def selected_games(self):
        return self.games_options[self.games_index]

    def self_play_output_dir(self):
        ckpt = self.selected_checkpoint_path()
        if ckpt is None:
            return None
        return SELF_PLAY_DIR / ckpt.parent.name / ckpt.stem

    def self_play_command(self):
        ckpt = self.selected_checkpoint_path()
        out_dir = self.self_play_output_dir()
        if ckpt is None or out_dir is None:
            return ""
        return (
            f"uv run train-self-play --checkpoint {ckpt.relative_to(ROOT)} "
            f"--output-dir {out_dir.relative_to(ROOT)} --cycles {self.selected_cycles()} "
            f"--games-per-cycle {self.selected_games()}"
        )

    def handle_click(self, pos):
        for name, rect in self.controls.items():
            if rect.collidepoint(pos):
                if name == "model_prev" and self.checkpoints:
                    self.selected_checkpoint = (self.selected_checkpoint - 1) % len(self.checkpoints)
                    self.status = "Selected previous checkpoint"
                elif name == "model_next" and self.checkpoints:
                    self.selected_checkpoint = (self.selected_checkpoint + 1) % len(self.checkpoints)
                    self.status = "Selected next checkpoint"
                elif name == "cycles_dec":
                    self.cycles_index = max(0, self.cycles_index - 1)
                elif name == "cycles_inc":
                    self.cycles_index = min(len(self.cycles_options) - 1, self.cycles_index + 1)
                elif name == "games_dec":
                    self.games_index = max(0, self.games_index - 1)
                elif name == "games_inc":
                    self.games_index = min(len(self.games_options) - 1, self.games_index + 1)
                elif name == "start":
                    self.start_self_play()
                return

    def start_self_play(self):
        if self.train_process and self.train_process.poll() is None:
            self.status = "Self-play training is already running"
            return
        ckpt = self.selected_checkpoint_path()
        out_dir = self.self_play_output_dir()
        if ckpt is None or out_dir is None:
            self.status = "No checkpoint available"
            return

        out_dir.mkdir(parents=True, exist_ok=True)
        self.train_log_path = out_dir / "ui_run.log"
        log_fp = self.train_log_path.open("a", encoding="utf-8")
        env = os.environ.copy()
        env.setdefault("UV_CACHE_DIR", str(ROOT / ".uv-cache"))
        args = [
            "uv",
            "run",
            "train-self-play",
            "--checkpoint",
            str(ckpt),
            "--output-dir",
            str(out_dir),
            "--cycles",
            str(self.selected_cycles()),
            "--games-per-cycle",
            str(self.selected_games()),
        ]
        try:
            self.train_process = subprocess.Popen(args, cwd=str(ROOT), env=env, stdout=log_fp, stderr=subprocess.STDOUT)
            self.status = f"Started self-play: pid={self.train_process.pid}, log={self.train_log_path.relative_to(ROOT)}"
        except Exception as exc:
            self.status = f"Start failed: {exc}"
        finally:
            log_fp.close()

    def poll_training(self):
        if not self.train_process:
            return
        code = self.train_process.poll()
        if code is not None:
            self.status = f"Self-play finished with exit code {code}"
            self.train_process = None

    def open_results(self):
        if RESULTS_PATH.exists():
            try:
                if sys.platform.startswith("win"):
                    subprocess.Popen(["notepad.exe", str(RESULTS_PATH)])
                self.status = f"Opened {RESULTS_PATH.name}"
            except Exception as exc:
                self.status = f"Open failed: {exc}"

    def draw(self):
        w, h = self.screen.get_size()
        self.screen.fill(PALETTE["bg"])
        self.draw_header(w)

        left = pygame.Rect(28, 92, int(w * 0.58), h - 126)
        right = pygame.Rect(left.right + 20, 92, w - left.right - 48, h - 126)
        self.panel(left)
        self.panel(right)

        self.draw_chart(left.inflate(-34, -42))
        self.draw_side_panel(right.inflate(-30, -32))
        self.draw_footer(w, h)

    def draw_header(self, width):
        self.text("GobangAI Research Dashboard", 28, 24, self.font_title)
        subtitle = "Expert distillation checkpoints by model scale and epoch"
        self.text(subtitle, 30, 58, self.font_small, PALETTE["muted"])
        badge = pygame.Rect(width - 315, 24, 286, 42)
        pygame.draw.rect(self.screen, PALETTE["panel"], badge, border_radius=8)
        device = self.info.get("device", "unknown")
        self.text(f"CUDA: {self.info.get('cuda')} | {device}", badge.x + 14, badge.y + 11, self.font_small, PALETTE["green"])

    def panel(self, rect):
        pygame.draw.rect(self.screen, PALETTE["panel"], rect, border_radius=8)
        pygame.draw.rect(self.screen, PALETTE["line"], rect, width=1, border_radius=8)

    def draw_chart(self, rect):
        self.text("Validation Top-1 Accuracy", rect.x, rect.y - 4, self.font, PALETTE["text"])
        plot = pygame.Rect(rect.x, rect.y + 36, rect.width, int(rect.height * 0.58))
        pygame.draw.rect(self.screen, PALETTE["panel_2"], plot, border_radius=6)
        pygame.draw.rect(self.screen, PALETTE["line"], plot, width=1, border_radius=6)

        epochs = [1, 3, 5, 10]
        y_min, y_max = 0.68, 0.88
        for i in range(5):
            y = plot.bottom - int((i / 4) * plot.height)
            pygame.draw.line(self.screen, (42, 49, 60), (plot.x + 44, y), (plot.right - 18, y), 1)
            val = y_min + (i / 4) * (y_max - y_min)
            self.text(f"{val:.2f}", plot.x + 8, y - 8, self.font_small, PALETTE["muted"])

        for e in epochs:
            x = self.x_for_epoch(e, plot)
            pygame.draw.line(self.screen, (42, 49, 60), (x, plot.y + 14), (x, plot.bottom - 28), 1)
            self.text(f"E{e:02d}", x - 14, plot.bottom - 22, self.font_small, PALETTE["muted"])

        by_scale = {"S": [], "M": [], "L": []}
        for p in self.points:
            if p.epoch in epochs:
                by_scale[p.scale].append(p)

        for scale, pts in by_scale.items():
            pts = sorted(pts, key=lambda p: p.epoch)
            color = PALETTE[{"S": "s16", "M": "m64", "L": "l128"}[scale]]
            coords = [(self.x_for_epoch(p.epoch, plot), self.y_for_value(p.val_top1, plot, y_min, y_max)) for p in pts]
            if len(coords) > 1:
                pygame.draw.lines(self.screen, color, False, coords, 3)
            for point, p in zip(coords, pts):
                pygame.draw.circle(self.screen, color, point, 5)
                self.text(f"{p.val_top1:.3f}", point[0] + 7, point[1] - 18, self.font_small, color)

        legend_y = plot.bottom + 20
        for i, (label, key) in enumerate([("S16", "s16"), ("M64", "m64"), ("L128", "l128")]):
            x = plot.x + i * 100
            pygame.draw.circle(self.screen, PALETTE[key], (x + 8, legend_y + 8), 5)
            self.text(label, x + 22, legend_y, self.font_small)

        self.draw_table(pygame.Rect(rect.x, legend_y + 42, rect.width, rect.bottom - legend_y - 42))

    def draw_table(self, rect):
        headers = ["Scale", "E01", "E03", "E05", "E10"]
        col_w = rect.width // len(headers)
        for i, head in enumerate(headers):
            self.text(head, rect.x + i * col_w + 8, rect.y, self.font_small, PALETTE["muted"])
        y = rect.y + 28
        for scale in ["S", "M", "L"]:
            color = PALETTE[{"S": "s16", "M": "m64", "L": "l128"}[scale]]
            self.text({"S": "S16", "M": "M64", "L": "L128"}[scale], rect.x + 8, y, self.font_mono, color)
            pts = {p.epoch: p for p in self.points if p.scale == scale}
            for i, epoch in enumerate([1, 3, 5, 10], start=1):
                p = pts.get(epoch)
                text = f"{p.val_top1:.4f}" if p else "--"
                self.text(text, rect.x + i * col_w + 8, y, self.font_mono)
            y += 32

    def draw_side_panel(self, rect):
        self.text("Checkpoint Status", rect.x, rect.y, self.font)
        y = rect.y + 38
        for scale, label in [("S16", "S16"), ("M64", "M64"), ("L128", "L128")]:
            folder = BASE_DIR / scale
            files = sorted(folder.glob("*.pth")) if folder.exists() else []
            color = PALETTE["green"] if files else PALETTE["red"]
            pygame.draw.rect(self.screen, PALETTE["panel_2"], (rect.x, y, rect.width, 58), border_radius=6)
            self.text(label, rect.x + 14, y + 9, self.font, color)
            self.text(f"{len(files)} checkpoints", rect.x + 88, y + 12, self.font_small, PALETTE["muted"])
            latest = files[-1].name if files else "missing"
            self.text(latest, rect.x + 14, y + 34, self.font_mono, PALETTE["text"])
            y += 72

        self.text("Self-Play Training", rect.x, y + 8, self.font)
        y += 44
        self.draw_self_play_controls(rect, y)
        y += 260

        self.text("Environment", rect.x, y + 4, self.font)
        y += 36
        for key in ["torch", "cuda", "device", "dataset"]:
            self.text(f"{key:>7}: {self.info.get(key)}", rect.x + 4, y, self.font_mono, PALETTE["muted"])
            y += 24

        y += 10
        commands = ["Enter/click Start: run", "R: reload", "O: open metrics", "Esc: quit"]
        for cmd in commands:
            self.text(cmd, rect.x + 4, y, self.font_mono, PALETTE["muted"])
            y += 24

    def draw_self_play_controls(self, rect, y):
        self.controls.clear()
        box = pygame.Rect(rect.x, y, rect.width, 236)
        pygame.draw.rect(self.screen, PALETTE["panel_2"], box, border_radius=6)
        pygame.draw.rect(self.screen, PALETTE["line"], box, width=1, border_radius=6)

        ckpt = self.selected_checkpoint_path()
        model_name = f"{ckpt.parent.name}/{ckpt.name}" if ckpt else "missing"
        self.text("Model", box.x + 14, box.y + 12, self.font_small, PALETTE["muted"])
        self.text(model_name, box.x + 14, box.y + 34, self.font_mono, PALETTE["text"])
        self.icon_button("model_prev", "<", box.right - 78, box.y + 28, 30, 28)
        self.icon_button("model_next", ">", box.right - 42, box.y + 28, 30, 28)

        row_y = box.y + 76
        self.value_control("cycles", "Cycles", self.selected_cycles(), box.x + 14, row_y, box.width - 28)
        self.value_control("games", "Games / cycle", self.selected_games(), box.x + 14, row_y + 54, box.width - 28)

        command = self.self_play_command()
        self.text("Command", box.x + 14, box.y + 154, self.font_small, PALETTE["muted"])
        self.text(self.compact_text(command, 48), box.x + 14, box.y + 176, self.font_mono, PALETTE["text"])

        start_color = PALETTE["green"] if not (self.train_process and self.train_process.poll() is None) else PALETTE["amber"]
        start_rect = pygame.Rect(box.x + 14, box.bottom - 42, 132, 30)
        pygame.draw.rect(self.screen, start_color, start_rect, border_radius=6)
        self.text("Start Training", start_rect.x + 14, start_rect.y + 6, self.font_small, (10, 16, 20))
        self.controls["start"] = start_rect
        if self.train_log_path:
            self.text(self.compact_text(f"log: {self.train_log_path.relative_to(ROOT)}", 42), start_rect.right + 14, start_rect.y + 7, self.font_small, PALETTE["muted"])

    def value_control(self, key, label, value, x, y, width):
        self.text(label, x, y, self.font_small, PALETTE["muted"])
        self.icon_button(f"{key}_dec", "-", x + width - 102, y - 4, 30, 28)
        value_rect = pygame.Rect(x + width - 66, y - 4, 36, 28)
        pygame.draw.rect(self.screen, PALETTE["bg"], value_rect, border_radius=6)
        pygame.draw.rect(self.screen, PALETTE["line"], value_rect, width=1, border_radius=6)
        self.text(str(value), value_rect.centerx - 8, value_rect.y + 6, self.font_small, PALETTE["text"])
        self.icon_button(f"{key}_inc", "+", x + width - 24, y - 4, 30, 28)

    def icon_button(self, name, label, x, y, width, height):
        rect = pygame.Rect(x, y, width, height)
        pygame.draw.rect(self.screen, PALETTE["bg"], rect, border_radius=6)
        pygame.draw.rect(self.screen, PALETTE["line"], rect, width=1, border_radius=6)
        surf = self.font_small.render(label, True, PALETTE["text"])
        self.screen.blit(surf, surf.get_rect(center=rect.center))
        self.controls[name] = rect

    def compact_text(self, value, max_chars):
        value = str(value)
        if len(value) <= max_chars:
            return value
        return value[: max_chars - 3] + "..."

    def draw_footer(self, width, height):
        pygame.draw.line(self.screen, PALETTE["line"], (28, height - 38), (width - 28, height - 38), 1)
        self.text(self.status, 30, height - 28, self.font_small, PALETTE["muted"])
        self.text(str(RESULTS_PATH.relative_to(ROOT)), width - 390, height - 28, self.font_small, PALETTE["muted"])

    def x_for_epoch(self, epoch, plot):
        pos = {1: 0.05, 3: 0.32, 5: 0.55, 10: 0.93}[epoch]
        return plot.x + 44 + int(pos * (plot.width - 72))

    def y_for_value(self, value, plot, y_min, y_max):
        value = max(y_min, min(y_max, value))
        return plot.bottom - 28 - int(((value - y_min) / (y_max - y_min)) * (plot.height - 48))

    def text(self, content, x, y, font, color=PALETTE["text"]):
        surf = font.render(str(content), True, color)
        self.screen.blit(surf, (x, y))


def main():
    DashboardApp().run()


if __name__ == "__main__":
    main()
