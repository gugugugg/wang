# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys
import time


APP_NAME = "GobangAI_PyTorch"


def log_dir() -> Path:
    base = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
    path = base / APP_NAME / "logs"
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_log(name: str, message: str) -> None:
    try:
        with (log_dir() / name).open("a", encoding="utf-8") as fp:
            fp.write(message.rstrip() + "\n")
    except Exception:
        pass


def candidate_roots() -> list[Path]:
    candidates: list[Path] = []
    if getattr(sys, "frozen", False):
        candidates.append(Path(sys.executable).resolve().parent)
    candidates.append(Path(__file__).resolve().parent)
    candidates.append(Path.cwd())
    candidates.extend(path.parent for path in candidates[:])
    return candidates


def find_project_root() -> Path:
    seen: set[Path] = set()
    for start in candidate_roots():
        for path in [start, *start.parents]:
            if path in seen:
                continue
            seen.add(path)
            if (path / "gobang_ai" / "ui" / "dashboard.py").exists() and (path / ".venv").exists():
                return path
    raise FileNotFoundError("Cannot find project root containing gobang_ai/ui/dashboard.py and .venv")


def python_executable(root: Path, console: bool = False) -> Path:
    exe_name = "python.exe" if console else "pythonw.exe"
    path = root / ".venv" / "Scripts" / exe_name
    if path.exists():
        return path
    fallback = root / ".venv" / "Scripts" / "python.exe"
    if fallback.exists():
        return fallback
    raise FileNotFoundError(f"Cannot find project Python under {root / '.venv' / 'Scripts'}")


def run_self_test(root: Path) -> int:
    python = python_executable(root, console=True)
    env = os.environ.copy()
    env.setdefault("UV_CACHE_DIR", str(root / ".uv-cache"))
    env.setdefault("SDL_VIDEODRIVER", "dummy")
    command = [
        str(python),
        "-c",
        "from gobang_ai.ui.dashboard import DashboardApp; app=DashboardApp(); print('launcher self-test ok')",
    ]
    proc = subprocess.run(command, cwd=str(root), env=env, capture_output=True, text=True, timeout=60)
    write_log(
        "launcher_self_test.log",
        f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] code={proc.returncode}\n{proc.stdout}\n{proc.stderr}",
    )
    return proc.returncode


def launch_dashboard(root: Path) -> int:
    python = python_executable(root, console=False)
    env = os.environ.copy()
    env.setdefault("UV_CACHE_DIR", str(root / ".uv-cache"))

    runtime_log = log_dir() / "launcher_runtime.log"
    with runtime_log.open("a", encoding="utf-8") as fp:
        fp.write(f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] launch root={root}\npython={python}\n")
        fp.flush()
        proc = subprocess.Popen(
            [str(python), "-m", "gobang_ai.ui.dashboard"],
            cwd=str(root),
            env=env,
            stdout=fp,
            stderr=subprocess.STDOUT,
        )
        code = proc.wait()
        fp.write(f"dashboard exit code={code}\n")
    return code


def main() -> int:
    parser = argparse.ArgumentParser(description="Launch the GobangAI dashboard from the live source tree.")
    parser.add_argument("--self-test", action="store_true", help="Initialize the dashboard with a dummy video driver.")
    args = parser.parse_args()

    try:
        root = find_project_root()
        if args.self_test:
            return run_self_test(root)
        code = launch_dashboard(root)
        if code != 0:
            write_log("launcher_error.log", f"Dashboard exited with code={code}. See {log_dir() / 'launcher_runtime.log'}")
        return code
    except Exception as exc:
        write_log("launcher_error.log", f"{type(exc).__name__}: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
