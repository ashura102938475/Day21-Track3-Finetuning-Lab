#!/usr/bin/env python3
"""Run one bonus stage with live logs and repeat its failure tail for Colab."""
from __future__ import annotations

import argparse
from collections import deque
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]


def run(command: list[str]) -> int:
    tail: deque[str] = deque(maxlen=80)
    process = subprocess.Popen(command, cwd=ROOT, stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT, text=True, bufsize=1)
    assert process.stdout is not None
    for line in process.stdout:
        print(line, end="", flush=True)
        tail.append(line)
    code = process.wait()
    if code:
        print("\n===== failure tail (repeat for Colab) =====", flush=True)
        print("".join(tail), end="", flush=True)
        print(f"===== stage exited with code {code} =====", flush=True)
    return code


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", nargs="?", choices=("b1", "b3", "b4"))
    parser.add_argument("--command", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    if args.command:
        return run(args.command)
    if not args.stage:
        parser.error("provide a stage or --command")
    return run([sys.executable, "-u", "notebooks/07_bonus_all.py", args.stage])


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
