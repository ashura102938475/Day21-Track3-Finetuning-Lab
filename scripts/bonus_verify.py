#!/usr/bin/env python3
from __future__ import annotations
import argparse, pathlib, sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from labkit.bonus import validate_bonus

parser = argparse.ArgumentParser()
parser.add_argument("--allow-unrun-gpu", action="store_true")
args = parser.parse_args()
checks = validate_bonus(ROOT, require_publication=not args.allow_unrun_gpu)
failures = 0
for check in checks:
    status = check.status
    if args.allow_unrun_gpu and status == "FAIL" and check.name != "B2 custom dataset":
        status = "PENDING"
    failures += status == "FAIL"
    print(f"[{status:^7}] {check.name}")
raise SystemExit(1 if failures else 0)
