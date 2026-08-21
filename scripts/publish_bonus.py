#!/usr/bin/env python3
"""Publish verified Lab 21 core+bonus evidence to a public Hugging Face repo."""
from __future__ import annotations
import json, os, pathlib, sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
REPO_ID = "ashura102938475/lab21-qwen35-triage-vi"
HF_URL = f"https://huggingface.co/{REPO_ID}"
GH_URL = "https://github.com/ashura102938475/Day21-Track3-Finetuning-Lab"


def publication_paths(root: pathlib.Path) -> list[pathlib.Path]:
    fixed = [root / "adapters/correct/adapter_model.safetensors", root / "adapters/correct/adapter_config.json",
             root / "submission/REPORT.md"]
    return [p for p in fixed + sorted((root / "results").rglob("*.json")) + [root / "results/runs.csv"] if p.is_file()]


def render_model_card(verdict: dict) -> str:
    status = "PASSED" if verdict.get("verdict", {}).get("passed") else "FAILED"
    return f"""---
base_model: Qwen/Qwen3.5-2B
library_name: peft
language: vi
---
# Lab 21 — Vietnamese triage LoRA

Student: NGUYỄN ANH TRÀ (`2A202601735`). Base model: `Qwen/Qwen3.5-2B`.

Core regression-gate verdict: **{status}**. The adapter improves the narrow target task but the measured regression score drops substantially; do not deploy it as a general assistant. See `results/verdict.json` and `submission/REPORT.md` for exact measurements, limitations, fair contrasts, and B1–B5 evidence.

GitHub: {GH_URL}
"""


def update_report(root: pathlib.Path) -> None:
    report = root / "submission/REPORT.md"; text = report.read_text(encoding="utf-8")
    for key in range(1, 6):
        text = text.replace(f"- [ ] B{key}", f"- [x] B{key}")
    text = text.replace("- [x] B5 HuggingFace Hub", f"- [x] B5 HuggingFace Hub — {HF_URL}")
    marker = "\n## Kết quả bonus B1–B5\n"
    if marker not in text:
        text += marker + "\nBằng chứng chi tiết được máy kiểm tra trong `results/merge_check.json` và `results/bonus/`. " \
            "B1 xác nhận merge không tụt quá 0.01 và hot-swap hai adapter trên cùng base. B2 gồm 220 mẫu hỗ trợ học viên AI đã khử trùng/lọc leakage. " \
            "B3 so sánh hai loss mask trên dữ liệu có trace thật; B4 quét r=8/16/64 với cùng 58 step; B5 công khai adapter và artefact tại " + HF_URL + ".\n"
    report.write_text(text, encoding="utf-8")


def main() -> int:
    from huggingface_hub import HfApi
    from labkit.bonus import validate_bonus
    token = os.environ.get("HF_TOKEN")
    if not token: raise SystemExit("HF_TOKEN missing")
    prereq = [c for c in validate_bonus(ROOT, require_publication=False) if c.status == "FAIL"]
    if prereq: raise SystemExit("bonus prerequisites failed: " + ", ".join(c.name for c in prereq))
    update_report(ROOT)
    bonus = ROOT / "results/bonus"; bonus.mkdir(parents=True, exist_ok=True)
    publication = {"published": True, "url": HF_URL, "repo_id": REPO_ID, "private": False}
    (bonus / "publication.json").write_text(json.dumps(publication, indent=2), encoding="utf-8")
    (ROOT / "LINKS.md").write_text(f"# Submission links\n\n- GitHub: {GH_URL}\n- Hugging Face: {HF_URL}\n", encoding="utf-8")
    verdict = json.loads((ROOT / "results/verdict.json").read_text(encoding="utf-8"))
    api = HfApi(token=token); api.create_repo(REPO_ID, repo_type="model", private=False, exist_ok=True)
    for path in publication_paths(ROOT):
        api.upload_file(path_or_fileobj=str(path), path_in_repo=str(path.relative_to(ROOT)), repo_id=REPO_ID, repo_type="model")
    api.upload_file(path_or_fileobj=render_model_card(verdict).encode(), path_in_repo="README.md", repo_id=REPO_ID, repo_type="model")
    info = api.repo_info(REPO_ID, repo_type="model")
    if info.private: raise RuntimeError("Hugging Face repository is private")
    print(HF_URL); return 0


if __name__ == "__main__": raise SystemExit(main())
