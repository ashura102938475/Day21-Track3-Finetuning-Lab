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
        merge = json.loads((root / "results/merge_check.json").read_text())
        trace = json.loads((root / "results/bonus/reasoning_trace.json").read_text())["runs"]
        sweep = json.loads((root / "results/bonus/rank_sweep.json").read_text())
        text += marker + f"\n**B1.** Merge: {merge['before_merge']:.4f} → {merge['after_merge']:.4f} " \
            f"(Δ {merge['delta']:+.4f}); `correct` và `attn_only` được hot-swap trên cùng base. Merge bỏ overhead adapter nhưng mất khả năng đổi hành vi theo request; giữ adapter riêng phù hợp multi-tenant.\n\n"
        text += "| B3 mask | target | regression | format | valid trace |\n|---|---:|---:|---:|---:|\n" + "\n".join(
            f"| {r['mask_mode']} | {r['target']:.4f} | {r['regression']:.4f} | {r['format']:.4f} | {r['valid_trace_rate']:.4f} |" for r in trace)
        text += "\n\nAccuracy một mình không phát hiện trace collapse; `valid_trace_rate` là tín hiệu trực tiếp cần đọc cùng target.\n\n"
        text += "| B4 rank | target | steps |\n|---:|---:|---:|\n" + "\n".join(
            f"| {r['r']} | {r['target']:.4f} | {r['max_steps']} |" for r in sweep['runs'])
        text += f"\n\nBiên độ rank = {sweep['rank_range']:.4f}, đổi vị trí = {sweep['placement_delta']:.4f}, đổi LR = {sweep['lr_delta']:.4f}. " \
            "So sánh ba biên độ này cho biết rank chỉ là đòn bẩy khi dữ liệu chứa đủ thông tin để dùng thêm capacity.\n\n"
        text += "**B2.** 220 mẫu hỗ trợ học viên AI có zero duplicate/leakage. **B5.** Adapter công khai tại " + HF_URL + ".\n"
    report.write_text(text, encoding="utf-8")


def main() -> int:
    from huggingface_hub import HfApi
    from labkit.bonus import assert_core_unchanged, validate_bonus
    sys.path.insert(0, str(ROOT / "scripts"))
    from bonus_data import core_hashes
    token = os.environ.get("HF_TOKEN")
    if not token: raise SystemExit("HF_TOKEN missing")
    prereq = [c for c in validate_bonus(ROOT, require_publication=False) if c.status == "FAIL"]
    if prereq: raise SystemExit("bonus prerequisites failed: " + ", ".join(c.name for c in prereq))
    frozen_hashes = json.loads((ROOT / "results/bonus/core_hashes.json").read_text(encoding="utf-8"))
    assert_core_unchanged(frozen_hashes, core_hashes(ROOT))
    update_report(ROOT)
    bonus = ROOT / "results/bonus"; bonus.mkdir(parents=True, exist_ok=True)
    publication = {"published": True, "url": HF_URL, "repo_id": REPO_ID, "private": False}
    (ROOT / "LINKS.md").write_text(f"# Submission links\n\n- GitHub: {GH_URL}\n- Hugging Face: {HF_URL}\n", encoding="utf-8")
    verdict = json.loads((ROOT / "results/verdict.json").read_text(encoding="utf-8"))
    api = HfApi(token=token); api.create_repo(REPO_ID, repo_type="model", private=False, exist_ok=True)
    for path in publication_paths(ROOT):
        api.upload_file(path_or_fileobj=str(path), path_in_repo=str(path.relative_to(ROOT)), repo_id=REPO_ID, repo_type="model")
    api.upload_file(path_or_fileobj=render_model_card(verdict).encode(), path_in_repo="README.md", repo_id=REPO_ID, repo_type="model")
    info = api.repo_info(REPO_ID, repo_type="model")
    if info.private: raise RuntimeError("Hugging Face repository is private")
    publication_path = bonus / "publication.json"
    publication_path.write_text(json.dumps(publication, indent=2), encoding="utf-8")
    api.upload_file(path_or_fileobj=str(publication_path), path_in_repo="results/bonus/publication.json", repo_id=REPO_ID, repo_type="model")
    print(HF_URL); return 0


if __name__ == "__main__": raise SystemExit(main())
