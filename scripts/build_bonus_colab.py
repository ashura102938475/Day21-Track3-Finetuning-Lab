#!/usr/bin/env python3
"""Generate the deterministic, student-facing Lab21_BONUS_ALL Colab notebook."""
from __future__ import annotations
import hashlib, json, pathlib
import nbformat

ROOT = pathlib.Path(__file__).resolve().parents[1]
DEST = ROOT / "colab/Lab21_BONUS_ALL.ipynb"

CELLS = [
    ("Overview", "markdown", """# Lab 21 — BONUS ALL (B1–B5)\n\nChọn GPU runtime, thêm `HF_TOKEN` và `GITHUB_TOKEN` trong Colab Secrets, rồi chạy tuần tự. Các stage train có resume."""),
    ("Setup", "code", """# @title Setup\nimport os, pathlib, subprocess, sys, torch\nREPO = "https://github.com/ashura102938475/Day21-Track3-Finetuning-Lab.git"\nWORK = pathlib.Path("/content/Day21-Track3-Finetuning-Lab")\nif not WORK.exists(): subprocess.run(["git", "clone", REPO, str(WORK)], check=True)\nos.chdir(WORK)\nsubprocess.run(["git", "pull", "--ff-only"], check=True)\nsubprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r", "requirements.txt"], check=True)\nif not torch.cuda.is_available(): raise RuntimeError("Chọn Runtime > Change runtime type > GPU")\nos.environ["COMPUTE_TIER"] = "T4"\nos.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"\nprint("GPU ready:", torch.cuda.get_device_name(0))"""),
    ("Secrets", "code", """# @title Secrets\nimport os\nfrom google.colab import userdata\nHF_TOKEN = userdata.get("HF_TOKEN")\nGITHUB_TOKEN = userdata.get("GITHUB_TOKEN")\nif not HF_TOKEN or not GITHUB_TOKEN: raise RuntimeError("Add HF_TOKEN and GITHUB_TOKEN to Colab Secrets")\nos.environ["HF_TOKEN"] = HF_TOKEN\nos.environ["GITHUB_TOKEN"] = GITHUB_TOKEN\nprint("Secrets loaded (values hidden)")"""),
    ("Data", "code", """# @title B2 — custom + reasoning datasets\nimport subprocess, sys\nsubprocess.run([sys.executable, "scripts/bonus_data.py", "--write"], check=True)\nsubprocess.run([sys.executable, "scripts/bonus_verify.py", "--allow-unrun-gpu"], check=True)"""),
    ("B1", "code", """# @title B1 — merge + hot-swap\nimport subprocess, sys\nsubprocess.run([sys.executable, "-u", "notebooks/07_bonus_all.py", "b1"], check=True)"""),
    ("B3", "code", """# @title B3 — reasoning-trace collapse (resumable)\nimport subprocess, sys\nsubprocess.run([sys.executable, "-u", "notebooks/07_bonus_all.py", "b3"], check=True)"""),
    ("B4", "code", """# @title B4 — controlled rank sweep (resumable)\nimport subprocess, sys\nsubprocess.run([sys.executable, "-u", "notebooks/07_bonus_all.py", "b4"], check=True)"""),
    ("Publish", "code", """# @title B5 + verify + publish\nimport os, subprocess, sys\nsubprocess.run([sys.executable, "scripts/publish_bonus.py"], check=True, env=os.environ.copy())\nsubprocess.run([sys.executable, "scripts/verify.py"], check=True)\nsubprocess.run([sys.executable, "scripts/bonus_verify.py"], check=True)\nsubprocess.run(["git", "config", "user.name", "NGUYỄN ANH TRÀ"], check=True)\nsubprocess.run(["git", "config", "user.email", "tra01020407@gmail.com"], check=True)\npaths = ["data/CUSTOM_DATASET.md", "data/train_custom_ai_course.jsonl", "data/train_reasoning_trace.jsonl", "data/trace_holdout.jsonl", "results", "submission/REPORT.md", "LINKS.md"]\nsubprocess.run(["git", "add", "-f", "--", *paths], check=True)\nif subprocess.run(["git", "diff", "--cached", "--quiet"]).returncode:\n    subprocess.run(["git", "commit", "-m", "Complete Lab 21 bonuses B1-B5"], check=True)\n    remote = "https://x-access-token:" + os.environ["GITHUB_TOKEN"] + "@github.com/ashura102938475/Day21-Track3-Finetuning-Lab.git"\n    subprocess.run(["git", "push", remote, "HEAD:main"], check=True)\nprint("All bonus evidence published")"""),
]


def render_notebook() -> str:
    nb = nbformat.v4.new_notebook(metadata={"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "colab": {"name": "Lab21_BONUS_ALL.ipynb", "provenance": []}})
    for i, (title, kind, source) in enumerate(CELLS):
        cell = nbformat.v4.new_markdown_cell(source) if kind == "markdown" else nbformat.v4.new_code_cell(source)
        cell.metadata["title"] = title; cell.id = hashlib.sha1(f"{i}\0{source}".encode()).hexdigest()[:8]
        nb.cells.append(cell)
    return json.dumps(nb, ensure_ascii=True, indent=1)


def main():
    DEST.write_text(render_notebook(), encoding="utf-8"); print("wrote", DEST)


if __name__ == "__main__": main()
