#!/usr/bin/env python3
"""Generate the deterministic, student-facing Lab21_BONUS_ALL Colab notebook."""
from __future__ import annotations
import hashlib, json, pathlib
import nbformat

ROOT = pathlib.Path(__file__).resolve().parents[1]
DEST = ROOT / "colab/Lab21_BONUS_ALL.ipynb"

CELLS = [
    ("Overview", "markdown", """# Lab 21 — BONUS ALL (B1–B5)\n\nChọn GPU runtime, thêm `HF_TOKEN` và `GITHUB_TOKEN` trong Colab Secrets, rồi chạy tuần tự. Các stage train có resume."""),
    ("Setup", "code", """# @title Setup\nimport os, pathlib, runpy, subprocess, sys, torch\nREPO = "https://github.com/ashura102938475/Day21-Track3-Finetuning-Lab.git"\nWORK = pathlib.Path("/content/Day21-Track3-Finetuning-Lab")\nif not WORK.exists(): subprocess.run(["git", "clone", REPO, str(WORK)], check=True)\nos.chdir(WORK)\nsubprocess.run(["git", "pull", "--ff-only"], check=True)\nsubprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r", "requirements.txt"], check=True)\nif not torch.cuda.is_available(): raise RuntimeError("Chọn Runtime > Change runtime type > GPU")\n# Match the already-frozen core model; Colab GPU is hardware, not a reason to change the experiment.\nos.environ["COMPUTE_TIER"] = "LAPTOP"\nos.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"\n_bonus = runpy.run_path("notebooks/07_bonus_all.py", run_name="lab21_bonus")\n_stages = {"b1": _bonus["stage_b1"], "b3": _bonus["stage_b3"], "b4": _bonus["stage_b4"]}\ndef run_stage(name):\n    print(f"Running {name} in the Colab kernel; exceptions below are the real root cause.")\n    return _stages[name]()\nprint("GPU ready:", torch.cuda.get_device_name(0))"""),
    ("Secrets", "code", """# @title Secrets\nimport os\nfrom google.colab import userdata\nHF_TOKEN = userdata.get("HF_TOKEN")\nGITHUB_TOKEN = userdata.get("GITHUB_TOKEN")\nif not HF_TOKEN or not GITHUB_TOKEN: raise RuntimeError("Add HF_TOKEN and GITHUB_TOKEN to Colab Secrets")\nos.environ["HF_TOKEN"] = HF_TOKEN\nos.environ["GITHUB_TOKEN"] = GITHUB_TOKEN\nprint("Secrets loaded (values hidden)")"""),
    ("Data", "code", """# @title B2 — custom + reasoning datasets\nfrom scripts import bonus_data, bonus_verify\nsummary = bonus_data.write_all()\nprint("dataset rows:", summary["n"], "schema errors:", summary["schema_errors"])\nif bonus_verify.main(["--allow-unrun-gpu"]):\n    raise RuntimeError("Bonus data verification failed; read the FAIL line above.")"""),
    ("B1", "code", """# @title B1 — merge + hot-swap\nrun_stage("b1")"""),
    ("B3", "code", """# @title B3 — reasoning-trace collapse (resumable)\nrun_stage("b3")"""),
    ("B4", "code", """# @title B4 — controlled rank sweep (resumable)\nrun_stage("b4")"""),
    ("Publish", "code", """# @title Verify first, then B5 publish\nimport os, subprocess\nfrom scripts import bonus_verify, publish_bonus, verify as core_verify\n\ndef require_ok(label, code):\n    if code:\n        raise RuntimeError(f"{label} failed; read the FAIL line above.")\n\ndef git(*args, check=True, env=None):\n    result = subprocess.run(["git", *args], text=True, capture_output=True, env=env)\n    output = (result.stdout or "") + (result.stderr or "")\n    if output.strip(): print(output.rstrip())\n    if check and result.returncode:\n        raise RuntimeError(f"git {' '.join(args)} failed with exit {result.returncode}\\n{output}")\n    return result\n\nrequire_ok("Core verification", core_verify.main([]))\nrequire_ok("Bonus prerequisites", bonus_verify.main(["--allow-unrun-gpu"]))\nrequire_ok("Hugging Face publication", publish_bonus.main())\nrequire_ok("Final bonus verification", bonus_verify.main([]))\npaths = ["data/CUSTOM_DATASET.md", "data/train_custom_ai_course.jsonl", "data/train_reasoning_trace.jsonl", "data/trace_holdout.jsonl", "results/merge_check.json", "results/bonus", "submission/REPORT.md", "LINKS.md"]\ngit("add", "-f", "--", *paths)\ncommit_env = {**os.environ, "GIT_AUTHOR_NAME": "NGUYỄN ANH TRÀ", "GIT_AUTHOR_EMAIL": "tra01020407@gmail.com", "GIT_COMMITTER_NAME": "NGUYỄN ANH TRÀ", "GIT_COMMITTER_EMAIL": "tra01020407@gmail.com"}\nif git("diff", "--cached", "--quiet", check=False).returncode:\n    git("commit", "-m", "Complete Lab 21 bonuses B1-B5", env=commit_env)\n    if git("remote", "get-url", "mine", check=False).returncode:\n        git("remote", "add", "mine", REPO)\n    else:\n        git("remote", "set-url", "mine", REPO)\n    helper = '!f() { echo username=x-access-token; echo password=$GITHUB_TOKEN; }; f'\n    git("-c", "credential.helper=" + helper, "push", "mine", "HEAD:main", env=os.environ.copy())\nprint("All bonus evidence published")"""),
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
