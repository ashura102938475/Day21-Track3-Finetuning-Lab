"""Pure validation helpers for optional Lab 21 bonus evidence."""
from __future__ import annotations

import json
import pathlib
from dataclasses import dataclass


@dataclass(frozen=True)
class Check:
    name: str
    status: str
    detail: str = ""


def adapter_complete(path: pathlib.Path) -> bool:
    return all((path / name).is_file() and (path / name).stat().st_size > 0 for name in (
        "adapter_model.safetensors", "adapter_config.json"))


def assert_core_unchanged(before: dict[str, str], after: dict[str, str]) -> None:
    changed = sorted(k for k in set(before) | set(after) if before.get(k) != after.get(k))
    if changed:
        raise ValueError("core evidence changed: " + ", ".join(changed))


def _load(path: pathlib.Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def validate_bonus(root: pathlib.Path, require_publication: bool = True) -> list[Check]:
    out: list[Check] = []
    merge = _load(root / "results/merge_check.json") or {}
    hot = _load(root / "results/bonus/hotswap.json") or {}
    names = hot.get("adapters", [])
    b1 = merge.get("delta", -99) >= -0.01 and hot.get("adapter_count", 0) >= 2
    b1 = b1 and len(names) >= 2 and len(set(names)) == len(names)
    b1 = b1 and hot.get("same_base_process") is True and set(hot.get("outputs", {})) == set(names)
    out.append(Check("B1 merge and hot-swap", "PASS" if b1 else "FAIL"))
    ds = _load(root / "results/bonus/dataset_validation.json") or {}
    b2 = ds.get("n", 0) >= 200 and all(ds.get(k, 1) == 0 for k in (
        "schema_errors", "exact_duplicates", "normalized_duplicates", "eval_overlaps"))
    b2 = b2 and ds.get("min_label_count", 0) >= 10
    out.append(Check("B2 custom dataset", "PASS" if b2 else "FAIL"))
    trace = _load(root / "results/bonus/reasoning_trace.json") or {}
    modes = trace.get("runs", [])
    b3 = len(modes) == 2 and {r.get("mask_mode") for r in modes} == {"assistant-only", "response-only"}
    b3 = b3 and len({r.get("mask_hash") for r in modes}) == 2 and len({r.get("max_steps") for r in modes}) == 1
    b3 = b3 and {r.get("r") for r in modes} == {16}
    b3 = b3 and all(all(k in r for k in ("target", "regression", "format", "valid_trace_rate", "dataset_hash")) for r in modes)
    out.append(Check("B3 reasoning trace", "PASS" if b3 else "FAIL"))
    sweep = _load(root / "results/bonus/rank_sweep.json") or {}
    runs = sweep.get("runs", [])
    b4 = {r.get("r") for r in runs} == {8, 16, 64}
    b4 = b4 and len({r.get("max_steps") for r in runs}) == 1 and len({r.get("learning_rate") for r in runs}) == 1
    b4 = b4 and {r.get("placement") for r in runs} == {"text-linear"} and all("target" in r for r in runs)
    b4 = b4 and all(k in sweep for k in ("rank_range", "placement_delta", "lr_delta"))
    out.append(Check("B4 controlled rank sweep", "PASS" if b4 else "FAIL"))
    pub = _load(root / "results/bonus/publication.json") or {}
    b5 = pub.get("published") is True and str(pub.get("url", "")).startswith("https://huggingface.co/")
    status = "PASS" if b5 else ("FAIL" if require_publication else "PENDING")
    out.append(Check("B5 Hugging Face publication", status))
    return out
