import importlib.util
import json
import pathlib


ROOT = pathlib.Path(__file__).resolve().parents[1]


def load_module():
    spec = importlib.util.spec_from_file_location("bonus_data", ROOT / "scripts/bonus_data.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def frozen_eval_inputs():
    rows = []
    for name in ("eval_target.jsonl", "eval_regression.jsonl"):
        for line in (ROOT / "data" / name).read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            rows.append(row.get("input") or row.get("instruction"))
    return rows


def test_custom_dataset_is_deterministic_valid_and_non_leaking():
    bonus_data = load_module()
    rows_a = bonus_data.build_custom_rows(seed=42, n=220)
    rows_b = bonus_data.build_custom_rows(seed=42, n=220)
    assert rows_a == rows_b and len(rows_a) == 220
    summary = bonus_data.validate_rows(rows_a, frozen_eval_inputs())
    assert summary["schema_errors"] == 0
    assert summary["exact_duplicates"] == 0
    assert summary["normalized_duplicates"] == 0
    assert summary["eval_overlaps"] == 0
    assert summary["min_label_count"] >= 10
    assert len({r["input"].split(":", 1)[1].strip().split()[0] for r in rows_a}) >= 3


def test_trace_dataset_contains_real_traces_and_fixed_holdout():
    bonus_data = load_module()
    train, holdout = bonus_data.build_trace_rows(seed=43, n=220)
    assert len(train) == 200 and len(holdout) == 20
    assert all("<think>" in r["output"] and "</think>" in r["output"] for r in train)
    assert all(
        len(r["output"].split("<think>", 1)[1].split("</think>", 1)[0].strip()) >= 10
        for r in train
    )


def test_core_hashes_cover_only_frozen_evidence():
    bonus_data = load_module()
    hashes = bonus_data.core_hashes(ROOT)
    assert "results/verdict.json" in hashes
    assert "adapters/correct/adapter_model.safetensors" in hashes
    assert not any("results/bonus" in key for key in hashes)
