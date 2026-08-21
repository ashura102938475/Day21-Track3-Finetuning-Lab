import json
import pathlib
import pytest

from labkit import bonus


def test_partial_adapter_is_not_resumable(tmp_path):
    adapter = tmp_path / "adapter"; adapter.mkdir()
    (adapter / "adapter_config.json").write_text("{}")
    assert bonus.adapter_complete(adapter) is False


def test_core_hash_guard_reports_changed_path():
    with pytest.raises(ValueError, match="results/verdict.json"):
        bonus.assert_core_unchanged({"results/verdict.json": "a"}, {"results/verdict.json": "b"})


def test_rank_and_trace_schema(tmp_path):
    root = tmp_path; (root / "results/bonus").mkdir(parents=True)
    checks = bonus.validate_bonus(root, require_publication=False)
    failed = {c.name for c in checks if c.status == "FAIL"}
    assert "B3 reasoning trace" in failed
    assert "B4 controlled rank sweep" in failed


def test_b1_rejects_duplicate_or_outputless_hotswap(tmp_path):
    (tmp_path / "results/bonus").mkdir(parents=True)
    (tmp_path / "results/merge_check.json").write_text(json.dumps({"delta": 0.0}))
    (tmp_path / "results/bonus/hotswap.json").write_text(json.dumps({
        "adapter_count": 2, "adapters": ["correct", "correct"], "same_base_process": True, "outputs": {}
    }))
    checks = bonus.validate_bonus(tmp_path, require_publication=False)
    assert next(c for c in checks if c.name.startswith("B1")).status == "FAIL"
