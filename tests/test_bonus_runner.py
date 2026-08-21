import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
SOURCE = ROOT / "notebooks/07_bonus_all.py"


def test_bonus_runner_exposes_independent_stages():
    source = SOURCE.read_text(encoding="utf-8")
    assert 'STAGES = {"b1"' in source
    for stage in ("b1", "b3", "b4", "report"):
        assert f'"{stage}"' in source
    assert "adapter_complete" in source


def test_rank_sweep_is_controlled():
    source = SOURCE.read_text(encoding="utf-8")
    assert "RANKS = [8, 16, 64]" in source
    assert "MAX_STEPS = 58" in source
    assert 'target="text-linear"' in source
    assert 'BONUS_ADAPTERS = ROOT / "adapters" / "bonus"' in source
    assert "run_meta.json" in source
    assert '"rank_range"' in source and '"placement_delta"' in source and '"lr_delta"' in source


def test_b1_builds_a_resumable_second_adapter_in_bonus_storage():
    source = SOURCE.read_text(encoding="utf-8")
    b1 = source.split("def stage_b1():", 1)[1].split("def mask_hash", 1)[0]
    assert 'train_adapter("rank_8"' in b1
    assert 'adapters/attn_only' not in b1
    assert 'BONUS_ADAPTERS / "rank_8"' in b1


def test_clean_clone_recreates_core_training_split():
    source = SOURCE.read_text(encoding="utf-8")
    assert "def core_train_rows" in source
    assert 'ROOT / "data/train_seed.jsonl"' in source
    assert "data.split(seed_rows, train_frac=0.9, seed=42)" in source
    assert source.count("core_train_rows()") >= 2
