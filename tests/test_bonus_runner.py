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
