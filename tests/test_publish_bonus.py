import importlib.util
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]


def module():
    spec = importlib.util.spec_from_file_location("publish_bonus", ROOT / "scripts/publish_bonus.py")
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod); return mod


def test_publication_manifest_contains_required_paths_only():
    paths = [str(p.relative_to(ROOT)) for p in module().publication_paths(ROOT)]
    assert "adapters/correct/adapter_model.safetensors" in paths
    assert "submission/REPORT.md" in paths
    assert all(".env" not in p and ".venv" not in p for p in paths)


def test_model_card_discloses_failed_verdict():
    card = module().render_model_card({"verdict": {"passed": False}, "comparison": []})
    assert "Qwen/Qwen3.5-2B" in card and "FAILED" in card and "regression" in card.lower()
