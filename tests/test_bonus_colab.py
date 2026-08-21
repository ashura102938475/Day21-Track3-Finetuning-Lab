import importlib.util
import json
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
NB = ROOT / "colab/Lab21_BONUS_ALL.ipynb"


def test_bonus_colab_cell_order_and_gpu_guard():
    nb = json.loads(NB.read_text(encoding="utf-8"))
    titles = [c.get("metadata", {}).get("title") for c in nb["cells"]]
    assert titles == ["Overview", "Setup", "Secrets", "Data", "B1", "B3", "B4", "Publish"]
    assert "torch.cuda.is_available()" in "".join(nb["cells"][1]["source"])


def test_tokens_are_read_but_never_persisted_or_printed():
    raw = json.loads(NB.read_text(encoding="utf-8"))
    text = "\n".join("".join(c.get("source", [])) for c in raw["cells"])
    assert 'userdata.get(\"HF_TOKEN\")' in text
    assert 'userdata.get(\"GITHUB_TOKEN\")' in text
    assert "print(HF_TOKEN" not in text and "print(GITHUB_TOKEN" not in text
    assert "x-access-token:" not in text
    assert text.index("scripts/verify.py") < text.index("scripts/publish_bonus.py")
    assert 'os.environ["COMPUTE_TIER"] = "LAPTOP"' in text
    assert '"remote", "add", "mine"' in text
    assert '"remote", "set-url", "mine"' in text


def test_build_is_deterministic():
    spec = importlib.util.spec_from_file_location("builder", ROOT / "scripts/build_bonus_colab.py")
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    assert module.render_notebook() == module.render_notebook()
