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
    assert text.index("core_verify.main([])") < text.index("publish_bonus.main()")
    assert 'os.environ["COMPUTE_TIER"] = "LAPTOP"' in text
    assert '"remote", "add", "mine"' in text
    assert '"remote", "set-url", "mine"' in text


def test_build_is_deterministic():
    spec = importlib.util.spec_from_file_location("builder", ROOT / "scripts/build_bonus_colab.py")
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    assert module.render_notebook() == module.render_notebook()


def test_gpu_stage_cells_use_diagnostic_runner():
    nb = json.loads(NB.read_text(encoding="utf-8"))
    by_title = {c["metadata"]["title"]: "".join(c.get("source", [])) for c in nb["cells"]}
    for title, stage in (("B1", "b1"), ("B3", "b3"), ("B4", "b4")):
        assert f'run_stage("{stage}")' in by_title[title]
        assert "subprocess" not in by_title[title]


def test_notebook_defines_in_process_stage_runner():
    nb = json.loads(NB.read_text(encoding="utf-8"))
    setup = "".join(nb["cells"][1]["source"])
    assert "runpy.run_path" in setup
    assert "stage_b1" in setup


def test_data_and_publish_checks_run_in_process():
    nb = json.loads(NB.read_text(encoding="utf-8"))
    by_title = {c["metadata"]["title"]: "".join(c.get("source", [])) for c in nb["cells"]}
    data_cell = by_title["Data"]
    publish = by_title["Publish"]
    assert "bonus_data.write_all()" in data_cell
    assert "bonus_verify.main([\"--allow-unrun-gpu\"])" in data_cell
    assert "core_verify.main([])" in publish
    assert "publish_bonus.main()" in publish
    assert "bonus_verify.main([])" in publish
    assert 'subprocess.run([sys.executable, "scripts/verify.py"]' not in publish
