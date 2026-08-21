#!/usr/bin/env python3
"""Resumable GPU stages for Lab 21 bonuses B1, B3, B4 and report evidence."""
from __future__ import annotations

import argparse, csv, hashlib, json, os, pathlib, sys, time
sys.path.insert(0, str(pathlib.Path.cwd() / "src"))

from labkit import data, evaluate as ev, generate, modeling, train
from labkit.bonus import adapter_complete
from labkit.config import LoraSpec, NAIVE_PROMPT, get_tier

ROOT = pathlib.Path.cwd()
TIER = get_tier(os.environ.get("COMPUTE_TIER", "T4"))
BONUS_RESULTS = ROOT / "results" / "bonus"
BONUS_ADAPTERS = ROOT / "adapters" / "bonus"
RANKS = [8, 16, 64]
MAX_STEPS = 58
BONUS_RESULTS.mkdir(parents=True, exist_ok=True); BONUS_ADAPTERS.mkdir(parents=True, exist_ok=True)


def rows(path):
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def core_train_rows():
    """Return NB1's deterministic train split, including on a clean Git clone."""
    split_path = ROOT / "data/split/train.jsonl"
    if split_path.is_file():
        return rows(split_path)
    seed_rows = rows(ROOT / "data/train_seed.jsonl")
    train_rows, _ = data.split(seed_rows, train_frac=0.9, seed=42)
    return train_rows


def atomic_json(path, value):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def score(path, target_rows, regression_rows=None):
    from peft import PeftModel
    model, tok = generate.load_base(TIER); model = PeftModel.from_pretrained(model, str(path)); model.eval()
    preds, latency = generate.generate_batch(model, tok, [r["input"] for r in target_rows], system=NAIVE_PROMPT, max_new_tokens=160)
    target = sum(ev.triage_field_accuracy(p, r["label"]) for p, r in zip(preds, target_rows)) / len(target_rows)
    fmt = sum(ev.has_required_keys(p, ev.TRIAGE_KEYS) for p in preds) / len(preds)
    trace = sum(ev.valid_reasoning_trace(p) for p in preds) / len(preds)
    regression = 0.0
    if regression_rows:
        rp, _ = generate.generate_batch(model, tok, [r["instruction"] for r in regression_rows], system=None, max_new_tokens=96)
        regression = sum(ev.keyword_recall(p, r["keywords"]) for p, r in zip(rp, regression_rows)) / len(rp)
    del model; generate.free_memory()
    return {"target": round(target, 4), "regression": round(regression, 4), "format": round(fmt, 4),
            "valid_trace_rate": round(trace, 4), "latency_ms": round(latency, 1)}


def train_adapter(key, train_rows, mask_mode, rank):
    out = BONUS_ADAPTERS / key
    dataset_hash = hashlib.sha256(json.dumps(train_rows, ensure_ascii=False, sort_keys=True).encode()).hexdigest()[:16]
    expected = {"key": key, "mask_mode": mask_mode, "rank": rank, "max_steps": MAX_STEPS,
                "model": TIER.model_id, "dataset_hash": dataset_hash}
    meta_path = out / "run_meta.json"
    try: current = json.loads(meta_path.read_text())
    except Exception: current = None
    if adapter_complete(out) and current == expected:
        print("resume:", key); return out
    from datasets import Dataset
    from peft import LoraConfig
    from trl import SFTConfig, SFTTrainer
    model, tok = generate.load_base(TIER)
    targets = modeling.resolve_target_modules(model, "text-linear")
    dataset = Dataset.from_list(data.to_training_dataset(tok, train_rows, max_length=TIER.max_length,
                                                         mask_mode=mask_mode))
    spec = LoraSpec(key=key, r=rank, alpha=2 * rank, target="text-linear", lr=1e-4,
                    load_in_4bit=False, label=key, teaches="controlled bonus")
    want = train.sft_config_kwargs(TIER, spec, str(out), max_steps=MAX_STEPS)
    skw, _ = train.filter_kwargs(SFTConfig, want, label=key)
    lkw, _ = train.filter_kwargs(LoraConfig, train.lora_config_kwargs(spec, targets), label=key)
    trainer = SFTTrainer(model=model, args=SFTConfig(**skw), train_dataset=dataset,
                         processing_class=tok, peft_config=LoraConfig(**lkw))
    trainer.train(); trainer.model.save_pretrained(out)
    atomic_json(meta_path, expected)
    del trainer, model; generate.free_memory(); return out


def stage_b1():
    from peft import PeftModel
    target = rows(ROOT / "data/eval_target.jsonl")
    # The submission intentionally ships only adapters/correct. Build the second
    # adapter in bonus storage instead of assuming the optional core contrasts were
    # uploaded. B4 reuses this exact fingerprint, so the work is never duplicated.
    train_rows = core_train_rows()
    second = train_adapter("rank_8", train_rows, "assistant-only", 8)
    model, tok = generate.load_base(TIER)
    model = PeftModel.from_pretrained(model, str(ROOT / "adapters/correct"), adapter_name="correct")
    before_preds, _ = generate.generate_batch(model, tok, [r["input"] for r in target], system=NAIVE_PROMPT)
    before = sum(ev.triage_field_accuracy(p, r["label"]) for p, r in zip(before_preds, target)) / len(target)
    merged = model.merge_and_unload()
    after_preds, _ = generate.generate_batch(merged, tok, [r["input"] for r in target], system=NAIVE_PROMPT)
    after = sum(ev.triage_field_accuracy(p, r["label"]) for p, r in zip(after_preds, target)) / len(target)
    assert after - before >= -0.01
    atomic_json(ROOT / "results/merge_check.json", {"before_merge": before, "after_merge": after,
                "delta": after-before, "tolerance": 0.01, "n": len(target)})
    del merged; generate.free_memory()
    model, tok = generate.load_base(TIER)
    model = PeftModel.from_pretrained(model, str(ROOT / "adapters/correct"), adapter_name="correct")
    model.load_adapter(str(BONUS_ADAPTERS / "rank_8"), adapter_name="rank_8")
    sample = target[0]["input"]; outputs = {}
    for name in ("correct", "rank_8"):
        model.set_adapter(name); pred, _ = generate.generate_batch(model, tok, [sample], system=NAIVE_PROMPT)
        outputs[name] = pred[0]
    atomic_json(BONUS_RESULTS / "hotswap.json", {"adapters": list(outputs), "adapter_count": 2,
                "same_base_process": True, "ticket": sample, "outputs": outputs})


def mask_hash(row, mode):
    _, tok = generate.load_base(TIER)
    example = data.build_example(tok, data.to_messages(row), max_length=TIER.max_length, mask_mode=mode)
    supervised = [i for i, label in enumerate(example.labels) if label != -100]
    generate.free_memory()
    return hashlib.sha256(json.dumps(supervised).encode()).hexdigest()[:16]


def stage_b3():
    train_rows = rows(ROOT / "data/train_reasoning_trace.jsonl")
    holdout = rows(ROOT / "data/trace_holdout.jsonl")
    regression = rows(ROOT / "data/eval_regression.jsonl")
    hashes = {mode: mask_hash(train_rows[0], mode) for mode in ("assistant-only", "response-only")}
    assert len(set(hashes.values())) == 2, "mask modes are identical; refusing false B3 evidence"
    runs = []
    for mode in ("assistant-only", "response-only"):
        key = "trace_" + mode.replace("-", "_")
        metrics = score(train_adapter(key, train_rows, mode, 16), holdout, regression)
        dataset_hash = hashlib.sha256(json.dumps(train_rows, ensure_ascii=False, sort_keys=True).encode()).hexdigest()[:16]
        runs.append({"run": key, "mask_mode": mode, "mask_hash": hashes[mode], "dataset_hash": dataset_hash,
                     "model": TIER.model_id, "r": 16,
                     "max_steps": MAX_STEPS, **metrics})
    atomic_json(BONUS_RESULTS / "reasoning_trace.json", {"runs": runs, "n_train": len(train_rows), "n_holdout": len(holdout)})


def stage_b4():
    train_rows = core_train_rows(); target = rows(ROOT / "data/eval_target.jsonl")
    autopsy = {r["run"]: r for r in json.loads((ROOT / "results/autopsy.json").read_text())}
    run_csv = {r["run"]: r for r in csv.DictReader((ROOT / "results/runs.csv").open())}
    result = [{"r": 16, "placement": "text-linear", "learning_rate": 1e-4, "max_steps": MAX_STEPS,
               "target": autopsy["correct"]["target"], "final_loss": float(run_csv["correct"]["final_loss"])}]
    for rank in (8, 64):
        path = train_adapter(f"rank_{rank}", train_rows, "assistant-only", rank)
        result.append({"r": rank, "placement": "text-linear", "learning_rate": 1e-4,
                       "model": TIER.model_id,
                       "dataset_hash": hashlib.sha256(json.dumps(train_rows, ensure_ascii=False, sort_keys=True).encode()).hexdigest()[:16],
                       "max_steps": MAX_STEPS, **score(path, target)})
    result.sort(key=lambda x: x["r"])
    targets = [r["target"] for r in result]
    atomic_json(BONUS_RESULTS / "rank_sweep.json", {"runs": result,
                "rank_range": round(max(targets) - min(targets), 4),
                "placement_delta": round(abs(autopsy["correct"]["target"] - autopsy["attn_only"]["target"]), 4),
                "lr_delta": round(abs(autopsy["correct"]["target"] - autopsy["wrong_lr"]["target"]), 4)})


def stage_report():
    print("GPU evidence ready. Publish stage renders the measured appendix after bonus_verify passes.")


STAGES = {"b1": stage_b1, "b3": stage_b3, "b4": stage_b4, "report": stage_report}
if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("stage", choices=STAGES)
    STAGES[parser.parse_args().stage]()
