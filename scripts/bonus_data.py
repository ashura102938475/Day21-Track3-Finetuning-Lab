#!/usr/bin/env python3
"""Build deterministic, leakage-checked datasets used only by Lab 21 bonuses."""
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import random
import re
import unicodedata

ROOT = pathlib.Path(__file__).resolve().parents[1]
INTENTS = ["doi_tra", "van_chuyen", "hoan_tien", "san_pham_loi", "hoi_thong_tin"]
URGENCIES = ["cao", "trung_binh", "thap"]
SENTIMENTS = ["tieu_cuc", "trung_tinh", "tich_cuc"]
PRODUCTS = [
    "khóa LoRA", "lab RAG", "bài học Transformer", "notebook Colab", "khóa Prompt Engineering",
    "lab Vector Database", "bài tập Python", "khóa MLOps", "lab Fine-tuning", "chứng chỉ AI",
    "buổi mentor", "tài khoản học viên", "GPU credit", "bộ dữ liệu thực hành", "video bài giảng",
]
ISSUES = {
    "doi_tra": ["muốn đổi sang lớp khác", "xin chuyển lịch học", "muốn đổi gói học"],
    "van_chuyen": ["chưa nhận được tài liệu", "email kích hoạt chưa tới", "GPU credit giao chậm"],
    "hoan_tien": ["xin hoàn học phí", "chưa thấy tiền hoàn", "cần hoàn khoản thanh toán"],
    "san_pham_loi": ["notebook không chạy", "video bị lỗi", "tài khoản không đăng nhập được"],
    "hoi_thong_tin": ["hỏi lịch khai giảng", "cần biết học phí", "hỏi điều kiện nhận chứng chỉ"],
}
URGENCY_TEXT = {"cao": "Mình cần xử lý ngay hôm nay", "trung_binh": "Mong phản hồi sớm", "thap": "Khi nào tiện hỗ trợ giúp mình"}
SENTIMENT_TEXT = {"tieu_cuc": "Mình khá thất vọng", "trung_tinh": "Nhờ đội ngũ kiểm tra", "tich_cuc": "Cảm ơn đội ngũ rất nhiều"}
INSTRUCTION = "Phân loại yêu cầu hỗ trợ học viên thành JSON đúng 4 khóa intent, urgency, product, sentiment."


def _norm(text: str) -> str:
    text = unicodedata.normalize("NFD", text.lower())
    text = "".join(c for c in text if unicodedata.category(c) != "Mn")
    return re.sub(r"\W+", " ", text.replace("đ", "d")).strip()


def _row(i: int, rng: random.Random, *, trace: bool = False) -> dict:
    intent = INTENTS[i % len(INTENTS)]
    urgency = URGENCIES[(i // len(INTENTS)) % len(URGENCIES)]
    sentiment = SENTIMENTS[(i // (len(INTENTS) * len(URGENCIES))) % len(SENTIMENTS)]
    product = PRODUCTS[(i * 7 + seed_offset(rng)) % len(PRODUCTS)]
    issue = ISSUES[intent][(i // 3) % len(ISSUES[intent])]
    ticket = f"HV{i + 10001}: Mình đang dùng {product}, {issue}. {URGENCY_TEXT[urgency]}. {SENTIMENT_TEXT[sentiment]}."
    label = {"intent": intent, "urgency": urgency, "product": product, "sentiment": sentiment}
    answer = json.dumps(label, ensure_ascii=False)
    if trace:
        reason = f"Yêu cầu nói về {issue}; dấu hiệu thời gian cho mức {urgency}; sắc thái là {sentiment}."
        answer = f"<think>\n{reason}\n</think>\n\n{answer}"
    return {"instruction": INSTRUCTION, "input": ticket, "output": answer, "label": label}


def seed_offset(rng: random.Random) -> int:
    return rng.randrange(len(PRODUCTS))


def build_custom_rows(seed: int = 42, n: int = 220) -> list[dict]:
    rng = random.Random(seed)
    return [_row(i, rng) for i in range(n)]


def build_trace_rows(seed: int = 43, n: int = 220) -> tuple[list[dict], list[dict]]:
    rng = random.Random(seed)
    rows = [_row(i + 1000, rng, trace=True) for i in range(n)]
    return rows[:-20], rows[-20:]


def validate_rows(rows: list[dict], eval_inputs: list[str]) -> dict:
    allowed = {"intent": set(INTENTS), "urgency": set(URGENCIES), "sentiment": set(SENTIMENTS)}
    schema_errors = 0
    for row in rows:
        label = row.get("label", {})
        if set(label) != {"intent", "urgency", "product", "sentiment"}:
            schema_errors += 1
            continue
        if any(label[k] not in values for k, values in allowed.items()) or not label["product"]:
            schema_errors += 1
    exact = [row.get("input", "") for row in rows]
    normalized = [_norm(x) for x in exact]
    eval_norm = {_norm(x) for x in eval_inputs}
    return {
        "n": len(rows),
        "schema_errors": schema_errors,
        "exact_duplicates": len(exact) - len(set(exact)),
        "normalized_duplicates": len(normalized) - len(set(normalized)),
        "eval_overlaps": sum(x in eval_norm for x in normalized),
    }


def core_hashes(root: pathlib.Path = ROOT) -> dict[str, str]:
    paths = sorted((root / "results").glob("*.json")) + [root / "results/runs.csv"]
    paths += [root / "adapters/correct/adapter_model.safetensors", root / "adapters/correct/adapter_config.json"]
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths if p.is_file()}


def _write_jsonl(path: pathlib.Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")


def write_all(root: pathlib.Path = ROOT) -> dict:
    custom = build_custom_rows()
    trace, holdout = build_trace_rows()
    eval_inputs = []
    for name in ("eval_target.jsonl", "eval_regression.jsonl"):
        for line in (root / "data" / name).read_text(encoding="utf-8").splitlines():
            row = json.loads(line); eval_inputs.append(row.get("input") or row.get("instruction"))
    summary = validate_rows(custom, eval_inputs)
    _write_jsonl(root / "data/train_custom_ai_course.jsonl", custom)
    _write_jsonl(root / "data/train_reasoning_trace.jsonl", trace)
    _write_jsonl(root / "data/trace_holdout.jsonl", holdout)
    (root / "data/CUSTOM_DATASET.md").write_text(
        "# Custom dataset — hỗ trợ học viên AI\n\n"
        "- Nguồn: dữ liệu tổng hợp deterministic, seed 42; không cào web hay chứa dữ liệu cá nhân.\n"
        f"- Quy mô: {summary['n']} yêu cầu hỗ trợ, schema JSON bốn trường của lab.\n"
        "- Miền: vận hành khóa học AI, Colab, GPU credit, bài lab và chứng chỉ; khác phân phối web phổ thông.\n"
        f"- Khử nhiễm: {summary['exact_duplicates']} trùng exact, {summary['normalized_duplicates']} trùng normalized, "
        f"{summary['eval_overlaps']} overlap với hai tập eval đóng băng.\n"
        "- Cách tạo: tổ hợp có kiểm soát giữa sản phẩm, loại yêu cầu, mức khẩn cấp và sắc thái; mọi dòng được validate vocabulary/schema.\n"
        "- Phạm vi: chỉ là bằng chứng bonus B2, không thay thế corpus hoặc baseline core đã đóng băng.\n",
        encoding="utf-8",
    )
    bonus = root / "results/bonus"; bonus.mkdir(parents=True, exist_ok=True)
    (bonus / "dataset_validation.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (bonus / "core_hashes.json").write_text(json.dumps(core_hashes(root), indent=2), encoding="utf-8")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    if args.write:
        print(json.dumps(write_all(), ensure_ascii=False, indent=2))
