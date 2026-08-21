# Colab Bonus-All Design

## Goal

Provide one resumable Google Colab workflow that earns Lab 21 bonuses B1–B5 without modifying or invalidating the already verified NB1–NB5 core artefacts.

## User flow

The student opens `colab/Lab21_BONUS_ALL.ipynb`, selects a GPU runtime, adds `HF_TOKEN` and `GITHUB_TOKEN` to Colab Secrets, and runs the cells in order. Each long-running bonus writes its own completion artefact and skips completed work after a runtime reconnect. The final cell verifies the core submission, updates the bonus appendix, publishes the Hugging Face model repository, and pushes the generated bonus artefacts back to the student's GitHub fork.

## Repository and authentication

The notebook clones `https://github.com/ashura102938475/Day21-Track3-Finetuning-Lab.git` and uses `COMPUTE_TIER=T4` by default. Secrets are read through `google.colab.userdata`; their values are injected only into subprocess environments or authenticated library calls and are never written to `.env`, notebook output, git configuration, or committed files. Git commits use the existing student identity metadata, while pushes target the personal fork's `main` branch.

## Isolation of core evidence

Existing files under `results/`, `adapters/correct/`, and the frozen baseline are read-only inputs for the bonus workflow. Bonus runs write to `results/bonus/` and `adapters/bonus/`. The workflow records the SHA-256 hashes of core result files before and after all bonus operations and fails if any core evidence changes. `scripts/verify.py` remains the final core gate.

## Bonus B1: merge and hot-swap

Run the existing NB6 against all 50 target items. Preserve `results/merge_check.json` and add `results/bonus/hotswap.json` containing the names of at least two adapters loaded into one base model plus their sample outputs. The bonus succeeds only when merge score delta is at least -0.01 and at least two adapters were actually selected in the same process.

## Bonus B2: custom domain dataset

Create a deterministic dataset of at least 220 Vietnamese support requests for an applied-AI course. The schema remains the lab's four-field JSON contract so existing masking and scoring utilities apply. Generation uses independently defined products, intents, urgency cues, sentiment cues, and paraphrase patterns, followed by validation for schema membership, exact duplicates, normalized-input duplicates, label balance, and overlap against both frozen eval files. `data/CUSTOM_DATASET.md` documents provenance as synthetic, the deterministic construction process, validation counts, deduplication, leakage checks, and how the distribution differs from generic web pretraining data. The custom dataset is bonus evidence and does not replace the core training corpus.

## Bonus B3: reasoning-trace collapse

Build a separate trace-bearing training set whose assistant messages contain a non-empty `<think>...</think>` block followed by the JSON answer. Train two otherwise identical text-linear r=16 adapters for the same optimizer-step budget: one with `MASK_MODE=assistant-only`, one with `MASK_MODE=response-only`. Evaluate both on a fixed trace-capable bonus holdout and the frozen regression set. Save `results/bonus/reasoning_trace.json` with target, regression, format, and `valid_trace_rate` for both modes, plus configuration and dataset hashes. If the two mask modes do not produce distinct supervised-token masks, fail before training rather than claim the bonus.

## Bonus B4: controlled rank sweep

Use text-linear placement, LR 1e-4, the same train split, mask, effective batch, precision, and 58-step budget as `correct`. Reuse the existing r=16 measurements and train only r=8 and r=64. Evaluate both new adapters on all 50 frozen target items and save `results/bonus/rank_sweep.json`. The report compares the target-score range from rank against the measured placement delta (`correct` versus `attn_only`) and LR delta (`correct` versus `wrong_lr`).

## Bonus B5: Hugging Face publication

Create or reuse the public model repository `ashura102938475/lab21-qwen35-triage-vi`. Upload the two required correct-adapter files, all core and bonus result JSON/CSV files, `submission/REPORT.md`, and a generated model card containing the base model, mask mode, evaluation verdict, limitations, and GitHub link. Record the public URL in the report and `LINKS.md`.

## Resume and failure handling

Every training unit treats an adapter as complete only when both `adapter_model.safetensors` and `adapter_config.json` exist and its result record passes schema validation. Partial directories are retrained. Network publication is idempotent. A failed bonus stops later dependent cells but never deletes existing artefacts. The final publish cell refuses to push when verification fails, core hashes drift, required bonus evidence is missing, or authentication secrets are unavailable.

## Source structure

- `notebooks/07_bonus_all.py`: source-of-truth bonus implementation, organized into independently callable stages.
- `scripts/bonus_data.py`: deterministic custom and trace dataset construction plus validation.
- `scripts/bonus_verify.py`: checks B1–B5 evidence and core-hash preservation.
- `scripts/build_bonus_colab.py`: generates the dedicated Colab notebook with setup, secret checks, staged execution, verification, and publication cells.
- `colab/Lab21_BONUS_ALL.ipynb`: generated notebook committed for direct use.
- `results/bonus/`: bonus-only measurements.
- `adapters/bonus/`: bonus-only adapters, ignored locally unless explicitly published.

## Testing and acceptance

CPU tests cover deterministic dataset generation, schema validity, deduplication, leakage detection, result schemas, core-hash guarding, notebook cell order, secret non-persistence, and resume predicates. A local notebook build must be deterministic. Colab acceptance requires B1 merge delta ≥ -0.01, B1 hot-swap count ≥2, B2 ≥200 validated samples with zero eval overlap, B3 two distinct masks and two evaluated adapters, B4 ranks 8/16/64 at one step budget, a public B5 URL, core `scripts/verify.py` exit code 0, and `scripts/bonus_verify.py` exit code 0.
