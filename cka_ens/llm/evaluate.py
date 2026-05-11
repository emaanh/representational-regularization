"""Per-member benchmark scorer; emits Model,Question,Benchmark,Reward CSV rows."""
from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path
from typing import Iterable

try:
    import torch
    import datasets
except ImportError as e:
    raise ImportError("Install with `uv pip install -e '.[llm]'`.") from e

from .model_loader import list_existing_members, load_member, DEFAULT_BACKBONE


BENCHMARK_CONFIG = {
    "mmlu":          {"hf_id": "cais/mmlu",         "split": "test",  "subset": "all"},
    "gsm8k":         {"hf_id": "gsm8k",             "split": "test",  "subset": "main"},
    "humaneval":     {"hf_id": "openai_humaneval",  "split": "test",  "subset": None},
    "arc_challenge": {"hf_id": "ai2_arc",           "split": "test",  "subset": "ARC-Challenge"},
    "arc_easy":      {"hf_id": "ai2_arc",           "split": "test",  "subset": "ARC-Easy"},
    "gpqa":          {"hf_id": "Idavidrein/gpqa",   "split": "train", "subset": "gpqa_main"},
}


def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--benchmark", choices=list(BENCHMARK_CONFIG), required=True)
    p.add_argument("--out", type=Path, default=Path("all_results.csv"))
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--backbone", type=str, default=DEFAULT_BACKBONE)
    return p.parse_args(argv)


def load_benchmark(name: str, limit=None):
    """Load a benchmark split via HF datasets."""
    cfg = BENCHMARK_CONFIG[name]
    ds = (datasets.load_dataset(cfg["hf_id"], cfg["subset"], split=cfg["split"])
          if cfg["subset"] else
          datasets.load_dataset(cfg["hf_id"], split=cfg["split"]))
    return ds.select(range(min(limit, len(ds)))) if limit else ds


def format_prompt(benchmark: str, example: dict) -> tuple[str, str]:
    """Return (prompt, gold_answer) for a given example."""
    if benchmark == "mmlu":
        opts = "\n".join(f"{chr(65+i)}. {c}" for i, c in enumerate(example["choices"]))
        return (f"The following is a multiple choice question.\n\n"
                f"Question: {example['question']}\n{opts}\nAnswer:",
                chr(65 + example["answer"]))
    if benchmark == "gsm8k":
        m = re.search(r"####\s*(-?\d[\d,]*\.?\d*)", example["answer"])
        return f"Question: {example['question']}\nAnswer:", (m.group(1).replace(",", "") if m else "")
    if benchmark == "humaneval":
        return example["prompt"], example["canonical_solution"]
    if benchmark in ("arc_challenge", "arc_easy"):
        opts = "\n".join(f"{lbl}. {tx}" for lbl, tx in
                          zip(example["choices"]["label"], example["choices"]["text"]))
        return f"Question: {example['question']}\n{opts}\nAnswer:", example["answerKey"]
    if benchmark == "gpqa":
        return (f"Question: {example['Question']}\n"
                f"A. {example['Correct Answer']}\n"
                f"B. {example['Incorrect Answer 1']}\n"
                f"C. {example['Incorrect Answer 2']}\n"
                f"D. {example['Incorrect Answer 3']}\nAnswer:", "A")
    raise ValueError(benchmark)


def is_correct(benchmark: str, generated: str, gold: str) -> bool:
    """0/1 reward given a generated string."""
    g = generated.strip()
    if benchmark in ("mmlu", "arc_challenge", "arc_easy", "gpqa"):
        return g[:1].upper() == gold.upper()
    if benchmark == "gsm8k":
        m = re.search(r"-?\d[\d,]*\.?\d*", g)
        return bool(m) and m.group(0).replace(",", "") == gold
    if benchmark == "humaneval":
        return _humaneval_passes(generated, gold)
    return False


def _humaneval_passes(_generated: str, _gold: str) -> bool:
    """Stub; the original results used lm-eval-harness's sandboxed scorer."""
    return False


@torch.inference_mode()
def evaluate_member(ref, benchmark: str, examples, backbone: str) -> Iterable[tuple]:
    """Yield (model_name, qid, benchmark, reward) for one member."""
    model, tokenizer = load_member(ref, backbone_id=backbone)
    model.eval()
    for i, ex in enumerate(examples):
        prompt, gold = format_prompt(benchmark, ex)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        out_ids = model.generate(**inputs, max_new_tokens=64, do_sample=False)
        gen = tokenizer.decode(out_ids[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        yield (f"model_{ref.index}", str(ex.get("id", i)), benchmark,
               1.0 if is_correct(benchmark, gen, gold) else 0.0)


def main(argv=None) -> int:
    args = parse_args(argv)
    examples = load_benchmark(args.benchmark, limit=args.limit)
    members = list_existing_members(args.root)
    print(f"[scan] {len(members)} member(s); {len(examples)} {args.benchmark} examples")

    header_needed = not args.out.exists()
    with args.out.open("a", newline="") as f:
        w = csv.writer(f)
        if header_needed:
            w.writerow(["Model", "Question", "Benchmark", "Reward"])
        for ref in members:
            print(f"  -> member {ref.name}")
            for row in evaluate_member(ref, args.benchmark, examples, args.backbone):
                w.writerow(row)
    print(f"[done] {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
