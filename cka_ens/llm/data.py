"""Alpaca-2k instruction-tuning data for the LLM members."""
from __future__ import annotations

from dataclasses import dataclass

try:
    import datasets
    from transformers import AutoTokenizer, PreTrainedTokenizerBase
    import torch
    from torch.utils.data import DataLoader, Dataset
except ImportError as e:
    raise ImportError("Install with `uv pip install -e '.[llm]'`.") from e


ALPACA_TEMPLATE_WITH_INPUT = (
    "Below is an instruction that describes a task, paired with an input "
    "that provides further context. Write a response that appropriately "
    "completes the request.\n\n"
    "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n"
)
ALPACA_TEMPLATE_NO_INPUT = (
    "Below is an instruction that describes a task. Write a response that "
    "appropriately completes the request.\n\n"
    "### Instruction:\n{instruction}\n\n### Response:\n"
)


@dataclass
class AlpacaConfig:
    """Slice hyperparameters."""
    n_samples: int = 2000
    max_length: int = 128
    seed: int = 0
    hf_dataset_id: str = "tatsu-lab/alpaca"


class AlpacaFineTuneDataset(Dataset):
    """Token-level supervision on the response only (prompt is masked)."""

    def __init__(self, cfg: AlpacaConfig, tokenizer: "PreTrainedTokenizerBase"):
        ds = datasets.load_dataset(cfg.hf_dataset_id, split="train")
        self.examples = ds.shuffle(seed=cfg.seed).select(range(cfg.n_samples))
        self.tokenizer = tokenizer
        self.max_length = cfg.max_length

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int):
        ex = self.examples[idx]
        prompt = (ALPACA_TEMPLATE_WITH_INPUT.format(
                    instruction=ex["instruction"], input=ex["input"])
                  if ex.get("input") else
                  ALPACA_TEMPLATE_NO_INPUT.format(instruction=ex["instruction"]))
        response = ex["output"] + self.tokenizer.eos_token
        prompt_ids = self.tokenizer(prompt, add_special_tokens=False)["input_ids"]
        response_ids = self.tokenizer(response, add_special_tokens=False)["input_ids"]
        full = (prompt_ids + response_ids)[: self.max_length]
        labels = ([-100] * len(prompt_ids) + response_ids)[: self.max_length]
        return {
            "input_ids": torch.tensor(full, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.tensor([1] * len(full), dtype=torch.long),
        }


def collate(batch, pad_id: int):
    """Right-pad to the longest sequence in the batch."""
    max_len = max(item["input_ids"].size(0) for item in batch)
    out = {"input_ids": [], "labels": [], "attention_mask": []}
    for item in batch:
        pad = max_len - item["input_ids"].size(0)
        out["input_ids"].append(torch.cat(
            [item["input_ids"], torch.full((pad,), pad_id, dtype=torch.long)]))
        out["labels"].append(torch.cat(
            [item["labels"], torch.full((pad,), -100, dtype=torch.long)]))
        out["attention_mask"].append(torch.cat(
            [item["attention_mask"], torch.zeros(pad, dtype=torch.long)]))
    return {k: torch.stack(v) for k, v in out.items()}


def build_dataloader(
    cfg: AlpacaConfig,
    tokenizer: "PreTrainedTokenizerBase",
    batch_size: int = 4,
    shuffle: bool = True,
) -> "DataLoader":
    """Yield a DataLoader over the Alpaca-2k slice."""
    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id
    return DataLoader(
        AlpacaFineTuneDataset(cfg, tokenizer),
        batch_size=batch_size, shuffle=shuffle,
        collate_fn=lambda b: collate(b, pad_id),
    )
