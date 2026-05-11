"""Load + snapshot LLM ensemble members (full fine-tune or LoRA)."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
except ImportError as e:
    raise ImportError("Install with `uv pip install -e '.[llm]'`.") from e


DEFAULT_BACKBONE = "deepseek-ai/deepseek-llm-7b-chat"


@dataclass
class MemberRef:
    """Pointer to one ensemble member on disk."""
    index: int
    name: str
    weights_dir: Path
    is_lora: bool = False

    @classmethod
    def for_index(cls, root: Path, idx: int, is_lora: bool = False) -> "MemberRef":
        name = _excel_name(idx)
        return cls(index=idx, name=name, weights_dir=root / f"member_{name}", is_lora=is_lora)


def _excel_name(n: int) -> str:
    """A, B, ..., Z, AA, AB, ..."""
    name = ""
    while n >= 0:
        name = chr(ord("A") + n % 26) + name
        n = n // 26 - 1
    return name


def load_backbone(
    model_id: str = DEFAULT_BACKBONE,
    dtype: "torch.dtype" = torch.bfloat16,
    device_map: Optional[str] = "auto",
):
    """Load backbone weights + tokenizer."""
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=dtype, device_map=device_map, trust_remote_code=True,
    )
    return model, tokenizer


def load_member(ref: MemberRef, backbone_id: str = DEFAULT_BACKBONE):
    """Load a saved member (full FT or LoRA adapter on the backbone)."""
    if ref.is_lora:
        try:
            from peft import PeftModel
        except ImportError as e:
            raise ImportError("Install with `uv pip install -e '.[llm]'`.") from e
        base, tok = load_backbone(backbone_id)
        return PeftModel.from_pretrained(base, str(ref.weights_dir)), tok

    tokenizer = AutoTokenizer.from_pretrained(str(ref.weights_dir), trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        str(ref.weights_dir), torch_dtype=torch.bfloat16,
        device_map="auto", trust_remote_code=True,
    )
    return model, tokenizer


def list_existing_members(root: Path) -> List[MemberRef]:
    """Discover saved member_* directories under root."""
    root = Path(root)
    out: List[MemberRef] = []
    idx = 0
    while True:
        cand = root / f"member_{_excel_name(idx)}"
        if not cand.exists():
            break
        out.append(MemberRef(index=idx, name=_excel_name(idx),
                             weights_dir=cand,
                             is_lora=(cand / "adapter_config.json").exists()))
        idx += 1
    return out
