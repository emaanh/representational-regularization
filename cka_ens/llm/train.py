"""Sequential LLM ensemble trainer (control or CKA-penalty), one member at a time."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

try:
    import torch
    from torch.optim import AdamW
except ImportError as e:
    raise ImportError("Install with `uv pip install -e '.[llm]'`.") from e

from .data import AlpacaConfig, build_dataloader
from .hooks import capture_layer_activations, select_anchor_layers
from .model_loader import (
    DEFAULT_BACKBONE, MemberRef,
    list_existing_members, load_backbone, load_member,
)
from .penalty import sequential_cka_penalty


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--idx", type=int, required=True)
    p.add_argument("--mode", choices=["control", "cka"], required=True)
    p.add_argument("--lambda", dest="lam", type=float, default=10.0)
    p.add_argument("--backbone", type=str, default=DEFAULT_BACKBONE)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--max-length", type=int, default=128)
    p.add_argument("--n-samples", type=int, default=2000)
    p.add_argument("--every", type=int, default=4)
    p.add_argument("--lora", action="store_true")
    return p.parse_args(argv)


def _wrap_lora(model):
    """Attach a rank-16 LoRA to the attention projections."""
    try:
        from peft import LoraConfig, get_peft_model
    except ImportError as e:
        raise ImportError("Install with `uv pip install -e '.[llm]'`.") from e
    cfg = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        task_type="CAUSAL_LM",
    )
    return get_peft_model(model, cfg)


def main(argv=None) -> int:
    args = parse_args(argv)
    args.root.mkdir(parents=True, exist_ok=True)
    ref = MemberRef.for_index(args.root, args.idx, is_lora=args.lora)
    if ref.weights_dir.exists():
        print(f"[skip] member {ref.name} already exists")
        return 0

    print(f"[load] backbone {args.backbone}")
    model, tokenizer = load_backbone(args.backbone, dtype=torch.bfloat16)
    if args.lora:
        model = _wrap_lora(model)
    model.gradient_checkpointing_enable()
    model.train()

    anchors: List = []
    if args.mode == "cka" and args.idx > 0:
        for prev_ref in list_existing_members(args.root)[: args.idx]:
            anchor, _ = load_member(prev_ref, backbone_id=args.backbone)
            anchor.eval()
            for p in anchor.parameters():
                p.requires_grad_(False)
            anchors.append(anchor)
        print(f"[load] {len(anchors)} frozen anchor(s).")

    loader = build_dataloader(
        AlpacaConfig(n_samples=args.n_samples, max_length=args.max_length),
        tokenizer, batch_size=args.batch_size, shuffle=True,
    )
    anchor_layers = select_anchor_layers(model, every=args.every)
    print(f"[hook] anchor layer indices: {anchor_layers}")
    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)

    step = 0
    for epoch in range(args.epochs):
        for batch in loader:
            batch = {k: v.to(model.device) for k, v in batch.items()}
            with capture_layer_activations(model, anchor_layers) as cur_acts:
                outputs = model(**batch)
                task_loss = outputs.loss

            prev_acts_per_member = []
            if anchors:
                with torch.no_grad():
                    for anchor in anchors:
                        with capture_layer_activations(anchor, anchor_layers) as a:
                            _ = anchor(
                                input_ids=batch["input_ids"],
                                attention_mask=batch["attention_mask"],
                            )
                        prev_acts_per_member.append([t.detach() for t in a])

            penalty = sequential_cka_penalty(cur_acts, prev_acts_per_member)
            loss = task_loss + args.lam * penalty
            loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            if step % 25 == 0:
                print(f"step {step:>5d}  task={task_loss.item():.4f}  "
                      f"pen={penalty.item():.4f}  total={loss.item():.4f}")
            step += 1

    ref.weights_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(ref.weights_dir))
    if not args.lora:
        tokenizer.save_pretrained(str(ref.weights_dir))
    print(f"[save] {ref.weights_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
