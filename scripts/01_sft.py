#!/usr/bin/env python3
"""
01_sft.py  —  Stage 1: Joint Supervised Fine-Tuning on Pluribus data.

Fine-tunes Qwen2.5-1.5B-Instruct on all 91K decision points from the
Pluribus dataset using QLoRA (4-bit NF4 base + LoRA). All 14 players
are trained jointly; player identity is captured via the system prompt.

Outputs:
  /data/models/qwen2.5-1.5b-instruct/sft/adapter/   — LoRA weights only
  /data/models/qwen2.5-1.5b-instruct/sft/merged/    — merged fp16 model
                                                        (use as Stage 2 base)
Usage:
    python 01_sft.py                         # full 3-epoch run
    python 01_sft.py --dry-run               # one step, verify setup
    python 01_sft.py --epochs 1 --batch 4   # quick smoke test
    python 01_sft.py --data /path/sft.jsonl # pre-generated dataset
"""

import argparse
import json
import sys
import time
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────

BASE_MODEL  = Path("/data/models/qwen2.5-1.5b-instruct/base")
SFT_OUTPUT  = Path("/data/models/qwen2.5-1.5b-instruct/sft")
PLURIBUS    = Path("/data/poker/phh-dataset/data/pluribus")
DATA_CACHE  = Path(__file__).parent.parent / "data" / "sft_pluribus.jsonl"

LORA_TARGETS = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]

# ── Args ──────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Stage 1 SFT — joint fine-tuning on all Pluribus players",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--base-model",   type=Path, default=BASE_MODEL)
    p.add_argument("--output",       type=Path, default=SFT_OUTPUT)
    p.add_argument("--data",         type=Path, default=None,
                   help="SFT JSONL file; auto-generated from --pluribus-dir if absent")
    p.add_argument("--pluribus-dir", type=Path, default=PLURIBUS)

    g = p.add_argument_group("training")
    g.add_argument("--epochs",      type=int,   default=3)
    g.add_argument("--batch",       type=int,   default=8,
                   help="Per-device train batch size")
    g.add_argument("--grad-accum",  type=int,   default=8,
                   help="Gradient accumulation steps")
    g.add_argument("--lr",          type=float, default=2e-4)
    g.add_argument("--max-seq-len", type=int,   default=512,
                   help="Max sequence length; examples range 210–380 tokens")
    g.add_argument("--eval-split",  type=float, default=0.1)
    g.add_argument("--seed",        type=int,   default=42)

    g = p.add_argument_group("LoRA")
    g.add_argument("--lora-r",     type=int, default=32)
    g.add_argument("--lora-alpha", type=int, default=64)

    p.add_argument("--no-merge", action="store_true",
                   help="Skip merging LoRA into base weights at end")
    p.add_argument("--dry-run",  action="store_true",
                   help="Run one training step to verify setup, then exit")
    return p.parse_args()


# ── Dataset ───────────────────────────────────────────────────────────────────

def ensure_dataset(args: argparse.Namespace) -> Path:
    """Return path to SFT JSONL, generating from Pluribus PHH files if needed."""
    path = args.data or DATA_CACHE
    if path.exists():
        n = sum(1 for _ in open(path))
        print(f"Dataset : {path}  ({n:,} examples, existing)")
        return path

    print(f"Generating SFT dataset from {args.pluribus_dir} ...")
    sys.path.insert(0, str(Path(__file__).parent))
    from parse_phh import collect_paths, iter_hands, iter_decisions, decision_to_sft

    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with open(path, "w") as f:
        for hand in iter_hands(collect_paths(args.pluribus_dir, [])):
            for dec in iter_decisions(hand):
                f.write(json.dumps(decision_to_sft(dec)) + "\n")
                count += 1
    print(f"Generated {count:,} examples → {path}")
    return path


def load_splits(data_path: Path, eval_split: float, seed: int):
    from datasets import load_dataset

    ds = load_dataset("json", data_files=str(data_path), split="train")
    ds = ds.select_columns(["messages"])
    ds = ds.shuffle(seed=seed)

    n_eval   = int(len(ds) * eval_split)
    eval_ds  = ds.select(range(n_eval))
    train_ds = ds.select(range(n_eval, len(ds)))

    print(f"Split   : {len(train_ds):,} train / {len(eval_ds):,} eval")
    return train_ds, eval_ds


# ── Model ─────────────────────────────────────────────────────────────────────

def load_model_and_tokenizer(base_model: Path):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    print(f"Loading : {base_model}  (4-bit NF4, bf16 compute)")
    model = AutoModelForCausalLM.from_pretrained(
        str(base_model),
        quantization_config=bnb,
        device_map="auto",
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
    )

    tokenizer = AutoTokenizer.from_pretrained(str(base_model))
    # <|fim_pad|> is a dedicated non-eos pad token in Qwen's vocab
    tokenizer.pad_token = "<|fim_pad|>"
    tokenizer.padding_side = "right"

    total   = sum(p.numel() for p in model.parameters())
    print(f"Model   : {total/1e9:.2f}B params  |  "
          f"device map: {set(str(p.device) for p in model.parameters())}")
    return model, tokenizer


# ── Callbacks ─────────────────────────────────────────────────────────────────

class SampleCallback:
    """Print a handful of model predictions at the end of each epoch."""

    def __init__(self, eval_ds, tokenizer, n_samples: int = 5):
        import random
        self.samples   = [eval_ds[i] for i in random.sample(range(len(eval_ds)), min(n_samples, len(eval_ds)))]
        self.tokenizer = tokenizer

    def on_epoch_end(self, args, state, control, model=None, **kwargs):
        if model is None:
            return
        import torch
        print("\n── Sample predictions ──────────────────────────────────")
        model.eval()
        for ex in self.samples:
            msgs  = ex["messages"]
            label = msgs[-1]["content"]
            # Format prompt without the assistant turn
            prompt = self.tokenizer.apply_chat_template(
                msgs[:-1],
                tokenize=False,
                add_generation_prompt=True,
            )
            enc = self.tokenizer(prompt, return_tensors="pt")
            ids  = enc.input_ids.to(model.device)
            mask = enc.attention_mask.to(model.device)
            with torch.no_grad():
                out = model.generate(
                    ids,
                    attention_mask=mask,
                    max_new_tokens=1,
                    do_sample=False,
                    pad_token_id=self.tokenizer.pad_token_id,
                )
            pred_char = self.tokenizer.decode(
                out[0][ids.shape[-1]:], skip_special_tokens=True
            ).strip()
            from action_classes import char_to_display, char_to_display
            pred_label = char_to_display(label) if label else "?"
            pred_disp  = f"{pred_char}={char_to_display(pred_char)}" if pred_char else "?"
            # Print just the relevant context line from the user message
            user_lines = msgs[1]["content"].split("\n")
            cards_line  = next((l for l in user_lines if "hole cards" in l), "")
            street_line = next((l for l in user_lines if "Street:" in l), "")
            print(f"  {street_line}  {cards_line}")
            print(f"  label={label}({pred_label})  pred={pred_disp}")
        print("────────────────────────────────────────────────────────\n")
        model.train()


# ── Training ──────────────────────────────────────────────────────────────────

def run_training(args, model, tokenizer, train_ds, eval_ds):
    from peft import LoraConfig
    from trl import SFTConfig, SFTTrainer
    from transformers import TrainerCallback

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=LORA_TARGETS,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        modules_to_save=["lm_head"],   # train classification head; saved with adapter
    )

    # Packing requires flash_attention_2 for correct cross-sequence masking.
    # With SDPA we run unpacked: one example per sequence, padded to max_length.
    # Compensate with larger batch size (batch=16 default gives ~64 eff batch).
    examples_per_seq  = 1
    seqs_per_step     = args.batch * args.grad_accum
    steps_per_epoch   = max(len(train_ds) // seqs_per_step, 1)
    eval_steps        = max(steps_per_epoch // 4, 50)   # ~4 evals per epoch

    ckpt_dir = args.output / "checkpoints"

    sft_config = SFTConfig(
        # Paths
        output_dir=str(ckpt_dir),

        # Schedule
        num_train_epochs=args.epochs,
        max_steps=1 if args.dry_run else -1,

        # Batching
        per_device_train_batch_size=args.batch,
        per_device_eval_batch_size=args.batch,
        gradient_accumulation_steps=args.grad_accum,

        # Memory / precision
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        bf16=True,
        fp16=False,
        optim="paged_adamw_8bit",

        # Learning rate
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,

        # Sequence / loss masking
        # packing=True requires flash_attention_2 for correct cross-sequence
        # masking; SDPA does not support it safely, so we run unpacked.
        max_length=args.max_seq_len,
        packing=False,
        assistant_only_loss=True,   # only backprop through assistant turns

        # Eval / saving
        eval_strategy="steps",
        eval_steps=eval_steps,
        save_strategy="steps",
        save_steps=eval_steps,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,

        # Logging
        logging_steps=10,
        report_to="none",

        # Reproducibility
        seed=args.seed,
        data_seed=args.seed,
        dataloader_num_workers=2,
    )

    sample_cb = SampleCallback(eval_ds, tokenizer)

    # Wrap as a HF TrainerCallback
    class _SampleCB(TrainerCallback):
        def on_epoch_end(self, a, state, ctrl, **kw):
            sample_cb.on_epoch_end(a, state, ctrl, **kw)

    eff_batch = args.batch * args.grad_accum * examples_per_seq
    print(f"\nConfig  : LoRA r={args.lora_r}/α={args.lora_alpha} | "
          f"epochs={args.epochs} | lr={args.lr}")
    print(f"Batching: {args.batch} seqs × {args.grad_accum} accum "
          f"= {eff_batch} examples/step  (packing disabled, requires flash-attn)")
    print(f"Eval    : every {eval_steps} steps  (~4× per epoch)\n")

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=tokenizer,
        peft_config=lora_config,
        callbacks=[_SampleCB()],
    )

    # Report trainable parameter count after PEFT wrapping
    trainable = sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in trainer.model.parameters())
    print(f"Params  : {trainable/1e6:.1f}M trainable / {total/1e9:.2f}B total "
          f"({100*trainable/total:.2f}%)\n")

    t0 = time.time()
    trainer.train()
    elapsed = time.time() - t0

    log = trainer.state.log_history
    # "loss" key appears in per-step logs; "train_loss" in the final summary
    final_train = next((e["loss"] for e in reversed(log) if "loss" in e and "eval_loss" not in e), None)
    if final_train is None:
        final_train = next((e.get("train_loss") for e in reversed(log) if "train_loss" in e), None)
    final_eval = next((e["eval_loss"] for e in reversed(log) if "eval_loss" in e), None)

    parts = [f"\nDone    : {elapsed/60:.1f} min"]
    if final_train is not None:
        parts.append(f"train_loss={final_train:.4f}")
    if final_eval is not None:
        parts.append(f"eval_loss={final_eval:.4f}")
    print("  |  ".join(parts))

    return trainer


# ── Save ──────────────────────────────────────────────────────────────────────

def save(args, trainer, tokenizer) -> None:
    import torch

    adapter_dir = args.output / "adapter"
    adapter_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nSaving adapter → {adapter_dir}")
    trainer.model.save_pretrained(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))

    if args.no_merge or args.dry_run:
        if args.dry_run:
            print("Dry run complete — skipping merge.")
        else:
            print("Skipping merge (--no-merge).  Load with base + adapter.")
        return

    # NOTE: merge_and_unload() on a 4-bit base keeps weights in NF4 uint8
    # (~1.1 GB). For a clean fp16/bf16 merge, run merge_adapter.py separately.
    print("Merging LoRA into base weights ...")
    merged = trainer.model.merge_and_unload()

    # Zero lm_head rows 22+ so non-class tokens never win argmax at inference.
    with torch.no_grad():
        merged.lm_head.weight[22:] = 0.0
    print("Classification head: zeroed lm_head rows 22+ (vocab size → 22 active classes)")

    merged_dir = args.output / "merged"
    merged_dir.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(str(merged_dir), safe_serialization=True)
    tokenizer.save_pretrained(str(merged_dir))

    size_gb = sum(f.stat().st_size for f in merged_dir.glob("*.safetensors")) / 1e9
    print(f"Merged  → {merged_dir}  ({size_gb:.2f} GB, bf16)")
    print(f"\nStage 1 complete.")
    print(f"  Adapter : {adapter_dir}")
    print(f"  Merged  : {merged_dir}  ← use as --base-model for 02_lora_per_player.py")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    if args.dry_run:
        print("=" * 60)
        print("DRY RUN  —  one training step, then exit")
        print("=" * 60)

    data_path          = ensure_dataset(args)
    train_ds, eval_ds  = load_splits(data_path, args.eval_split, args.seed)
    model, tokenizer   = load_model_and_tokenizer(args.base_model)
    trainer            = run_training(args, model, tokenizer, train_ds, eval_ds)
    save(args, trainer, tokenizer)


if __name__ == "__main__":
    main()
