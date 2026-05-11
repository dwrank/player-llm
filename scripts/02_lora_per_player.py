#!/usr/bin/env python3
"""
02_lora_per_player.py  —  Stage 2: Per-player LoRA adapters.

Trains a separate small LoRA adapter for each well-represented player,
starting from the Stage 1 SFT merged model. Adapters are ~20-50 MB each
and can be swapped at inference time without reloading the base weights.

Default players trained (>= 5,000 decisions):
    Pluribus (15,169), MrBlue (15,015), MrOrange (10,765), Bill (10,348),
    MrPink  (9,134),  Eddie  (8,582),  MrWhite  (6,799)

Adapters are saved to:
    /data/models/qwen2.5-1.5b-instruct/adapters/<player_name>/

Usage:
    python 02_lora_per_player.py                           # all default players
    python 02_lora_per_player.py --players Pluribus MrBlue # specific players
    python 02_lora_per_player.py --min-examples 3000       # lower the threshold
    python 02_lora_per_player.py --dry-run                 # 1 step per player
"""

import argparse
import gc
import json
import sys
import time
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────

SFT_MERGED   = Path("/data/models/qwen2.5-1.5b-instruct/sft/merged")
BASE_MODEL   = Path("/data/models/qwen2.5-1.5b-instruct/base")   # fallback
ADAPTER_ROOT = Path("/data/models/qwen2.5-1.5b-instruct/adapters")
DATA_CACHE   = Path(__file__).parent.parent / "data" / "sft_pluribus.jsonl"
PLURIBUS_DIR = Path("/data/poker/phh-dataset/data/pluribus")

LORA_TARGETS = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]

# Players with enough data for a reliable individual adapter
DEFAULT_MIN_EXAMPLES = 5_000

# ── Args ──────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Stage 2 — per-player LoRA adapters on top of SFT model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--base-model", type=Path, default=None,
                   help="Starting checkpoint (defaults to SFT merged, then base model)")
    p.add_argument("--output",     type=Path, default=ADAPTER_ROOT)
    p.add_argument("--data",       type=Path, default=None,
                   help="SFT JSONL; auto-generated if absent")

    g = p.add_argument_group("player selection")
    g.add_argument("--players",      nargs="*", default=None,
                   help="Explicit list of player names to train")
    g.add_argument("--min-examples", type=int, default=DEFAULT_MIN_EXAMPLES,
                   help="Skip players with fewer examples than this")

    g = p.add_argument_group("training")
    g.add_argument("--epochs",     type=int,   default=3)
    g.add_argument("--batch",      type=int,   default=16)
    g.add_argument("--grad-accum", type=int,   default=2)
    g.add_argument("--lr",         type=float, default=1e-4,
                   help="Lower than Stage 1; we're refining style, not learning poker")
    g.add_argument("--max-seq-len", type=int,  default=256)
    g.add_argument("--eval-split",  type=float, default=0.1)
    g.add_argument("--seed",        type=int,   default=42)

    g = p.add_argument_group("LoRA")
    g.add_argument("--lora-r",     type=int, default=16,
                   help="Rank — smaller than Stage 1 (r=32); style needs less capacity")
    g.add_argument("--lora-alpha", type=int, default=32)

    p.add_argument("--skip-existing", action="store_true",
                   help="Skip players whose adapter directory already exists")
    p.add_argument("--dry-run", action="store_true",
                   help="One training step per player, then exit")
    return p.parse_args()


# ── Dataset ───────────────────────────────────────────────────────────────────

def ensure_dataset(args: argparse.Namespace) -> Path:
    path = args.data or DATA_CACHE
    if path.exists():
        return path

    print(f"Generating SFT dataset from {PLURIBUS_DIR} ...")
    sys.path.insert(0, str(Path(__file__).parent))
    from parse_phh import collect_paths, iter_hands, iter_decisions, decision_to_sft

    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with open(path, "w") as f:
        for hand in iter_hands(collect_paths(PLURIBUS_DIR, [])):
            for dec in iter_decisions(hand):
                f.write(json.dumps(decision_to_sft(dec)) + "\n")
                count += 1
    print(f"Generated {count:,} examples → {path}")
    return path


def load_all_examples(data_path: Path) -> dict[str, list]:
    """Return dict mapping player name → list of {messages: [...]} dicts."""
    by_player: dict[str, list] = {}
    with open(data_path) as f:
        for line in f:
            ex    = json.loads(line)
            name  = ex["messages"][0]["content"].split("You are ")[1].split(",")[0]
            by_player.setdefault(name, []).append({"messages": ex["messages"]})
    return by_player


def select_players(
    by_player: dict[str, list],
    requested: list[str] | None,
    min_examples: int,
) -> list[tuple[str, int]]:
    """Return [(player_name, n_examples)] sorted by n_examples desc."""
    if requested:
        missing = [p for p in requested if p not in by_player]
        if missing:
            print(f"WARNING: players not found in dataset: {missing}")
        selected = [(p, len(by_player[p])) for p in requested if p in by_player]
    else:
        selected = [
            (name, len(examples))
            for name, examples in by_player.items()
            if len(examples) >= min_examples
        ]
    return sorted(selected, key=lambda x: -x[1])


def make_hf_dataset(examples: list, eval_split: float, seed: int):
    from datasets import Dataset
    import random

    rng = random.Random(seed)
    shuffled = list(examples)
    rng.shuffle(shuffled)

    n_eval   = max(1, int(len(shuffled) * eval_split))
    eval_ex  = shuffled[:n_eval]
    train_ex = shuffled[n_eval:]

    return (
        Dataset.from_list(train_ex),
        Dataset.from_list(eval_ex),
    )


# ── Model ─────────────────────────────────────────────────────────────────────

def resolve_base(args: argparse.Namespace) -> Path:
    if args.base_model:
        return args.base_model
    if SFT_MERGED.exists() and any(SFT_MERGED.glob("*.safetensors")):
        return SFT_MERGED
    print(f"WARNING: Stage 1 merged model not found at {SFT_MERGED}.")
    print(f"         Falling back to base model at {BASE_MODEL}.")
    print(f"         Run 01_sft.py first for best results.\n")
    return BASE_MODEL


def load_model(base_path: Path):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        str(base_path),
        quantization_config=bnb,
        device_map="auto",
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
    )
    tokenizer = AutoTokenizer.from_pretrained(str(base_path))
    tokenizer.pad_token    = "<|fim_pad|>"
    tokenizer.padding_side = "right"
    return model, tokenizer


# ── Training ──────────────────────────────────────────────────────────────────

def train_player(
    args:      argparse.Namespace,
    player:    str,
    train_ds,
    eval_ds,
    base_path: Path,
) -> dict:
    """Train one player adapter. Returns metrics dict."""
    import torch
    from peft import LoraConfig
    from trl import SFTConfig, SFTTrainer

    model, tokenizer = load_model(base_path)

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=LORA_TARGETS,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    seqs_per_step  = args.batch * args.grad_accum
    steps_per_epoch = max(len(train_ds) // seqs_per_step, 1)
    eval_steps      = max(steps_per_epoch // 3, 10)   # ~3 evals per epoch

    adapter_dir = args.output / player
    adapter_dir.mkdir(parents=True, exist_ok=True)

    sft_config = SFTConfig(
        output_dir=str(adapter_dir / "checkpoints"),

        num_train_epochs=args.epochs,
        max_steps=1 if args.dry_run else -1,

        per_device_train_batch_size=args.batch,
        per_device_eval_batch_size=args.batch,
        gradient_accumulation_steps=args.grad_accum,

        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        bf16=True,
        fp16=False,
        optim="adamw_bnb_8bit",

        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,       # slightly longer warmup for smaller datasets

        max_length=args.max_seq_len,
        packing=False,
        assistant_only_loss=True,

        eval_strategy="steps",
        eval_steps=eval_steps,
        save_strategy="steps",
        save_steps=eval_steps,
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,

        logging_steps=max(steps_per_epoch // 10, 5),
        report_to="none",
        seed=args.seed,
        data_seed=args.seed,
        dataloader_num_workers=2,
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=tokenizer,
        peft_config=lora_config,
    )

    trainable = sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)
    print(f"  LoRA params : {trainable/1e6:.1f}M  |  "
          f"steps/epoch: {steps_per_epoch}  |  eval every: {eval_steps}")

    t0 = time.time()
    trainer.train()
    elapsed = time.time() - t0

    # Save the adapter weights (not the merged model)
    trainer.model.save_pretrained(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))

    # Read final metrics before releasing trainer state
    log = trainer.state.log_history
    final_train = next(
        (e["loss"] for e in reversed(log) if "loss" in e and "eval_loss" not in e),
        None,
    )
    if final_train is None:
        final_train = next((e.get("train_loss") for e in reversed(log) if "train_loss" in e), None)
    final_eval = next((e["eval_loss"] for e in reversed(log) if "eval_loss" in e), None)
    final_acc  = next((e["eval_mean_token_accuracy"] for e in reversed(log) if "eval_mean_token_accuracy" in e), None)

    # Copy metrics out so log/trainer state can be fully released
    final_train = float(final_train) if final_train is not None else None
    final_eval  = float(final_eval)  if final_eval  is not None else None
    final_acc   = float(final_acc)   if final_acc   is not None else None

    # Clean up to free VRAM before next player.
    # Order matters: release PEFT wrapper first, then base model, then tokenizer.
    del trainer.model
    del trainer
    del model
    del tokenizer
    gc.collect()
    torch.cuda.synchronize()   # wait for all GPU kernels to finish
    torch.cuda.empty_cache()   # return freed blocks to CUDA allocator

    return {
        "player":      player,
        "train_ex":    len(train_ds),
        "eval_ex":     len(eval_ds),
        "train_loss":  final_train,
        "eval_loss":   final_eval,
        "eval_acc":    final_acc,
        "elapsed_min": elapsed / 60,
        "adapter_dir": str(adapter_dir),
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    args      = parse_args()
    base_path = resolve_base(args)

    print(f"Base model : {base_path}")
    print(f"Adapters   → {args.output}\n")

    data_path  = ensure_dataset(args)
    by_player  = load_all_examples(data_path)
    to_train   = select_players(by_player, args.players, args.min_examples)

    if not to_train:
        print("No players selected. Use --players or lower --min-examples.")
        return

    print(f"Players to train ({len(to_train)}):")
    for name, n in to_train:
        status = "(exists — skipping)" if (
            args.skip_existing and (args.output / name / "adapter_config.json").exists()
        ) else ""
        print(f"  {name:12s}  {n:6,} examples  {status}")
    print()

    if args.dry_run:
        print("DRY RUN — one training step per player\n")

    results = []
    for i, (player, n_ex) in enumerate(to_train, 1):
        adapter_cfg = args.output / player / "adapter_config.json"
        if args.skip_existing and adapter_cfg.exists():
            print(f"[{i}/{len(to_train)}] {player} — skipping (adapter exists)")
            continue

        print(f"[{i}/{len(to_train)}] Training {player}  ({n_ex:,} examples)")

        train_ds, eval_ds = make_hf_dataset(
            by_player[player], args.eval_split, args.seed
        )
        print(f"  Split : {len(train_ds):,} train / {len(eval_ds):,} eval")

        metrics = train_player(args, player, train_ds, eval_ds, base_path)
        results.append(metrics)

        tl = f"{metrics['train_loss']:.4f}" if metrics["train_loss"] else "—"
        el = f"{metrics['eval_loss']:.4f}"  if metrics["eval_loss"]  else "—"
        ac = f"{metrics['eval_acc']:.4f}"   if metrics["eval_acc"]   else "—"
        print(f"  Done  : {metrics['elapsed_min']:.1f} min  |  "
              f"train_loss={tl}  eval_loss={el}  eval_acc={ac}")
        print(f"  Saved : {metrics['adapter_dir']}\n")

    # Summary table
    if len(results) > 1:
        print("=" * 68)
        print(f"{'Player':<12}  {'Train':>6}  {'Eval':>6}  "
              f"{'train_loss':>10}  {'eval_loss':>9}  {'acc':>6}  {'min':>5}")
        print("-" * 68)
        for r in results:
            tl = f"{r['train_loss']:.4f}" if r["train_loss"] else "    —"
            el = f"{r['eval_loss']:.4f}"  if r["eval_loss"]  else "    —"
            ac = f"{r['eval_acc']:.4f}"   if r["eval_acc"]   else "   —"
            print(f"{r['player']:<12}  {r['train_ex']:>6,}  {r['eval_ex']:>6,}  "
                  f"{tl:>10}  {el:>9}  {ac:>6}  {r['elapsed_min']:>5.1f}")
        print("=" * 68)
        print(f"\nAll adapters saved under {args.output}/")
        print("Load at inference with:")
        print("  from peft import PeftModel")
        print("  model = PeftModel.from_pretrained(base_model, adapter_dir)")


if __name__ == "__main__":
    main()
