#!/usr/bin/env python3
"""
inference.py  —  Run a player adapter for poker action prediction.

Loads the SFT merged base model and a per-player LoRA adapter, then generates
actions for game states using the same chat format as training.

Game state format (user message, passed via --state, file, or stdin):
    Street: Preflop
    Position: BTN
    Your hole cards: Ah Kd
    Board: none
    Pot: $300
    Your stack: $9,800
    To call: $100
    Other active players: SB $9,950, BB $9,900
    Action so far this street: UTG fold, HJ fold, CO fold
    Your action?

Usage:
    # Interactive REPL
    python inference.py --player Pluribus

    # Single prediction (state from stdin)
    cat game_state.txt | python inference.py --player MrBlue

    # Single prediction (state from file)
    python inference.py --player Eddie --state game_state.txt

    # Lower memory: load base in 4-bit (requires CUDA + bitsandbytes)
    python inference.py --player Pluribus --load-in-4bit

    # Multiple players, same game state
    python inference.py --players Pluribus MrBlue MrOrange --state game_state.txt

    # Batch: JSONL of {player, state} objects → JSONL of {player, state, action}
    python inference.py --batch input.jsonl --output predictions.jsonl
"""

import argparse
import json
import sys
from pathlib import Path

BASE_MODEL   = Path("/data/models/qwen2.5-1.5b-instruct/sft/merged")
ADAPTER_ROOT = Path("/data/models/qwen2.5-1.5b-instruct/adapters")

KNOWN_PLAYERS = [
    "Pluribus", "MrBlue", "MrOrange", "Bill", "MrPink", "Eddie", "MrWhite",
]

# Match training system prompt exactly
def system_prompt(player: str, blinds: str = "$50/$100", stacks: str = "$10,000") -> str:
    return (
        f"You are {player}, playing 6-handed no-limit Texas Hold'em "
        f"({blinds} blinds, {stacks} starting stacks). "
        f"Make the decision {player} would make. "
        f"Reply with exactly one of: fold | check | call | raise <total_amount>"
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Poker action prediction using a player LoRA adapter",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    g = p.add_argument_group("player / model")
    g.add_argument("--player",  default=None,
                   help="Player name (single-player mode)")
    g.add_argument("--players", nargs="+", default=None,
                   help="Multiple player names (prints one prediction per player)")
    g.add_argument("--base",    type=Path, default=BASE_MODEL,
                   help="Merged SFT base model directory")
    g.add_argument("--adapters", type=Path, default=ADAPTER_ROOT,
                   help="Root directory containing per-player adapter subdirs")
    g.add_argument("--no-adapter", action="store_true",
                   help="Run base model only (no player adapter)")

    g = p.add_argument_group("game parameters")
    g.add_argument("--blinds", default="$50/$100",
                   help="Blind levels for system prompt")
    g.add_argument("--stacks", default="$10,000",
                   help="Starting stacks for system prompt")

    g = p.add_argument_group("input")
    g.add_argument("--state", type=Path, default=None,
                   help="File containing the game state (user message). "
                        "Reads from stdin if omitted and not interactive.")
    g.add_argument("--interactive", "-i", action="store_true",
                   help="REPL mode: paste game state, blank line to submit")
    g.add_argument("--batch", type=Path, default=None,
                   help="JSONL with {player, state} objects")
    g.add_argument("--output", type=Path, default=None,
                   help="Output JSONL path for --batch mode (default: stdout)")

    g = p.add_argument_group("generation")
    g.add_argument("--max-new-tokens", type=int, default=16)
    g.add_argument("--temperature",    type=float, default=0.0,
                   help="0 = greedy (deterministic), >0 = sampling")
    g.add_argument("--top-p",          type=float, default=1.0)

    g = p.add_argument_group("memory")
    g.add_argument("--load-in-4bit", action="store_true",
                   help="Load base in 4-bit NF4 (saves ~2 GB VRAM, requires CUDA)")
    g.add_argument("--load-in-8bit", action="store_true",
                   help="Load base in 8-bit (saves ~1.5 GB VRAM, requires CUDA)")
    g.add_argument("--device", default=None,
                   help="Force device: cuda / cpu / mps (default: auto)")

    return p.parse_args()


# ── Model loading ─────────────────────────────────────────────────────────────

def load_base(args: argparse.Namespace):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    if not args.base.exists():
        print(f"ERROR: base model not found: {args.base}", file=sys.stderr)
        print(f"  Run merge_adapter.py to create it, or pass --base.", file=sys.stderr)
        sys.exit(1)

    kwargs: dict = {"device_map": args.device or "auto", "attn_implementation": "sdpa"}

    if args.load_in_4bit:
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        label = "4-bit NF4"
    elif args.load_in_8bit:
        kwargs["load_in_8bit"] = True
        label = "8-bit"
    else:
        kwargs["dtype"] = torch.float16
        label = "fp16"

    print(f"Loading base ({label}): {args.base}", file=sys.stderr)
    model = AutoModelForCausalLM.from_pretrained(str(args.base), **kwargs)
    tokenizer = AutoTokenizer.from_pretrained(str(args.base))
    return model, tokenizer


def attach_adapter(model, player: str, adapter_root: Path):
    from peft import PeftModel

    adapter_dir = adapter_root / player
    if not (adapter_dir / "adapter_config.json").exists():
        print(f"ERROR: adapter not found for {player}: {adapter_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"Loading adapter : {adapter_dir}", file=sys.stderr)
    return PeftModel.from_pretrained(model, str(adapter_dir), adapter_name=player)


# ── Inference ─────────────────────────────────────────────────────────────────

def predict(
    model,
    tokenizer,
    player: str,
    state: str,
    args: argparse.Namespace,
) -> str:
    import torch

    messages = [
        {"role": "system",    "content": system_prompt(player, args.blinds, args.stacks)},
        {"role": "user",      "content": state.strip()},
    ]

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    enc  = tokenizer(prompt, return_tensors="pt")
    ids  = enc.input_ids.to(model.device)
    mask = enc.attention_mask.to(model.device)

    gen_kwargs: dict = {
        "attention_mask":  mask,
        "max_new_tokens":  args.max_new_tokens,
        "pad_token_id":    tokenizer.pad_token_id or tokenizer.eos_token_id,
        "eos_token_id":    tokenizer.eos_token_id,
    }
    if args.temperature > 0:
        gen_kwargs.update(do_sample=True, temperature=args.temperature, top_p=args.top_p)
    else:
        gen_kwargs["do_sample"] = False

    with torch.no_grad():
        out = model.generate(ids, **gen_kwargs)

    action = tokenizer.decode(
        out[0][ids.shape[-1]:], skip_special_tokens=True
    ).strip()
    return action


# ── Input helpers ─────────────────────────────────────────────────────────────

def read_state_from_file(path: Path) -> str:
    return path.read_text()


def read_state_from_stdin() -> str:
    if sys.stdin.isatty():
        print("Paste game state (blank line to submit):", file=sys.stderr)
        lines = []
        while True:
            line = input()
            if line == "":
                break
            lines.append(line)
        return "\n".join(lines)
    return sys.stdin.read()


def interactive_loop(model, tokenizer, players: list[str], args: argparse.Namespace) -> None:
    print(f"Player(s): {', '.join(players)}", file=sys.stderr)
    print("Paste game state, blank line to predict. Ctrl-C to quit.\n", file=sys.stderr)
    while True:
        try:
            state = read_state_from_stdin()
            if not state.strip():
                continue
            for player in players:
                if len(players) > 1 and hasattr(model, "set_adapter"):
                    model.set_adapter(player)
                action = predict(model, tokenizer, player, state, args)
                prefix = f"{player}: " if len(players) > 1 else ""
                print(f"{prefix}{action}")
            print()
        except KeyboardInterrupt:
            print("\nBye.", file=sys.stderr)
            break
        except EOFError:
            break


def run_batch(model, tokenizer, args: argparse.Namespace) -> None:
    out = open(args.output, "w") if args.output else sys.stdout
    try:
        with open(args.batch) as f:
            for i, line in enumerate(f, 1):
                rec    = json.loads(line)
                player = rec["player"]
                state  = rec["state"]
                if hasattr(model, "set_adapter"):
                    try:
                        model.set_adapter(player)
                    except Exception:
                        pass
                action = predict(model, tokenizer, player, state, args)
                out.write(json.dumps({"player": player, "state": state, "action": action}) + "\n")
                if i % 100 == 0:
                    print(f"  {i} predictions done", file=sys.stderr)
    finally:
        if args.output:
            out.close()


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    # Resolve player list
    players: list[str] = []
    if args.players:
        players = args.players
    elif args.player:
        players = [args.player]
    elif args.batch:
        players = []  # determined per-row in batch mode
    elif not args.no_adapter:
        print("ERROR: specify --player, --players, --no-adapter, or --batch", file=sys.stderr)
        sys.exit(1)

    model, tokenizer = load_base(args)

    if not args.no_adapter and players:
        # Load first adapter; additional adapters loaded lazily via set_adapter
        model = attach_adapter(model, players[0], args.adapters)
        for player in players[1:]:
            from peft import PeftModel
            model.load_adapter(str(args.adapters / player), adapter_name=player)

    model.eval()

    if args.batch:
        run_batch(model, tokenizer, args)
        return

    if args.interactive or (not args.state and sys.stdin.isatty()):
        interactive_loop(model, tokenizer, players or ["base"], args)
        return

    # Single prediction
    state = read_state_from_file(args.state) if args.state else sys.stdin.read()
    for player in (players or ["base"]):
        if len(players) > 1 and hasattr(model, "set_adapter"):
            model.set_adapter(player)
        action = predict(model, tokenizer, player, state, args)
        prefix = f"{player}: " if len(players) > 1 else ""
        print(f"{prefix}{action}")


if __name__ == "__main__":
    main()
