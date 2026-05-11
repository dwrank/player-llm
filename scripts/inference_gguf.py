#!/usr/bin/env python3
"""
inference_gguf.py  —  Poker action prediction using GGUF models via llama-cpp-python.

Two modes:

  Merged mode (convert_to_gguf.py output):
    One GGUF per player (base + adapter merged). Simple but ~1 GB per player.

  Adapter mode (convert_adapters_to_gguf.py output):
    Shared base GGUF + small per-player adapter GGUF. More storage-efficient.

Install:
    pip install llama-cpp-python
    # With CUDA: CMAKE_ARGS="-DGGML_CUDA=on" pip install llama-cpp-python --no-cache-dir
    # With Metal: CMAKE_ARGS="-DGGML_METAL=on" pip install llama-cpp-python --no-cache-dir

Merged mode usage:
    cat game_state.txt | python inference_gguf.py --player Pluribus
    python inference_gguf.py --players Pluribus MrBlue Eddie --state game_state.txt
    python inference_gguf.py --player Pluribus --interactive
    python inference_gguf.py --batch input.jsonl --output predictions.jsonl
    python inference_gguf.py --player Pluribus --state game_state.txt --gpu-layers -1

Adapter mode usage:
    python inference_gguf.py --base-gguf /data/models/qwen2.5-1.5b-instruct/gguf-adapters/base.Q8_0.gguf \\
        --player Pluribus --state game_state.txt
    python inference_gguf.py --base-gguf base.Q8_0.gguf \\
        --lora-dir /data/models/qwen2.5-1.5b-instruct/gguf-adapters/adapters \\
        --players Pluribus MrBlue Eddie --state game_state.txt
"""

import argparse
import json
import sys
from pathlib import Path

# Resolve action_classes relative to this script's directory
sys.path.insert(0, str(Path(__file__).resolve().parent))
from action_classes import (
    CLASS_VOCAB_IDS, CLASS_TOKENS_LIST,
    char_to_display, CHAR_TO_CLASS,
)

GGUF_DIR      = Path("/data/models/qwen2.5-1.5b-instruct/gguf")
DEFAULT_QUANT = "Q4_K_M"

KNOWN_PLAYERS = [
    "Pluribus", "MrBlue", "MrOrange", "Bill", "MrPink", "Eddie", "MrWhite",
]

# Logit bias: suppress every token that is not one of the 22 class tokens.
# Applied at inference so the model always outputs a valid class character.
_NON_CLASS_BIAS = {i: -1e9 for i in range(32000) if i not in CLASS_VOCAB_IDS}


def system_prompt(player: str, blinds: str = "$50/$100", stacks: str = "$10,000") -> str:
    return (
        f"You are {player}, playing 6-handed no-limit Texas Hold'em "
        f"({blinds} blinds, {stacks} starting stacks). "
        f"Make the decision {player} would make. "
        f"Reply with exactly one of: " + CLASS_TOKENS_LIST
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Poker action prediction from GGUF models (llama-cpp-python)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    g = p.add_argument_group("merged mode (one GGUF per player)")
    g.add_argument("--model",   type=Path, default=None,
                   help="Explicit path to a merged .gguf file")
    g.add_argument("--player",  default=None,
                   help="Player name; resolves to <gguf-dir>/<player>.<quant>.gguf")
    g.add_argument("--players", nargs="+", default=None,
                   help="Multiple players (loads each model in sequence)")
    g.add_argument("--gguf-dir", type=Path, default=GGUF_DIR,
                   help="Directory containing per-player merged GGUF files")
    g.add_argument("--quant",   default=DEFAULT_QUANT,
                   help="Quant suffix used when resolving player name to filename")

    g = p.add_argument_group("adapter mode (shared base + per-player adapter GGUFs)")
    g.add_argument("--base-gguf", type=Path, default=None,
                   help="Path to base GGUF (from convert_adapters_to_gguf.py)")
    g.add_argument("--lora-dir",  type=Path, default=None,
                   help="Directory containing adapter GGUFs. Defaults to "
                        "<base-gguf parent>/adapters if not set.")
    g.add_argument("--lora-scale", type=float, default=1.0,
                   help="LoRA scale / strength")

    g = p.add_argument_group("game parameters")
    g.add_argument("--blinds", default="$50/$100")
    g.add_argument("--stacks", default="$10,000")

    g = p.add_argument_group("input")
    g.add_argument("--state",       type=Path, default=None,
                   help="File containing the game state (user message). "
                        "Reads from stdin if omitted and not interactive.")
    g.add_argument("--interactive", "-i", action="store_true",
                   help="REPL: paste game state, blank line to submit")
    g.add_argument("--batch",  type=Path, default=None,
                   help="JSONL with {player, state} objects")
    g.add_argument("--output", type=Path, default=None,
                   help="Output JSONL for --batch (default: stdout)")

    g = p.add_argument_group("generation")
    g.add_argument("--max-tokens",  type=int,   default=1)
    g.add_argument("--temperature", type=float, default=0.0,
                   help="0 = greedy (deterministic)")
    g.add_argument("--top-p",       type=float, default=1.0)

    g = p.add_argument_group("runtime")
    g.add_argument("--gpu-layers", type=int, default=-1,
                   help="Layers to offload to GPU. -1=all layers (default), 0=CPU only")
    g.add_argument("--threads",    type=int, default=None,
                   help="CPU threads (default: llama.cpp auto-detect)")
    g.add_argument("--ctx",        type=int, default=512,
                   help="Context window size (our prompts are ~200 tokens)")
    g.add_argument("--verbose",    action="store_true",
                   help="Show llama.cpp loading output")

    return p.parse_args()


# ── Model loading ─────────────────────────────────────────────────────────────

def resolve_model_path(player: str, gguf_dir: Path, quant: str) -> Path:
    path = gguf_dir / f"{player}.{quant}.gguf"
    if not path.exists():
        # Fall back to any gguf with that player name
        candidates = sorted(gguf_dir.glob(f"{player}.*.gguf"))
        if candidates:
            path = candidates[0]
            print(f"  (using {path.name})", file=sys.stderr)
        else:
            print(f"ERROR: no GGUF found for {player} in {gguf_dir}", file=sys.stderr)
            print(f"  Expected: {gguf_dir}/{player}.{quant}.gguf", file=sys.stderr)
            print(f"  Run convert_to_gguf.py to generate it.", file=sys.stderr)
            sys.exit(1)
    return path


def load_model(model_path: Path, args: argparse.Namespace, lora_path: Path | None = None):
    try:
        from llama_cpp import Llama
    except ImportError:
        print("ERROR: llama-cpp-python not installed.", file=sys.stderr)
        print("  pip install llama-cpp-python", file=sys.stderr)
        print("  # or with CUDA: CMAKE_ARGS='-DGGML_CUDA=on' pip install llama-cpp-python --no-cache-dir", file=sys.stderr)
        sys.exit(1)

    kwargs: dict = {
        "model_path":   str(model_path),
        "n_ctx":        args.ctx,
        "n_gpu_layers": args.gpu_layers,
        "verbose":      args.verbose,
    }
    if args.threads is not None:
        kwargs["n_threads"] = args.threads
    if lora_path is not None:
        kwargs["lora_path"]  = str(lora_path)
        kwargs["lora_scale"] = args.lora_scale

    label = model_path.name
    if lora_path:
        label += f" + {lora_path.name}"
    print(f"Loading : {label}  (gpu_layers={args.gpu_layers})", file=sys.stderr)
    return Llama(**kwargs)


# ── Inference ─────────────────────────────────────────────────────────────────

def predict(llm, player: str, state: str, args: argparse.Namespace) -> str:
    messages = [
        {"role": "system", "content": system_prompt(player, args.blinds, args.stacks)},
        {"role": "user",   "content": state.strip()},
    ]

    kwargs: dict = {
        "messages":    messages,
        "max_tokens":  args.max_tokens,
        "temperature": args.temperature,
        "top_p":       args.top_p,
        "logit_bias":  _NON_CLASS_BIAS,
        "stream":      False,
    }
    if args.temperature == 0.0:
        kwargs["top_k"] = 1

    response = llm.create_chat_completion(**kwargs)
    char = response["choices"][0]["message"]["content"].strip()
    return char_to_display(char)   # e.g. '/' → '1/3 pot'


# ── Input helpers ─────────────────────────────────────────────────────────────

def read_state(path: Path | None) -> str:
    if path:
        return path.read_text()
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


def interactive_loop(llm, player: str, args: argparse.Namespace) -> None:
    print(f"Player: {player}  |  Ctrl-C to quit\n", file=sys.stderr)
    while True:
        try:
            state = read_state(None)
            if not state.strip():
                continue
            action = predict(llm, player, state, args)
            print(action)
            print()
        except KeyboardInterrupt:
            print("\nBye.", file=sys.stderr)
            break
        except EOFError:
            break


def run_batch(args: argparse.Namespace, players_arg: list[str]) -> None:
    # Group by player so we load each model once
    rows: list[dict] = []
    with open(args.batch) as f:
        for line in f:
            rows.append(json.loads(line))

    # Determine all players needed
    needed_players = {r["player"] for r in rows}

    # Build model-path map
    model_paths: dict[str, Path] = {}
    for p in needed_players:
        model_paths[p] = resolve_model_path(p, args.gguf_dir, args.quant)

    results: dict[int, dict] = {}

    for player, model_path in model_paths.items():
        llm = load_model(model_path, args)
        player_rows = [(i, r) for i, r in enumerate(rows) if r["player"] == player]
        for i, row in player_rows:
            action = predict(llm, player, row["state"], args)
            results[i] = {"player": player, "state": row["state"], "action": action}
        del llm  # free memory before loading next model

    out = open(args.output, "w") if args.output else sys.stdout
    try:
        for i in range(len(rows)):
            out.write(json.dumps(results[i]) + "\n")
    finally:
        if args.output:
            out.close()


# ── Main ──────────────────────────────────────────────────────────────────────

def resolve_lora_path(player: str, args: argparse.Namespace) -> Path:
    lora_dir = args.lora_dir or (args.base_gguf.parent / "adapters")
    path = lora_dir / f"{player}.gguf"
    if not path.exists():
        print(f"ERROR: adapter GGUF not found: {path}", file=sys.stderr)
        print(f"  Run convert_adapters_to_gguf.py to generate it.", file=sys.stderr)
        sys.exit(1)
    return path


def main() -> None:
    args = parse_args()

    # Resolve players list
    players: list[str] = []
    if args.players:
        players = args.players
    elif args.player:
        players = [args.player]
    elif args.model:
        players = [args.model.stem.split(".")[0]]
    elif not args.batch:
        print("ERROR: specify --player, --players, --model, or --batch", file=sys.stderr)
        sys.exit(1)

    adapter_mode = args.base_gguf is not None

    if args.batch:
        run_batch(args, players)
        return

    state = read_state(args.state) if (not args.interactive and
                                       not (not args.state and sys.stdin.isatty())) else None

    if len(players) == 1:
        player = players[0]
        if adapter_mode:
            llm = load_model(args.base_gguf, args, lora_path=resolve_lora_path(player, args))
        elif args.model:
            llm = load_model(args.model, args)
        else:
            llm = load_model(resolve_model_path(player, args.gguf_dir, args.quant), args)

        if args.interactive or (not args.state and sys.stdin.isatty()):
            interactive_loop(llm, player, args)
            return

        action = predict(llm, player, state, args)
        print(action)

    else:
        # Multiple players: load each model in sequence
        if state is None:
            state = read_state(args.state)
        for player in players:
            if adapter_mode:
                llm = load_model(args.base_gguf, args, lora_path=resolve_lora_path(player, args))
            else:
                path = resolve_model_path(player, args.gguf_dir, args.quant) \
                       if not args.model else args.model
                llm = load_model(path, args)
            action = predict(llm, player, state, args)
            print(f"{player}: {action}")
            del llm


if __name__ == "__main__":
    main()
