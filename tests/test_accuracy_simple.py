#!/usr/bin/env python3
"""
test_accuracy.py  —  Compare inference variants against ground truth player actions.

Uses the exact validation split from 02_lora_per_player.py (seed=42, eval_split=0.1).
Samples --n-samples examples per player.

Scoring:
  exact    = exact string match
  weighted = exact for fold/check/call; raise scored by 1 - |pred-amt - truth-amt| / truth-amt

Available variants:
  hf        HuggingFace fp16 base + safetensors adapters
  gguf-f16  GGUF fp16 base + fp16 LoRA adapter
  gguf-q8   GGUF Q8_0 base + fp16 LoRA adapter
  gguf-q4   GGUF Q4_K_M base + fp16 LoRA adapter

Run:
    python tests/test_accuracy.py                          # all variants
    python tests/test_accuracy.py --variants hf            # HF only
    python tests/test_accuracy.py --variants gguf-f16 gguf-q8
    python tests/test_accuracy.py --variants gguf-q4 --n-samples 200
"""

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

DATA_PATH        = Path(__file__).parent.parent / "data" / "sft_pluribus.jsonl"
BASE_MODEL       = Path("/data/models/qwen2.5-1.5b-instruct/sft/merged")
ADAPTER_ROOT     = Path("/data/models/qwen2.5-1.5b-instruct/adapters")
GGUF_ADAPTER_DIR = Path("/data/models/qwen2.5-1.5b-instruct/gguf-adapters")
GGUF_ADAPTERS    = GGUF_ADAPTER_DIR / "adapters"

TRAINED_PLAYERS = ["Pluribus", "MrBlue", "MrOrange", "Bill", "MrPink", "Eddie", "MrWhite"]

EVAL_SPLIT  = 0.1
SEED        = 42
TOLERANCES  = [0.05, 0.10, 0.25]
RANK_ORDER  = "23456789TJQKA"

# Variant registry: name → (display label, base gguf or None for HF)
VARIANT_DEFS: dict[str, tuple[str, Path | None]] = {
    "hf":       ("HF fp16",        None),
    "gguf-f16": ("GGUF f16+lora",  GGUF_ADAPTER_DIR / "base.f16.gguf"),
    "gguf-q8":  ("GGUF Q8+lora",   GGUF_ADAPTER_DIR / "base.Q8_0.gguf"),
    "gguf-q4":  ("GGUF Q4+lora",   GGUF_ADAPTER_DIR / "base.Q4_K_M.gguf"),
}
ALL_VARIANTS = list(VARIANT_DEFS)


# ── Dataset ───────────────────────────────────────────────────────────────────

def load_eval_split(players: list[str]) -> dict[str, list[dict]]:
    """Reproduce the exact validation split from 02_lora_per_player.py."""
    by_player: dict[str, list] = {}
    with open(DATA_PATH) as f:
        for line in f:
            ex = json.loads(line)
            p = ex["player"]
            if p in players:
                by_player.setdefault(p, []).append(ex)

    eval_by_player: dict[str, list] = {}
    for player, examples in by_player.items():
        rng = random.Random(SEED)
        shuffled = list(examples)
        rng.shuffle(shuffled)
        n_eval = max(1, int(len(shuffled) * EVAL_SPLIT))
        eval_by_player[player] = shuffled[:n_eval]

    return eval_by_player


def sample_examples(eval_by_player: dict[str, list], n: int, seed: int) -> dict[str, list]:
    sampled = {}
    rng = random.Random(seed)
    for player, examples in eval_by_player.items():
        k = min(n, len(examples))
        sampled[player] = rng.sample(examples, k)
    return sampled


# ── Scoring ───────────────────────────────────────────────────────────────────

def action_type(action: str) -> str:
    a = action.strip().lower()
    return a.split()[0] if a else ""


def parse_raise_amount(action: str) -> int | None:
    parts = action.strip().lower().split()
    if len(parts) == 2 and parts[0] == "raise":
        try:
            return int(parts[1].lstrip("$"))
        except ValueError:
            pass
    return None


def score(pred: str, truth: str) -> float:
    """
    Score in [0, 1]. Non-raise: exact match. Raise: linear decay by relative error
    when both are raises, 0.0 when the predicted action type differs.
    """
    t_amt = parse_raise_amount(truth)
    if t_amt is None:
        return 1.0 if pred == truth else 0.0
    p_amt = parse_raise_amount(pred)
    if p_amt is None:
        return 0.0
    return max(0.0, 1.0 - abs(p_amt - t_amt) / t_amt)


def _print_progress(label: str, player: str, pairs: list[tuple[str, str]]) -> None:
    n = len(pairs)
    exact = sum(p == t for p, t in pairs)
    wtd   = sum(score(p, t) for p, t in pairs)
    print(f"  {label:<14} {player:<12} "
          f"exact {exact:3}/{n} ({100*exact/n:.1f}%)  "
          f"weighted {wtd:.1f}/{n} ({100*wtd/n:.1f}%)", flush=True)


# ── Board texture ─────────────────────────────────────────────────────────────

def parse_board(ex: dict) -> list[str]:
    for line in ex["messages"][1]["content"].split("\n"):
        if line.startswith("Board: "):
            val = line[7:].strip()
            return [] if val == "none" else val.split()
    return []


def board_meta(ex: dict) -> dict:
    board  = parse_board(ex)
    street = ex.get("street", "")

    if not board:
        return {"street": street, "postflop": False}

    ranks = [c[:-1] for c in board]
    suits = [c[-1]  for c in board]
    max_suited = max(Counter(suits).values())

    paired = len(ranks) != len(set(ranks))

    if max_suited >= 3:
        flush_tex = "flush_possible"
    elif max_suited == 2:
        flush_tex = "two_tone"
    else:
        flush_tex = "rainbow"

    nums = sorted(set(RANK_ORDER.index(r) for r in ranks if r in RANK_ORDER))
    connected = (
        any(nums[i + 1] - nums[i] <= 3 for i in range(len(nums) - 1))
        if len(nums) >= 2 else False
    )

    return {
        "street":    street,
        "postflop":  True,
        "paired":    paired,
        "flush_tex": flush_tex,   # "rainbow" | "two_tone" | "flush_possible"
        "connected": connected,
    }


def build_metadata(samples: dict[str, list]) -> dict[str, list[dict]]:
    return {player: [board_meta(ex) for ex in exs] for player, exs in samples.items()}


# ── Inference ─────────────────────────────────────────────────────────────────

def run_hf(samples: dict[str, list], label: str) -> dict[str, list[tuple[str, str]]]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    from inference import predict

    inf_args = argparse.Namespace(
        blinds="$50/$100", stacks="$10,000",
        max_new_tokens=16, temperature=0.0, top_p=1.0,
    )

    print("Loading HF base model ...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        str(BASE_MODEL), torch_dtype=torch.float16,
        device_map="auto", attn_implementation="sdpa",
    )
    tokenizer = AutoTokenizer.from_pretrained(str(BASE_MODEL))
    model.eval()

    players = list(samples.keys())
    print(f"Loading adapters for {len(players)} players ...", flush=True)
    model = PeftModel.from_pretrained(model, str(ADAPTER_ROOT / players[0]),
                                      adapter_name=players[0])
    for player in players[1:]:
        model.load_adapter(str(ADAPTER_ROOT / player), adapter_name=player)

    results: dict[str, list[tuple[str, str]]] = {}
    for player, examples in samples.items():
        model.set_adapter(player)
        pairs = []
        for ex in examples:
            state = ex["messages"][1]["content"]
            truth = ex["messages"][2]["content"].strip().lower()
            pred  = predict(model, tokenizer, player, state, inf_args).strip().lower()
            pairs.append((pred, truth))
        results[player] = pairs
        _print_progress(label, player, pairs)

    del model
    return results


def run_gguf(samples: dict[str, list], base_path: Path,
             label: str) -> dict[str, list[tuple[str, str]]]:
    from llama_cpp import Llama
    from inference_gguf import predict

    if not base_path.exists():
        print(f"  SKIP {label} — {base_path.name} not found", flush=True)
        return {}

    inf_args = argparse.Namespace(
        blinds="$50/$100", stacks="$10,000",
        max_tokens=16, temperature=0.0, top_p=1.0,
        gpu_layers=-1, threads=None, ctx=512, verbose=False,
        lora_scale=1.0,
    )

    results: dict[str, list[tuple[str, str]]] = {}
    for player, examples in samples.items():
        lora_path = GGUF_ADAPTERS / f"{player}.gguf"
        if not lora_path.exists():
            print(f"  {label:<14} {player:<12} SKIP (adapter not found)", flush=True)
            continue

        llm = Llama(model_path=str(base_path), lora_path=str(lora_path),
                    lora_scale=1.0, n_ctx=512, n_gpu_layers=-1, verbose=False)
        pairs = []
        for ex in examples:
            state = ex["messages"][1]["content"]
            truth = ex["messages"][2]["content"].strip().lower()
            pred  = predict(llm, player, state, inf_args).strip().lower()
            pairs.append((pred, truth))
        llm.close()
        del llm

        results[player] = pairs
        _print_progress(label, player, pairs)

    return results


# ── Reporting ─────────────────────────────────────────────────────────────────

def report(all_results: dict[str, dict[str, list[tuple[str, str]]]],
           samples: dict[str, list],
           metadata: dict[str, list[dict]]) -> None:
    players  = list(samples.keys())
    variants = list(all_results.keys())   # display labels in run order
    total_n  = sum(len(v) for v in samples.values())
    col_w    = 12
    vcol_w   = 19   # "exact 75.3%  wtd 84.1%"

    sep = "=" * (col_w + len(variants) * (vcol_w + 2))

    print()
    print(sep)
    print("Accuracy vs ground truth")
    print("  exact    = exact string match")
    print("  weighted = exact for fold/check/call; raise scored by 1-|pred-truth|/truth")
    print(sep)

    # Header
    header = f"{'Player':<{col_w}}"
    for vlabel in variants:
        header += f"  {vlabel:^{vcol_w}}"
    print(header)

    subhdr = f"{'':^{col_w}}"
    for _ in variants:
        subhdr += f"  {'exact':>8}  {'weighted':>8}"
    print(subhdr)
    print("─" * len(sep))

    totals: dict[str, list[float, float]] = {v: [0.0, 0.0] for v in variants}

    for player in players:
        line = f"{player:<{col_w}}"
        n = len(samples[player])
        for vlabel in variants:
            pairs = all_results[vlabel].get(player)
            if pairs:
                ex  = sum(p == t for p, t in pairs)
                wtd = sum(score(p, t) for p, t in pairs)
                totals[vlabel][0] += ex
                totals[vlabel][1] += wtd
                line += f"  {100*ex/n:6.1f}%  {100*wtd/n:6.1f}%"
            else:
                line += f"  {'—':>8}  {'—':>8}"
        print(line)

    print("─" * len(sep))
    tline = f"{'TOTAL':<{col_w}}"
    for vlabel in variants:
        ex, wtd = totals[vlabel]
        tline += f"  {100*ex/total_n:6.1f}%  {100*wtd/total_n:6.1f}%"
    print(tline)

    # ── Action-type breakdown ─────────────────────────────────────────────────
    print()
    print("Exact match by action type")
    print("─" * len(sep))

    hdr = f"  {'action':<8}"
    for vlabel in variants:
        hdr += f"  {vlabel:^16}"
    print(hdr)

    for atype in ["fold", "call", "check", "raise"]:
        line = f"  {atype:<8}"
        for vlabel in variants:
            c = t = 0
            for player in players:
                for pred, truth in all_results[vlabel].get(player, []):
                    if action_type(truth) == atype:
                        t += 1
                        if pred == truth:
                            c += 1
            line += f"  {c:3}/{t} ({100*c/t:5.1f}%)" if t else f"  {'—':^16}"
        print(line)

    # ── Raise sizing analysis ─────────────────────────────────────────────────
    print()
    print("Raise sizing  (ground truth = raise)")
    print("─" * len(sep))

    for vlabel in variants:
        raise_pairs: list[tuple[int | None, int]] = []
        for player in players:
            for pred, truth in all_results[vlabel].get(player, []):
                t_amt = parse_raise_amount(truth)
                if t_amt is None:
                    continue
                raise_pairs.append((parse_raise_amount(pred), t_amt))

        if not raise_pairs:
            continue

        n_raise      = len(raise_pairs)
        n_type_right = sum(p is not None for p, _ in raise_pairs)
        both         = [(p, t) for p, t in raise_pairs if p is not None]

        print(f"  {vlabel}  n={n_raise}  "
              f"predicted raise: {n_type_right}/{n_raise} ({100*n_type_right/n_raise:.1f}%)")

        if both:
            rel_errs = [abs(p - t) / t for p, t in both]
            abs_errs = [abs(p - t)     for p, t in both]
            tols = "  ".join(
                f"±{int(tol*100)}%: {sum(e<=tol for e in rel_errs)}/{len(both)}"
                f" ({100*sum(e<=tol for e in rel_errs)/len(both):.0f}%)"
                for tol in TOLERANCES
            )
            print(f"        when both raise  n={len(both)}  "
                  f"mean rel err: {sum(rel_errs)/len(rel_errs):.1%}  "
                  f"mean abs: ${sum(abs_errs)/len(abs_errs):.0f}  "
                  f"median abs: ${sorted(abs_errs)[len(abs_errs)//2]:.0f}")
            print(f"        {tols}")
        print()

    # ── Street breakdown ──────────────────────────────────────────────────────
    print()
    print("Accuracy by street")
    print("─" * len(sep))

    hdr = f"  {'street':<10}"
    for vlabel in variants:
        hdr += f"  {vlabel:^{vcol_w}}"
    print(hdr)
    subhdr = f"  {'':^10}"
    for _ in variants:
        subhdr += f"  {'exact':>8}  {'weighted':>8}"
    print(subhdr)
    print("─" * len(sep))

    for street in ["Preflop", "Flop", "Turn", "River"]:
        line = f"  {street:<10}"
        for vlabel in variants:
            c = w = n = 0
            for player in players:
                for i, (pred, truth) in enumerate(all_results[vlabel].get(player, [])):
                    m = metadata.get(player, [{}] * (i + 1))
                    if i < len(m) and m[i].get("street") == street:
                        c += pred == truth
                        w += score(pred, truth)
                        n += 1
            line += (f"  {100*c/n:6.1f}%  {100*w/n:6.1f}%" if n
                     else f"  {'—':>8}  {'—':>8}")
        print(line)

    # ── Board texture breakdown ───────────────────────────────────────────────
    print()
    print("Accuracy by board texture  (flop / turn / river)")
    print("─" * len(sep))

    hdr = f"  {'texture':<16}"
    for vlabel in variants:
        hdr += f"  {vlabel:^{vcol_w}}"
    print(hdr)
    subhdr = f"  {'':^16}"
    for _ in variants:
        subhdr += f"  {'exact':>8}  {'weighted':>8}"
    print(subhdr)
    print("─" * len(sep))

    texture_groups = [
        ("unpaired",       lambda m: m["postflop"] and not m["paired"]),
        ("paired",         lambda m: m["postflop"] and m["paired"]),
        ("rainbow",        lambda m: m["postflop"] and m["flush_tex"] == "rainbow"),
        ("two-tone",       lambda m: m["postflop"] and m["flush_tex"] == "two_tone"),
        ("flush possible", lambda m: m["postflop"] and m["flush_tex"] == "flush_possible"),
        ("unconnected",    lambda m: m["postflop"] and not m["connected"]),
        ("connected",      lambda m: m["postflop"] and m["connected"]),
    ]

    for label, pred_fn in texture_groups:
        line = f"  {label:<16}"
        for vlabel in variants:
            c = w = n = 0
            for player in players:
                metas = metadata.get(player, [])
                for i, (pred, truth) in enumerate(all_results[vlabel].get(player, [])):
                    if i < len(metas) and pred_fn(metas[i]):
                        c += pred == truth
                        w += score(pred, truth)
                        n += 1
            line += (f"  {100*c/n:6.1f}%  {100*w/n:6.1f}%" if n
                     else f"  {'—':>8}  {'—':>8}")
        print(line)

    print(sep)


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Compare inference variants against ground truth",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--variants", nargs="+", default=ALL_VARIANTS,
                   choices=ALL_VARIANTS, metavar="VARIANT",
                   help=f"Variants to run: {', '.join(ALL_VARIANTS)}")
    p.add_argument("--n-samples",   type=int, default=100,
                   help="Eval examples per player")
    p.add_argument("--sample-seed", type=int, default=0,
                   help="Seed for sampling from the eval split")
    p.add_argument("--players", nargs="+", default=TRAINED_PLAYERS)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    print(f"Variants : {', '.join(args.variants)}")
    print(f"Loading eval split (seed={SEED}, split={EVAL_SPLIT}) ...", flush=True)
    eval_by_player = load_eval_split(args.players)
    samples = sample_examples(eval_by_player, args.n_samples, args.sample_seed)

    total = sum(len(v) for v in samples.values())
    print(f"Sampled {total} examples across {len(samples)} players "
          f"({args.n_samples} per player max)\n")

    all_results: dict[str, dict[str, list[tuple[str, str]]]] = {}

    for variant in args.variants:
        vlabel, base_path = VARIANT_DEFS[variant]
        print(f"── {vlabel} " + "─" * max(0, 60 - len(vlabel)))
        if base_path is None:
            all_results[vlabel] = run_hf(samples, vlabel)
        else:
            all_results[vlabel] = run_gguf(samples, base_path, vlabel)
        print()

    metadata = build_metadata(samples)
    report(all_results, samples, metadata)


if __name__ == "__main__":
    main()
