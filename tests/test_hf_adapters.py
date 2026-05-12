#!/usr/bin/env python3
"""
test_hf_adapters.py  —  Test HuggingFace fp16 inference with safetensors adapters.

Loads the SFT merged model (fp16) once, then swaps per-player LoRA adapters.
This is the highest-fidelity inference path (no quantization).

Expected results:
    Preflop spot: all players fold.
    Flop spot (KJo on 7s9cTc): all players raise (class varies by player style).

Run:
    python tests/test_hf_adapters.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
from action_classes import CLASSES
from game_states import ALL_PLAYERS, PREFLOP_FOLD, PREFLOP_FOLD_LABEL, FLOP_RAISE

BASE_MODEL   = Path("/data/models/qwen2.5-1.5b-instruct/sft/merged")
ADAPTER_ROOT = Path("/data/models/qwen2.5-1.5b-instruct/adapters")

_RAISE_LABELS = set(CLASSES[3:])


def is_valid(action: str) -> bool:
    return action.strip() in set(CLASSES)


def is_raise(action: str) -> bool:
    return action.strip() in _RAISE_LABELS


def main() -> None:
    import argparse
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    from inference import predict, system_prompt, get_class_token_ids

    args = argparse.Namespace(
        blinds="$50/$100", stacks="$10,000",
        temperature=0.0, top_p=1.0,
    )

    print("=" * 60)
    print("HuggingFace fp16 inference — safetensors adapters")
    print("=" * 60)
    print(f"Base  : {BASE_MODEL}")
    print(f"Dtype : fp16  (no quantization)")
    print()

    print("Loading base model ...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        str(BASE_MODEL), dtype=torch.float16, device_map="auto",
        attn_implementation="sdpa",
    )
    tokenizer = AutoTokenizer.from_pretrained(str(BASE_MODEL))
    class_token_ids = get_class_token_ids(tokenizer)
    model.eval()

    print("Loading adapters ...", flush=True)
    model = PeftModel.from_pretrained(model, str(ADAPTER_ROOT / ALL_PLAYERS[0]),
                                      adapter_name=ALL_PLAYERS[0])
    for player in ALL_PLAYERS[1:]:
        model.load_adapter(str(ADAPTER_ROOT / player), adapter_name=player)

    failures = 0

    for expected_label, state, note in [
        (PREFLOP_FOLD_LABEL, PREFLOP_FOLD, "preflop — all should fold"),
        (None,               FLOP_RAISE,   "flop — all should raise (class varies)"),
    ]:
        print(f"\n── {note} " + "─" * (50 - len(note)))
        for player in ALL_PLAYERS:
            model.set_adapter(player)
            action = predict(model, tokenizer, player, state, args, class_token_ids)
            valid  = is_valid(action)
            ok     = (action == expected_label) if expected_label else is_raise(action)

            status = "OK" if (valid and ok) else "FAIL"
            if status == "FAIL":
                failures += 1
            suffix = f"  (expected {expected_label})" if expected_label and not ok else ""
            print(f"  {status}  {player:<12} → {action}{suffix}")

    # MrWhite sanity check: should raise, not check
    print(f"\n── MrWhite sanity check (fp16 baseline) " + "─" * 20)
    model.set_adapter("MrWhite")
    action = predict(model, tokenizer, "MrWhite", FLOP_RAISE, args, class_token_ids)
    ok = is_raise(action)
    status = "OK" if ok else "FAIL"
    if not ok:
        failures += 1
    print(f"  {status}  MrWhite → {action}  (must be a raise class, not check)")

    print()
    print("=" * 60)
    if failures == 0:
        print("ALL TESTS PASSED")
    else:
        print(f"FAILURES: {failures}")
    print("=" * 60)
    sys.exit(0 if failures == 0 else 1)


if __name__ == "__main__":
    main()
