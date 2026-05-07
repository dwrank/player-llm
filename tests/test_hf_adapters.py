#!/usr/bin/env python3
"""
test_hf_adapters.py  —  Test HuggingFace fp16 inference with safetensors adapters.

Loads the SFT merged model (fp16) once, then swaps per-player LoRA adapters.
This is the highest-fidelity inference path (no quantization).

Expected results on the flop spot (KJo on 7s9cTc, check to act):
    All players raise; sizing varies by player style.
    MrWhite raises ~352 — confirming adapter trained correctly.

Run:
    python tests/test_hf_adapters.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
from game_states import ALL_PLAYERS, PREFLOP_FOLD, PREFLOP_FOLD_LABEL, FLOP_RAISE

BASE_MODEL   = Path("/data/models/qwen2.5-1.5b-instruct/sft/merged")
ADAPTER_ROOT = Path("/data/models/qwen2.5-1.5b-instruct/adapters")

VALID_ACTIONS = {"fold", "check", "call"}


def is_valid(action: str) -> bool:
    a = action.strip().lower()
    if a in VALID_ACTIONS:
        return True
    parts = a.split()
    return len(parts) == 2 and parts[0] == "raise" and parts[1].lstrip("$").isdigit()


def main() -> None:
    import argparse
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    # Minimal args namespace for predict()
    args = argparse.Namespace(
        blinds="$50/$100", stacks="$10,000",
        max_new_tokens=16, temperature=0.0, top_p=1.0,
    )

    # Import predict + system_prompt from inference.py
    from inference import predict, system_prompt

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
    model.eval()

    print("Loading adapters ...", flush=True)
    model = PeftModel.from_pretrained(model, str(ADAPTER_ROOT / ALL_PLAYERS[0]),
                                      adapter_name=ALL_PLAYERS[0])
    for player in ALL_PLAYERS[1:]:
        model.load_adapter(str(ADAPTER_ROOT / player), adapter_name=player)

    failures = 0

    for label, state, note in [
        (PREFLOP_FOLD_LABEL, PREFLOP_FOLD,  "preflop — all should fold"),
        (None,               FLOP_RAISE,    "flop — all should raise (sizing varies)"),
    ]:
        print(f"\n── {note} " + "─" * (50 - len(note)))
        for player in ALL_PLAYERS:
            model.set_adapter(player)
            action = predict(model, tokenizer, player, state, args)
            valid  = is_valid(action)
            match  = (action == label) if label else action.startswith("raise")

            status = "OK" if (valid and (label is None or match)) else "FAIL"
            if status == "FAIL":
                failures += 1
            note_str = f"  (expected {label})" if label and not match else ""
            print(f"  {status}  {player:<12} → {action}{note_str}")

    # MrWhite sanity check: should raise, not check
    print(f"\n── MrWhite sanity check (fp16 baseline) " + "─" * 20)
    model.set_adapter("MrWhite")
    action = predict(model, tokenizer, "MrWhite", FLOP_RAISE, args)
    expected_raise = action.startswith("raise")
    status = "OK" if expected_raise else "FAIL"
    if not expected_raise:
        failures += 1
    print(f"  {status}  MrWhite → {action}  (must be raise, not check)")

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
