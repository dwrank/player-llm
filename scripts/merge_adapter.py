#!/usr/bin/env python3
"""
merge_adapter.py  —  Merge a LoRA adapter into a base model in full precision.

Loads the base model in fp16 (no quantization), applies the adapter, merges
the LoRA weights, and saves a clean fp16 model that loads without bitsandbytes.

Usage:
    # Merge Stage 1 adapter
    python merge_adapter.py

    # Merge a specific player adapter onto the SFT merged model
    python merge_adapter.py \\
        --base   /data/models/qwen2.5-1.5b-instruct/sft/merged \\
        --adapter /data/models/qwen2.5-1.5b-instruct/adapters/Pluribus \\
        --output  /data/models/qwen2.5-1.5b-instruct/adapters/Pluribus/merged

    # Merge but keep bf16 instead of fp16
    python merge_adapter.py --dtype bf16
"""

import argparse
import sys
from pathlib import Path

BASE_MODEL   = Path("/data/models/qwen2.5-1.5b-instruct/base")
SFT_ADAPTER  = Path("/data/models/qwen2.5-1.5b-instruct/sft/adapter")
SFT_MERGED   = Path("/data/models/qwen2.5-1.5b-instruct/sft/merged")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Merge LoRA adapter into base model weights (full precision)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--base",    type=Path, default=BASE_MODEL,
                   help="Base model to load in full precision")
    p.add_argument("--adapter", type=Path, default=SFT_ADAPTER,
                   help="LoRA adapter directory (contains adapter_config.json)")
    p.add_argument("--output",  type=Path, default=SFT_MERGED,
                   help="Where to save the merged model")
    p.add_argument("--dtype",   choices=["fp16", "bf16"], default="fp16",
                   help="Output weight dtype")
    p.add_argument("--no-verify", action="store_true",
                   help="Skip post-merge tensor verification")
    return p.parse_args()


def verify(output: Path) -> None:
    from safetensors import safe_open
    import os

    files = sorted(output.glob("*.safetensors"))
    if not files:
        print("ERROR: no .safetensors files found in output", file=sys.stderr)
        sys.exit(1)

    total_params = 0
    dtypes = set()
    for f in files:
        with safe_open(f, framework="pt") as sf:
            keys = list(sf.keys())
            for k in keys:
                t = sf.get_tensor(k)
                total_params += t.numel()
                dtypes.add(str(t.dtype))

    print(f"\nVerification:")
    print(f"  Files  : {len(files)} shard(s)")
    print(f"  Tensors: {sum(1 for f in files for _ in [None])}")   # placeholder
    print(f"  Params : {total_params/1e9:.3f}B")
    print(f"  Dtypes : {dtypes}")

    if len(dtypes) > 1:
        print("  WARNING: mixed dtypes — merge may not be clean")
    elif "torch.uint8" in dtypes:
        print("  WARNING: uint8 present — base was loaded quantized, not full precision")
    else:
        print("  OK: single dtype, no quantization artifacts")


def main() -> None:
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    args = parse_args()

    dtype = torch.float16 if args.dtype == "fp16" else torch.bfloat16
    dtype_name = "fp16" if args.dtype == "fp16" else "bf16"

    # Validate inputs
    if not args.base.exists():
        print(f"ERROR: base model not found: {args.base}", file=sys.stderr)
        sys.exit(1)
    if not (args.adapter / "adapter_config.json").exists():
        print(f"ERROR: adapter not found: {args.adapter}", file=sys.stderr)
        sys.exit(1)

    print(f"Base    : {args.base}")
    print(f"Adapter : {args.adapter}")
    print(f"Output  : {args.output}")
    print(f"Dtype   : {dtype_name}\n")

    # Load base in full precision — no BitsAndBytesConfig
    print(f"Loading base model in {dtype_name} (no quantization) ...")
    model = AutoModelForCausalLM.from_pretrained(
        str(args.base),
        dtype=dtype,
        device_map="auto",
        attn_implementation="sdpa",
    )

    params_gb = sum(p.numel() * p.element_size() for p in model.parameters()) / 1e9
    print(f"  Loaded : {sum(p.numel() for p in model.parameters())/1e9:.3f}B params  "
          f"({params_gb:.2f} GB in {dtype_name})")

    # Apply and merge adapter
    print(f"\nApplying adapter ...")
    model = PeftModel.from_pretrained(model, str(args.adapter))

    print("Merging LoRA weights into base ...")
    model = model.merge_and_unload()

    # Confirm all weights are the expected dtype
    dtypes_after = {str(p.dtype) for p in model.parameters()}
    print(f"  Dtypes after merge: {dtypes_after}")
    if dtypes_after != {f"torch.{dtype_name.replace('16', '16')}"}:
        # normalize check
        expected = f"torch.{'float16' if args.dtype == 'fp16' else 'bfloat16'}"
        unexpected = dtypes_after - {expected}
        if unexpected:
            print(f"  WARNING: unexpected dtypes {unexpected}")

    # Save
    print(f"\nSaving merged model → {args.output} ...")
    args.output.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(args.output), safe_serialization=True)

    # Save tokenizer from adapter dir (has the updated pad token config)
    tokenizer = AutoTokenizer.from_pretrained(str(args.adapter))
    tokenizer.save_pretrained(str(args.output))

    size_gb = sum(f.stat().st_size for f in args.output.glob("*.safetensors")) / 1e9
    print(f"  Size   : {size_gb:.2f} GB")

    if not args.no_verify:
        verify(args.output)

    print(f"\nDone. Load with:")
    print(f"  AutoModelForCausalLM.from_pretrained('{args.output}')")
    print(f"  # No BitsAndBytesConfig needed — weights are plain {dtype_name}")


if __name__ == "__main__":
    main()
