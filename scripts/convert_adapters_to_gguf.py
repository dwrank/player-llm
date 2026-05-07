#!/usr/bin/env python3
"""
convert_adapters_to_gguf.py  —  Convert base model and LoRA adapters to GGUF separately.

Produces a single shared base GGUF plus a small per-player adapter GGUF for each
player. Much smaller total storage than one merged GGUF per player.

Output layout (default /data/models/qwen2.5-1.5b-instruct/gguf-adapters/):
    base.Q8_0.gguf          ~1.6 GB  (shared across all players)
    adapters/
        Pluribus.gguf        ~150 MB  (fp16 LoRA deltas)
        MrBlue.gguf
        ...

Load at inference with llama-cpp-python:
    Llama(model_path="base.Q8_0.gguf", lora_path="adapters/Pluribus.gguf")

Or with llama-cli:
    llama-cli -m base.Q8_0.gguf --lora adapters/Pluribus.gguf -p "<prompt>"

Requires llama.cpp with convert_hf_to_gguf.py and convert_lora_to_gguf.py.

Usage:
    python convert_adapters_to_gguf.py --llama-cpp ~/dev/ai/llama.cpp

    # Different base quant (Q4_K_M is smaller but less accurate with adapters)
    python convert_adapters_to_gguf.py --llama-cpp ~/dev/ai/llama.cpp --base-quant Q4_K_M

    # Specific players only
    python convert_adapters_to_gguf.py --llama-cpp ~/dev/ai/llama.cpp --players Pluribus MrWhite

    # Skip base conversion if it already exists
    python convert_adapters_to_gguf.py --llama-cpp ~/dev/ai/llama.cpp --skip-base
"""

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

SFT_MERGED   = Path("/data/models/qwen2.5-1.5b-instruct/sft/merged")
ADAPTER_ROOT = Path("/data/models/qwen2.5-1.5b-instruct/adapters")
GGUF_OUT     = Path("/data/models/qwen2.5-1.5b-instruct/gguf-adapters")

DEFAULT_PLAYERS = [
    "Pluribus", "MrBlue", "MrOrange", "Bill", "MrPink", "Eddie", "MrWhite",
]

BASE_QUANT_TYPES = ["Q4_K_M", "Q5_K_M", "Q6_K", "Q8_0", "f16"]
LORA_OUT_TYPES   = ["f16", "bf16", "f32", "q8_0"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Convert base + LoRA adapters to GGUF separately",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--llama-cpp", type=Path, required=True,
                   help="Path to llama.cpp repo")
    p.add_argument("--base",     type=Path, default=SFT_MERGED,
                   help="HuggingFace merged SFT model to use as base")
    p.add_argument("--adapters", type=Path, default=ADAPTER_ROOT)
    p.add_argument("--output",   type=Path, default=GGUF_OUT)
    p.add_argument("--players",  nargs="+", default=DEFAULT_PLAYERS)

    g = p.add_argument_group("base conversion")
    g.add_argument("--base-quant", default="Q8_0", choices=BASE_QUANT_TYPES,
                   help="Quantization for the base GGUF. Q8_0 recommended when "
                        "using with adapters (less quantization error than Q4_K_M).")
    g.add_argument("--skip-base", action="store_true",
                   help="Skip base conversion if base GGUF already exists")

    g = p.add_argument_group("adapter conversion")
    g.add_argument("--lora-outtype", default="f16", choices=LORA_OUT_TYPES,
                   help="Adapter weight dtype (adapters are small; f16 is fine)")
    g.add_argument("--skip-existing", action="store_true",
                   help="Skip adapters whose GGUF already exists")

    return p.parse_args()


def find_tools(llama_cpp: Path) -> tuple[Path, Path, Path | None]:
    """Return (convert_script, lora_script, quantize_binary)."""
    convert = llama_cpp / "convert_hf_to_gguf.py"
    if not convert.exists():
        print(f"ERROR: convert_hf_to_gguf.py not found in {llama_cpp}", file=sys.stderr)
        sys.exit(1)

    lora_convert = llama_cpp / "convert_lora_to_gguf.py"
    if not lora_convert.exists():
        print(f"ERROR: convert_lora_to_gguf.py not found in {llama_cpp}", file=sys.stderr)
        sys.exit(1)

    for name in ["llama-quantize", "build/bin/llama-quantize",
                 "build/bin/quantize", "quantize"]:
        candidate = llama_cpp / name
        if candidate.exists():
            return convert, lora_convert, candidate

    in_path = shutil.which("llama-quantize")
    return convert, lora_convert, Path(in_path) if in_path else None


def convert_base(
    converter: Path,
    quantizer: Path | None,
    base: Path,
    output: Path,
    quant: str,
) -> Path:
    output.mkdir(parents=True, exist_ok=True)
    final = output / f"base.{quant}.gguf"

    if quant == "f16":
        # Convert directly to fp16 GGUF, no quantization step needed
        cmd = [sys.executable, str(converter), str(base),
               "--outtype", "f16", "--outfile", str(final)]
        print(f"Converting base to fp16 GGUF ...")
        result = subprocess.run(cmd, check=False)
        if result.returncode != 0:
            print(f"ERROR: base conversion failed", file=sys.stderr)
            sys.exit(1)
    else:
        if quantizer is None:
            print("ERROR: llama-quantize not found — cannot quantize base", file=sys.stderr)
            sys.exit(1)
        fp16 = output / "base.fp16.gguf"
        print(f"Converting base to fp16 GGUF (intermediate) ...")
        cmd = [sys.executable, str(converter), str(base),
               "--outtype", "f16", "--outfile", str(fp16)]
        result = subprocess.run(cmd, check=False)
        if result.returncode != 0:
            print(f"ERROR: base conversion failed", file=sys.stderr)
            sys.exit(1)

        print(f"Quantizing base to {quant} ...")
        result = subprocess.run(
            [str(quantizer), str(fp16), str(final), quant], check=False
        )
        if result.returncode != 0:
            print(f"ERROR: base quantization failed", file=sys.stderr)
            sys.exit(1)
        fp16.unlink()
        print(f"Removed fp16 intermediate")

    size_gb = final.stat().st_size / 1e9
    print(f"Base GGUF → {final}  ({size_gb:.2f} GB)\n")
    return final


def convert_adapter(
    lora_converter: Path,
    base: Path,
    adapter_dir: Path,
    out_path: Path,
    outtype: str,
) -> None:
    cmd = [
        sys.executable, str(lora_converter),
        "--base",    str(base),
        "--outfile", str(out_path),
        "--outtype", outtype,
        str(adapter_dir),
    ]
    result = subprocess.run(cmd, check=False, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  ERROR: adapter conversion failed", file=sys.stderr)
        print(result.stderr[-500:] if result.stderr else "", file=sys.stderr)
        return

    size_mb = out_path.stat().st_size / 1e6
    print(f"  → {out_path.name}  ({size_mb:.0f} MB)")


def main() -> None:
    args = parse_args()

    if not args.llama_cpp.exists():
        print(f"ERROR: llama.cpp not found at {args.llama_cpp}", file=sys.stderr)
        sys.exit(1)
    if not args.base.exists():
        print(f"ERROR: base model not found: {args.base}", file=sys.stderr)
        sys.exit(1)

    converter, lora_converter, quantizer = find_tools(args.llama_cpp)

    print(f"Base      : {args.base}")
    print(f"Base quant: {args.base_quant}")
    print(f"Lora type : {args.lora_outtype}")
    print(f"Output    : {args.output}")
    print(f"Players   : {', '.join(args.players)}\n")

    # ── Convert base ──────────────────────────────────────────────────────────
    base_gguf = args.output / f"base.{args.base_quant}.gguf"
    if args.skip_base and base_gguf.exists():
        size_gb = base_gguf.stat().st_size / 1e9
        print(f"Base GGUF exists, skipping  ({size_gb:.2f} GB)\n")
    else:
        convert_base(converter, quantizer, args.base, args.output, args.base_quant)

    # ── Convert adapters ──────────────────────────────────────────────────────
    adapter_out = args.output / "adapters"
    adapter_out.mkdir(parents=True, exist_ok=True)

    for i, player in enumerate(args.players, 1):
        out_path    = adapter_out / f"{player}.gguf"
        adapter_dir = args.adapters / player

        if args.skip_existing and out_path.exists():
            print(f"[{i}/{len(args.players)}] {player} — skipping (exists)")
            continue

        if not (adapter_dir / "adapter_config.json").exists():
            print(f"[{i}/{len(args.players)}] {player} — WARNING: adapter not found, skipping")
            continue

        print(f"[{i}/{len(args.players)}] {player}")
        convert_adapter(lora_converter, args.base, adapter_dir, out_path, args.lora_outtype)

    print(f"\nDone.")
    print(f"Load with llama-cpp-python:")
    print(f"  from llama_cpp import Llama")
    print(f"  llm = Llama(")
    print(f"      model_path='{args.output}/base.{args.base_quant}.gguf',")
    print(f"      lora_path='{adapter_out}/Pluribus.gguf',")
    print(f"  )")
    print(f"\nOr with llama-cli:")
    print(f"  llama-cli -m {args.output}/base.{args.base_quant}.gguf \\")
    print(f"            --lora {adapter_out}/Pluribus.gguf -p '<prompt>'")


if __name__ == "__main__":
    main()
