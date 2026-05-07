#!/usr/bin/env python3
"""
convert_to_gguf.py  —  Convert player models to GGUF for llama.cpp / CPU inference.

For each player:
  1. Merges the LoRA adapter into the fp16 SFT base (same as merge_adapter.py)
  2. Converts the merged model to GGUF (fp16 intermediate)
  3. Quantizes to the requested quant type (default: Q4_K_M)

Requires llama.cpp:
    git clone https://github.com/ggerganov/llama.cpp /opt/llama.cpp
    cd /opt/llama.cpp && pip install -r requirements.txt
    cmake -B build && cmake --build build --config Release -j$(nproc)

Output (default /data/models/gguf/):
    Pluribus.Q4_K_M.gguf
    MrBlue.Q4_K_M.gguf
    ...

Usage:
    # All default players, Q4_K_M
    python convert_to_gguf.py --llama-cpp /opt/llama.cpp

    # Specific players, different quant
    python convert_to_gguf.py --llama-cpp /opt/llama.cpp --players Pluribus MrBlue --quant Q8_0

    # Skip conversion, only quantize already-converted fp16 GGUFs
    python convert_to_gguf.py --llama-cpp /opt/llama.cpp --quant-only

    # Skip quantization, produce fp16 GGUFs only
    python convert_to_gguf.py --llama-cpp /opt/llama.cpp --quant none
"""

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

SFT_MERGED   = Path("/data/models/qwen2.5-1.5b-instruct/sft/merged")
ADAPTER_ROOT = Path("/data/models/qwen2.5-1.5b-instruct/adapters")
GGUF_OUT     = Path("/data/models/qwen2.5-1.5b-instruct/gguf")

DEFAULT_PLAYERS = [
    "Pluribus", "MrBlue", "MrOrange", "Bill", "MrPink", "Eddie", "MrWhite",
]

QUANT_TYPES = ["Q4_K_M", "Q4_K_S", "Q5_K_M", "Q6_K", "Q8_0", "none"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Convert player LoRA adapters to GGUF for llama.cpp",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--llama-cpp", type=Path, required=True,
                   help="Path to llama.cpp repo (must be built)")
    p.add_argument("--base",     type=Path, default=SFT_MERGED,
                   help="fp16 merged SFT base model")
    p.add_argument("--adapters", type=Path, default=ADAPTER_ROOT)
    p.add_argument("--output",   type=Path, default=GGUF_OUT,
                   help="Directory for output GGUF files")
    p.add_argument("--players",  nargs="+", default=DEFAULT_PLAYERS)
    p.add_argument("--quant",    default="Q4_K_M", choices=QUANT_TYPES,
                   help="Quantization type. 'none' = fp16 GGUF only.")
    p.add_argument("--quant-only", action="store_true",
                   help="Skip merge+convert; only (re)quantize existing fp16 GGUFs")
    p.add_argument("--keep-fp16", action="store_true",
                   help="Keep the fp16 GGUF after quantization")
    p.add_argument("--skip-existing", action="store_true",
                   help="Skip players whose final GGUF already exists")
    return p.parse_args()


def find_tools(llama_cpp: Path) -> tuple[Path, Path | None]:
    """Return (convert_script, quantize_binary). quantize_binary may be None."""
    # Converter: convert_hf_to_gguf.py (newer) or convert.py (older)
    for name in ["convert_hf_to_gguf.py", "convert.py"]:
        candidate = llama_cpp / name
        if candidate.exists():
            converter = candidate
            break
    else:
        print(f"ERROR: no convert script found in {llama_cpp}", file=sys.stderr)
        print(f"  Expected convert_hf_to_gguf.py or convert.py", file=sys.stderr)
        sys.exit(1)

    # Quantizer: check relative paths first, then PATH
    for name in ["llama-quantize", "build/bin/llama-quantize",
                 "build/bin/quantize", "quantize"]:
        candidate = llama_cpp / name
        if candidate.exists():
            return converter, candidate

    in_path = shutil.which("llama-quantize")
    if in_path:
        return converter, Path(in_path)

    return converter, None


def merge_adapter(base: Path, adapter: Path, tmp_dir: Path) -> None:
    """Merge a LoRA adapter into the fp16 base and save to tmp_dir."""
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"  Merging adapter {adapter.name} into fp16 base ...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        str(base),
        dtype=torch.float16,
        device_map="auto",
        attn_implementation="sdpa",
    )
    model = PeftModel.from_pretrained(model, str(adapter))
    model = model.merge_and_unload()
    model.save_pretrained(str(tmp_dir), safe_serialization=True)

    tokenizer = AutoTokenizer.from_pretrained(str(adapter))
    tokenizer.save_pretrained(str(tmp_dir))
    print(f"  Merged → {tmp_dir}", flush=True)


def convert_to_fp16_gguf(converter: Path, model_dir: Path, gguf_path: Path) -> None:
    cmd = [
        sys.executable, str(converter),
        str(model_dir),
        "--outtype", "f16",
        "--outfile", str(gguf_path),
    ]
    print(f"  Converting to GGUF fp16 ...", flush=True)
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        print(f"  ERROR: conversion failed (exit {result.returncode})", file=sys.stderr)
        sys.exit(1)
    size_gb = gguf_path.stat().st_size / 1e9
    print(f"  GGUF fp16 → {gguf_path}  ({size_gb:.2f} GB)", flush=True)


def quantize_gguf(quantizer: Path, fp16_gguf: Path, out_gguf: Path, quant: str) -> None:
    cmd = [str(quantizer), str(fp16_gguf), str(out_gguf), quant]
    print(f"  Quantizing {quant} ...", flush=True)
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        print(f"  ERROR: quantization failed (exit {result.returncode})", file=sys.stderr)
        sys.exit(1)
    size_gb = out_gguf.stat().st_size / 1e9
    print(f"  {quant} GGUF → {out_gguf}  ({size_gb:.2f} GB)", flush=True)


def process_player(
    player: str,
    args: argparse.Namespace,
    converter: Path,
    quantizer: Path | None,
) -> None:
    args.output.mkdir(parents=True, exist_ok=True)

    fp16_gguf  = args.output / f"{player}.fp16.gguf"
    final_gguf = args.output / f"{player}.{args.quant}.gguf" if args.quant != "none" else fp16_gguf

    if args.skip_existing and final_gguf.exists():
        print(f"  Skipping {player} — {final_gguf.name} exists")
        return

    if not args.quant_only:
        adapter_dir = args.adapters / player
        if not (adapter_dir / "adapter_config.json").exists():
            print(f"  WARNING: adapter not found for {player}, skipping", file=sys.stderr)
            return

        with tempfile.TemporaryDirectory(prefix=f"gguf_{player}_") as tmp:
            tmp_path = Path(tmp)
            merge_adapter(args.base, adapter_dir, tmp_path)
            convert_to_fp16_gguf(converter, tmp_path, fp16_gguf)
            # tmp dir (merged HF model) cleaned up automatically

    elif not fp16_gguf.exists():
        print(f"  ERROR: --quant-only set but {fp16_gguf} not found", file=sys.stderr)
        return

    if args.quant != "none":
        if quantizer is None:
            print(f"  WARNING: llama-quantize binary not found — skipping quantization", file=sys.stderr)
            print(f"  Build llama.cpp first: cmake -B build && cmake --build build -j$(nproc)", file=sys.stderr)
            return
        quantize_gguf(quantizer, fp16_gguf, final_gguf, args.quant)
        if not args.keep_fp16 and fp16_gguf != final_gguf:
            fp16_gguf.unlink()
            print(f"  Removed fp16 intermediate", flush=True)


def main() -> None:
    args = parse_args()

    if not args.llama_cpp.exists():
        print(f"ERROR: llama.cpp not found at {args.llama_cpp}", file=sys.stderr)
        print(f"  git clone https://github.com/ggerganov/llama.cpp {args.llama_cpp}", file=sys.stderr)
        sys.exit(1)

    converter, quantizer = find_tools(args.llama_cpp)
    print(f"Converter : {converter}")
    print(f"Quantizer : {quantizer or '(not found — quantization unavailable)'}")
    print(f"Output    : {args.output}")
    print(f"Quant     : {args.quant}")
    print(f"Players   : {', '.join(args.players)}\n")

    if args.quant != "none" and quantizer is None:
        print("WARNING: quantizer not found. Only fp16 GGUFs will be produced.\n")

    for i, player in enumerate(args.players, 1):
        print(f"[{i}/{len(args.players)}] {player}")
        process_player(player, args, converter, quantizer)
        print()

    print("Done.")
    print(f"Load with llama.cpp:")
    quant_suffix = args.quant if args.quant != "none" else "fp16"
    print(f"  ./llama-cli -m {args.output}/Pluribus.{quant_suffix}.gguf -p '<prompt>'")
    print(f"Or with llama-cpp-python:")
    print(f"  from llama_cpp import Llama")
    print(f"  llm = Llama(model_path='{args.output}/Pluribus.{quant_suffix}.gguf')")


if __name__ == "__main__":
    main()
