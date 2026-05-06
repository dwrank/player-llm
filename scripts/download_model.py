#!/usr/bin/env python3
"""
Download Qwen2.5-1.5B-Instruct from HuggingFace Hub.

Usage:
    python download_model.py
    python download_model.py --token hf_...
    python download_model.py --verify-only
"""

import argparse
import sys
from pathlib import Path

MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
LOCAL_DIR = Path("/data/models/qwen2.5-1.5b-instruct/base")


def download(model_id: str, local_dir: Path, token: str | None) -> None:
    from huggingface_hub import snapshot_download

    print(f"Downloading {model_id}")
    print(f"Destination: {local_dir}\n")
    local_dir.mkdir(parents=True, exist_ok=True)

    snapshot_download(
        repo_id=model_id,
        local_dir=str(local_dir),
        token=token,
        ignore_patterns=[
            "*.msgpack", "*.h5",
            "flax_model*", "tf_model*", "rust_model*",
            "original/*",
        ],
    )


def verify(local_dir: Path) -> bool:
    required = [
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "generation_config.json",
    ]
    missing = [f for f in required if not (local_dir / f).exists()]
    if missing:
        print(f"ERROR: missing files: {missing}", file=sys.stderr)
        return False

    shards = sorted(local_dir.glob("*.safetensors"))
    if not shards:
        print("ERROR: no .safetensors weight files found", file=sys.stderr)
        return False

    total_gb = sum(f.stat().st_size for f in shards) / 1e9
    print(f"Model verified at {local_dir}")
    print(f"  Weight shards : {len(shards)}")
    print(f"  Total size    : {total_gb:.2f} GB")
    return True


def load_test(local_dir: Path) -> None:
    print("\nRunning load test...")
    try:
        from transformers import AutoTokenizer, AutoConfig

        tok = AutoTokenizer.from_pretrained(str(local_dir))
        cfg = AutoConfig.from_pretrained(str(local_dir))
        n_tokens = len(tok("Hello poker world")["input_ids"])

        print(f"  Tokenizer  : vocab size {tok.vocab_size:,}")
        print(f"  Config     : {cfg.model_type}, "
              f"{cfg.num_hidden_layers} layers, "
              f"hidden dim {cfg.hidden_size}")
        print(f"  Sample     : 'Hello poker world' → {n_tokens} tokens")
    except Exception as e:
        print(f"  Load test failed: {e}", file=sys.stderr)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download Qwen2.5-1.5B-Instruct")
    parser.add_argument("--model", default=MODEL_ID)
    parser.add_argument("--dir", type=Path, default=LOCAL_DIR)
    parser.add_argument("--token", default=None, help="HuggingFace API token")
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--no-test", action="store_true", help="Skip tokenizer load test")
    args = parser.parse_args()

    if not args.verify_only:
        if args.dir.exists() and any(args.dir.glob("*.safetensors")):
            print(f"Model already present at {args.dir}")
        else:
            download(args.model, args.dir, args.token)

    if not verify(args.dir):
        sys.exit(1)

    if not args.no_test:
        load_test(args.dir)

    print(f"\nReady. Base model at: {args.dir}")


if __name__ == "__main__":
    main()
