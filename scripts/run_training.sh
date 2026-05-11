#!/bin/bash
set -e

SCRIPTS=/home/drank/dev/poker/player-llm/scripts
DATA=/home/drank/dev/poker/player-llm/data/sft_pluribus.jsonl
LLAMA=/home/drank/dev/ai/llama.cpp
export POKER_EVAL_LIB=/home/drank/dev/poker-eval/poker-eval-cpp/build/libpoker-eval.so

log() { echo "[$(date '+%H:%M:%S')] $*"; }

log "=== Stage 1: Joint SFT ==="
python "$SCRIPTS/01_sft.py" --data "$DATA"
log "Stage 1 complete"

log "=== Merge: Stage 1 adapter → clean fp16 ==="
python "$SCRIPTS/merge_adapter.py"
log "Merge complete"

log "=== Stage 2: Per-player adapters ==="
python "$SCRIPTS/02_lora_per_player.py" --data "$DATA"
log "Stage 2 complete"

log "=== GGUF: adapter mode ==="
python "$SCRIPTS/convert_adapters_to_gguf.py" --llama-cpp "$LLAMA"
log "GGUF adapter mode complete"

log "=== GGUF: merged mode ==="
python "$SCRIPTS/convert_to_gguf.py" --llama-cpp "$LLAMA"
log "GGUF merged mode complete"

log "=== All stages done ==="
