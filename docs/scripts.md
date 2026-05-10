# Scripts and Tests Reference

Fine-tunes Qwen2.5-1.5B-Instruct to play poker in the style of each player
from the Pluribus dataset, then converts and evaluates the result across several
inference backends.

---

## Pipeline Overview

```
Raw PHH hands  ──parse_phh.py──►  sft_pluribus.jsonl
                                         │
                                    01_sft.py          ← Stage 1: joint SFT (all players)
                                         │
                              sft/merged/  (fp16 base)
                                         │
                              02_lora_per_player.py    ← Stage 2: per-player adapters
                                         │
                         adapters/<player>/  (safetensors, ~37 MB each)
                                         │
                    ┌────────────────────┴───────────────────────┐
                    │                                            │
          convert_to_gguf.py                   convert_adapters_to_gguf.py
          (merged, one GGUF per player)         (shared base + per-player adapters)
                    │                                            │
            gguf/<player>.Q4_K_M.gguf          gguf-adapters/base.{f16,Q8_0,Q4_K_M}.gguf
            gguf/MrWhite.Q8_0.gguf             gguf-adapters/adapters/<player>.gguf
```

### Model paths

| Path | Contents |
|------|----------|
| `/data/models/qwen2.5-1.5b-instruct/base/` | Original Qwen2.5-1.5B-Instruct weights |
| `/data/models/qwen2.5-1.5b-instruct/sft/adapter/` | Stage 1 LoRA weights |
| `/data/models/qwen2.5-1.5b-instruct/sft/merged/` | Stage 1 merged fp16 model (Stage 2 base) |
| `/data/models/qwen2.5-1.5b-instruct/adapters/<player>/` | Stage 2 per-player safetensors adapters |
| `/data/models/qwen2.5-1.5b-instruct/gguf/<player>.Q4_K_M.gguf` | Merged GGUF, one per player |
| `/data/models/qwen2.5-1.5b-instruct/gguf/MrWhite.Q8_0.gguf` | MrWhite at Q8_0 (Q4 artifact) |
| `/data/models/qwen2.5-1.5b-instruct/gguf-adapters/base.{f16,Q8_0,Q4_K_M}.gguf` | Shared base GGUFs |
| `/data/models/qwen2.5-1.5b-instruct/gguf-adapters/adapters/<player>.gguf` | Per-player fp16 LoRA GGUFs |

---

## Scripts

### `download_model.py`

Downloads Qwen2.5-1.5B-Instruct from HuggingFace Hub to
`/data/models/qwen2.5-1.5b-instruct/base/`.

```bash
python scripts/download_model.py
python scripts/download_model.py --token hf_...   # private access
python scripts/download_model.py --verify-only    # check existing download
```

---

### `parse_phh.py`

Converts raw PHH/PHHS poker hand history files into LLM training data.
Two output modes:

**`sft` mode** — one JSON line per player decision. Each record contains a
system prompt (player identity), a user message (full game state visible to
that player: their hole cards, board, pot, stacks, action history), and an
assistant response (the action taken). Opponent hole cards are never included.

**`dapt` mode** — one compact text block per hand with no hole cards, for
domain-adaptive pretraining on HandHQ data.

```bash
# Generate SFT training data from Pluribus hands
python scripts/parse_phh.py --mode sft \
    --input /data/poker/phh-dataset/data/pluribus \
    --output player-llm/data/sft_pluribus.jsonl

# Restrict to specific players
python scripts/parse_phh.py --mode sft \
    --input /data/poker/phh-dataset/data/pluribus \
    --output player-llm/data/sft_pluribus.jsonl \
    --players Pluribus MrBlue

# Generate DAPT pretraining corpus from HandHQ high-stakes hands
python scripts/parse_phh.py --mode dapt \
    --input /data/poker/phh-dataset/data/handhq \
    --output player-llm/data/dapt_handhq.txt \
    --stakes 400NLH 600NLH 1000NLH \
    --limit 500000
```

Output dataset: `player-llm/data/sft_pluribus.jsonl` — 91,356 records,
14 players, 10,000 Pluribus hands.

---

### `01_sft.py`

**Stage 1** — Joint supervised fine-tuning on all 91K Pluribus decision points.
Trains Qwen2.5-1.5B-Instruct using QLoRA (4-bit NF4 base + LoRA r=32, alpha=64)
via `trl.SFTTrainer`. All 14 players are trained jointly; player identity is
encoded in the system prompt. Uses a 90/10 train/eval split (seed 42).

Outputs:
- `sft/adapter/` — Stage 1 LoRA weights only
- `sft/merged/` — adapter merged into fp16 base (used as the Stage 2 starting point)

```bash
python scripts/01_sft.py                          # full 3-epoch run
python scripts/01_sft.py --epochs 1 --batch 4    # smoke test
python scripts/01_sft.py --dry-run               # one step only
python scripts/01_sft.py --data /path/sft.jsonl  # custom dataset
```

---

### `02_lora_per_player.py`

**Stage 2** — Per-player LoRA adapters, one per well-represented player.
Starts from `sft/merged/` and trains a small LoRA (r=16, alpha=32) for each
player with ≥ 5,000 decision points. The smaller rank compared to Stage 1
focuses the adapter on style rather than poker knowledge.

Default players: Pluribus, MrBlue, MrOrange, Bill, MrPink, Eddie, MrWhite.

Adapters saved to `/data/models/qwen2.5-1.5b-instruct/adapters/<player>/`.
Each adapter is ~37 MB (fp16 safetensors).

Uses `adamw_bnb_8bit` optimizer (not `paged_adamw_8bit`, which causes CUDA
memory errors when training players sequentially in one process).

```bash
python scripts/02_lora_per_player.py                            # all 7 players
python scripts/02_lora_per_player.py --players Pluribus MrBlue # specific players
python scripts/02_lora_per_player.py --min-examples 3000       # lower threshold
python scripts/02_lora_per_player.py --dry-run                 # 1 step per player
```

---

### `merge_adapter.py`

Merges a LoRA adapter into its base model in full fp16 precision and saves a
standalone HuggingFace model (no bitsandbytes dependency at load time).
Used to produce `sft/merged/` after Stage 1, and can also merge individual
player adapters for inspection or GGUF conversion.

```bash
# Merge the Stage 1 SFT adapter (default)
python scripts/merge_adapter.py

# Merge a specific player adapter onto the SFT merged model
python scripts/merge_adapter.py \
    --base    /data/models/qwen2.5-1.5b-instruct/sft/merged \
    --adapter /data/models/qwen2.5-1.5b-instruct/adapters/Pluribus \
    --output  /data/models/qwen2.5-1.5b-instruct/adapters/Pluribus/merged

python scripts/merge_adapter.py --dtype bf16   # save as bf16 instead of fp16
```

---

### `convert_to_gguf.py`

Produces **one merged GGUF per player** by merging each player's adapter into
the SFT base model (fp16), then quantizing with `llama-quantize`. Simple
single-file inference but uses ~941 MB × 7 players = ~6.5 GB total storage.

**Known artifact:** MrWhite's Q4_K_M GGUF predicts "check" on a flop raise
spot where fp16 and Q8_0 both predict "raise 352". Quantization noise shifts
a marginal decision when the adapter is merged before quantization. MrWhite
is also converted to Q8_0 (~1.6 GB) to work around this.

```bash
# Requires llama.cpp built or installed (llama-quantize in PATH)
python scripts/convert_to_gguf.py --llama-cpp ~/dev/ai/llama.cpp

# Specific players or quant
python scripts/convert_to_gguf.py --llama-cpp ~/dev/ai/llama.cpp \
    --players MrWhite --quants Q8_0

# Skip players whose GGUF already exists
python scripts/convert_to_gguf.py --llama-cpp ~/dev/ai/llama.cpp --skip-existing
```

Output: `/data/models/qwen2.5-1.5b-instruct/gguf/<player>.<quant>.gguf`

---

### `convert_adapters_to_gguf.py`

Produces a **single shared base GGUF** plus **one small adapter GGUF per player**.
The adapter (~37 MB fp16) is applied at inference time over the quantized base
using llama.cpp's LoRA support. Much more storage-efficient than merged GGUFs.

Storage comparison:
| Mode | Size |
|------|------|
| Merged Q4_K_M (7 players) | ~6.5 GB |
| Adapter mode Q4_K_M base + adapters | ~1.2 GB |
| Adapter mode Q8_0 base + adapters | ~1.9 GB |
| Adapter mode fp16 base + adapters | ~3.2 GB |

The fp16 base matches HuggingFace fp16 accuracy most closely. Q8_0 is
essentially identical to fp16 in practice.

```bash
python scripts/convert_adapters_to_gguf.py --llama-cpp ~/dev/ai/llama.cpp

# fp16 base (matches HF accuracy)
python scripts/convert_adapters_to_gguf.py --llama-cpp ~/dev/ai/llama.cpp \
    --base-quant f16

# Skip adapters that already exist
python scripts/convert_adapters_to_gguf.py --llama-cpp ~/dev/ai/llama.cpp \
    --base-quant f16 --skip-existing

# Supported base quants: f16, Q8_0, Q4_K_M, Q5_K_M, Q6_K
```

Output:
- `/data/models/qwen2.5-1.5b-instruct/gguf-adapters/base.<quant>.gguf`
- `/data/models/qwen2.5-1.5b-instruct/gguf-adapters/adapters/<player>.gguf`

---

### `inference.py`

HuggingFace inference using the fp16 merged base + safetensors LoRA adapters.
Highest fidelity (no quantization). Loads the base once, then swaps adapters
per player with `model.set_adapter()`.

```bash
# Single player, game state from file
python scripts/inference.py --player Pluribus --state game_state.txt

# Multiple players on the same state
python scripts/inference.py --players Pluribus MrBlue MrWhite --state game_state.txt

# Read state from stdin
echo "Street: Preflop..." | python scripts/inference.py --player Bill

# Interactive REPL (paste game state, blank line to submit)
python scripts/inference.py --player Eddie --interactive

# Batch mode: JSONL with {player, state} records
python scripts/inference.py --batch input.jsonl --output predictions.jsonl

# Lower memory with quantization
python scripts/inference.py --player Pluribus --state game.txt --load-in-4bit
```

---

### `inference_gguf.py`

llama-cpp-python inference supporting two modes:

**Merged mode** — one GGUF per player, no adapter at inference time:
```bash
python scripts/inference_gguf.py --player Pluribus --state game.txt
python scripts/inference_gguf.py --players Pluribus MrBlue --state game.txt
python scripts/inference_gguf.py --player MrWhite --quant Q8_0 --state game.txt
python scripts/inference_gguf.py --player Pluribus --interactive
python scripts/inference_gguf.py --batch input.jsonl --output predictions.jsonl
```

**Adapter mode** — shared quantized base + fp16 LoRA adapter per player:
```bash
python scripts/inference_gguf.py \
    --base-gguf /data/models/qwen2.5-1.5b-instruct/gguf-adapters/base.f16.gguf \
    --player Pluribus --state game.txt

# Multiple players
python scripts/inference_gguf.py \
    --base-gguf gguf-adapters/base.Q8_0.gguf \
    --players Pluribus MrBlue Eddie --state game.txt
```

Key flags: `--gpu-layers -1` (all on GPU, default), `--ctx 512`, `--temperature 0.0`
(greedy, default), `--quant Q4_K_M` (default for merged mode).

---

## Tests

All tests are run from the repo root with `uv run python tests/<script>.py`.

---

### `tests/game_states.py`

Shared constants used across all test scripts. Not a test itself.

- `ALL_PLAYERS` — list of the 7 trained players in training order
- `PREFLOP_FOLD` — preflop spot where all players should fold (6c 7s BTN vs HJ raise)
- `PREFLOP_FOLD_LABEL` — expected action string `"fold"`
- `FLOP_RAISE` — flop spot where all players should raise (KJo on 7s9cTc, checked to)

---

### `tests/test_hf_adapters.py`

Tests HuggingFace fp16 inference with safetensors adapters. Loads the merged
SFT base once, loads all 7 player adapters, and runs both game states through
each player using `model.set_adapter()`. Also runs a MrWhite sanity check
(must predict raise, not check, on the flop spot).

```bash
uv run python tests/test_hf_adapters.py
```

Expected: all PASS. GPU usage ~3 GB. Inference is slow (~2–5 s/call) compared
to GGUF.

---

### `tests/test_merged_gguf.py`

Tests merged GGUF inference (one GGUF file per player, base + adapter baked in).
Runs all 7 players through both game states at Q4_K_M quantization, then
compares MrWhite at Q4_K_M vs Q8_0 on the flop raise spot.

```bash
uv run python tests/test_merged_gguf.py
```

Expected: all PASS except MrWhite Q4_K_M on flop raise, which is documented as
NOTE (predicts "check" — known Q4_K_M quantization artifact when adapter is
merged before quantization). MrWhite Q8_0 correctly predicts "raise 352".

Requires llama-cpp-python built with CUDA (`CMAKE_ARGS="-DGGML_CUDA=on"`).

---

### `tests/test_gguf_adapters.py`

Tests GGUF adapter mode (shared quantized base + per-player fp16 LoRA adapter).
Runs all 7 players through both game states for each base quant (fp16, Q8_0,
Q4_K_M). Also runs a MrWhite cross-quant comparison on the flop raise spot.

```bash
uv run python tests/test_gguf_adapters.py
```

Expected: all PASS. The fp16 and Q8_0 bases are run first (while VRAM is
unfragmented) since fp16 requires a contiguous 2.9 GB allocation.

**Implementation note:** Each player model is loaded once and run through all
game states before being freed (`llm.close()` then `del llm`). This avoids
VRAM exhaustion caused by a reference cycle through the LoRA adapter C struct
in llama-cpp-python 0.3.22, which prevents the Python garbage collector from
freeing GPU memory between loop iterations when using `del llm` alone.

---

### `tests/test_accuracy.py`

Evaluates one or more inference variants against ground-truth player actions
from the Pluribus validation set (exact reproduction of the 10% eval split
used during Stage 2 training: seed=42, per-player shuffle).

**Scoring:**
- **Exact** — exact string match (e.g., `"raise 352"` vs `"raise 350"` = wrong)
- **Weighted** — exact for fold/check/call; raises scored by
  `max(0, 1 − |pred_amount − truth_amount| / truth_amount)` so close
  sizings get partial credit

**Variants:**

| Name | Description | Storage |
|------|-------------|---------|
| `hf` | HuggingFace fp16 base + safetensors adapters | ~3 GB |
| `gguf-f16` | GGUF fp16 base + fp16 LoRA adapter | ~3.2 GB total |
| `gguf-q8` | GGUF Q8_0 base + fp16 LoRA adapter | ~1.9 GB total |
| `gguf-q4` | GGUF Q4_K_M base + fp16 LoRA adapter | ~1.2 GB total |

```bash
# All variants, 100 examples per player (default)
uv run python tests/test_accuracy.py

# Single variant
uv run python tests/test_accuracy.py --variants hf
uv run python tests/test_accuracy.py --variants gguf-q4

# Compare two variants with more examples
uv run python tests/test_accuracy.py --variants gguf-f16 gguf-q8 --n-samples 500

# Specific players only
uv run python tests/test_accuracy.py --variants gguf-q4 --players MrWhite Pluribus
```

**Key findings (100 samples/player):**

| Variant | Exact | Weighted |
|---------|-------|----------|
| HF fp16 | 74.9% | 79.8% |
| GGUF f16+lora | 83.0% | 86.8% |
| GGUF Q8+lora | 83.1% | 86.9% |
| GGUF Q4+lora | 84.7% | 87.8% |

GGUF variants outperform HF fp16 primarily because HF mispredicts check spots
(24% accuracy vs 76–86% for GGUF). Raise exact-match accuracy is similar across
all variants (~50–54%); weighted raise accuracy is higher (79–85%) since the
model usually sizes within 25% of the target. Q4+lora leads overall due to
stronger check and call accuracy despite slightly lower raise type prediction.

---

## Accuracy Notes

**Why token accuracy (~95%) is higher than action accuracy (~75–84%):**

Token accuracy during training measures correctness per output token, not per
complete action. Folds account for ~53% of examples and are a single token the
model gets right nearly 100% of the time. This alone pushes the metric above
95% before accounting for other actions. Raises get partial token credit when
the model produces the right verb ("raise") but the wrong amount — e.g., "raise
350" vs "raise 352" is ~50% token accuracy but 0% exact action accuracy.

**Why GGUF outperforms HF fp16:**

HF uses `attn_implementation="sdpa"` with PyTorch attention kernels, while
llama.cpp uses its own fused CUDA kernels. Even at fp16, the two produce
different numerical results due to different floating-point operation ordering.
For marginal decisions (especially check vs. raise), this shifts HF toward
raising in spots the ground truth player checked. The GGUF inference path
happens to be better calibrated for check spots in this dataset.

**Why fp16 GGUF ≠ fp16 HF exactly:**

GGUF stores RMSNorm weights and attention biases as F32 even in an "f16" file,
while HF keeps them in fp16. Accumulated rounding differences across 28 layers
are small but enough to shift a marginal token decision.
