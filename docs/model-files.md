# Model Files Reference

All model files live under `/data/models/qwen2.5-1.5b-instruct/`. This document
describes every file and directory, what produced it, and whether it is needed
at runtime.

---

## How LoRA adapters work (background)

LoRA (Low-Rank Adaptation) freezes the base model weights and adds a small set
of trainable delta matrices. For each targeted weight matrix W (shape d×k), two
small matrices A (d×r) and B (r×k) are trained. At inference the effective weight
is W + A×B. The rank r controls capacity: Stage 1 uses r=32, Stage 2 uses r=16.

The adapter file stores only A and B — not W — which is why adapters are much
smaller than the base model. At inference the adapter is either merged into the
base weights once (merged mode) or applied dynamically on top of a shared base
(adapter mode).

---

## Directory tree

```
/data/models/qwen2.5-1.5b-instruct/
├── base/                        ← downloaded from HuggingFace
├── sft/
│   ├── adapter/                 ← Stage 1 LoRA weights (intermediate)
│   ├── checkpoints/             ← Stage 1 training checkpoints
│   └── merged/                  ← Stage 1 baked into base (fp16) ← used by Stage 2
├── adapters/
│   └── <player>/                ← Stage 2 per-player LoRA weights (HF inference)
│       └── checkpoints/
├── gguf/                        ← merged GGUFs (one per player, merged mode)
└── gguf-adapters/               ← shared base GGUFs + adapter GGUFs (adapter mode)
    └── adapters/
```

---

## Base model — `base/`

Downloaded by `download_model.py` from `Qwen/Qwen2.5-1.5B-Instruct` on HuggingFace.

| File | Size | Description |
|------|------|-------------|
| `model.safetensors` | 2.9 GB | All 1.54B model weights in fp16 (2 bytes × 1.54B params) |
| `config.json` | 1 KB | Architecture: 28 layers, hidden=1536, heads=12, KV heads=2 |
| `tokenizer.json` | 6.8 MB | Full tokenizer vocabulary and merge rules (151,936 tokens) |
| `tokenizer_config.json` | 7.2 KB | Tokenizer settings and chat template config |
| `merges.txt` | 1.6 MB | BPE merge rules (human-readable form of tokenizer merges) |
| `vocab.json` | 2.7 MB | Token-to-id mapping |
| `generation_config.json` | 242 B | Default generation parameters |
| `.cache/huggingface/` | — | HuggingFace download metadata (checksums, ETags) |

**Runtime use:** Required by Stage 1 training. Not used directly at inference
(the merged model is used instead).

---

## Stage 1 SFT — `sft/`

Produced by `01_sft.py` (LoRA adapter) and `merge_adapter.py` (merged model).
Stage 1 trains general poker decision-making across all players.

### `sft/adapter/` — Stage 1 LoRA weights

| File | Size | Description |
|------|------|-------------|
| `adapter_model.safetensors` | 141 MB | LoRA delta matrices: rank=32, alpha=64, all 7 attention+MLP projection layers |
| `adapter_config.json` | 1.1 KB | LoRA config: r=32, lora_alpha=64, target_modules=[q,k,v,o,gate,up,down]_proj |
| `tokenizer.json` | 11 MB | Tokenizer copy (saved alongside adapter by PEFT) |
| `tokenizer_config.json` | 691 B | Tokenizer config copy |
| `chat_template.jinja` | 2.5 KB | Qwen2.5 chat template |

The adapter is larger than Stage 2 adapters (141 MB vs 71 MB) because it uses
rank 32 vs Stage 2's rank 16 — double the trainable parameters per layer.

**Runtime use:** Only needed to regenerate `sft/merged/` via `merge_adapter.py`.
Not used at inference.

### `sft/merged/` — Stage 1 merged fp16 model

Produced by `merge_adapter.py`. The Stage 1 adapter delta weights are added into
the base weights and saved as a clean fp16 model. This is the base model for
Stage 2 training.

| File | Size | Description |
|------|------|-------------|
| `model.safetensors` | 2.9 GB | Base + Stage 1 adapter merged, fp16. Same size as base — same parameter count, same dtype |
| `config.json` | 1.4 KB | Architecture config (same as base, slightly modified by merge) |
| `tokenizer.json` | 11 MB | Tokenizer |
| `tokenizer_config.json` | 691 B | Tokenizer config |
| `generation_config.json` | 241 B | Generation config |
| `chat_template.jinja` | 2.5 KB | Chat template |

**Runtime use:** Required by Stage 2 training (`02_lora_per_player.py`). Also
used directly for HuggingFace fp16 inference via `inference.py`.

### `sft/checkpoints/` — Stage 1 training checkpoints

Saved periodically during `01_sft.py` training. The two most recent are kept
(`save_total_limit=2`).

| File per checkpoint | Size | Description |
|--------------------|------|-------------|
| `adapter_model.safetensors` | 71 MB | Adapter weights at this checkpoint step |
| `optimizer.pt` | 72 MB | Adam optimizer state (needed to resume training) |
| `rng_state.pth` | 14 KB | Python/CUDA RNG state (for exact resume) |
| `scheduler.pt` | 1.1 KB | Learning rate scheduler state |
| `trainer_state.json` | 108–118 KB | Loss history, best checkpoint info |
| `training_args.bin` | 5.3 KB | All training hyperparameters |
| `tokenizer.*` | 11 MB | Tokenizer copy |

**Runtime use:** None — only needed to resume an interrupted training run.
Safe to delete once training is complete and `sft/adapter/` is saved.

---

## Stage 2 per-player adapters — `adapters/<player>/`

Produced by `02_lora_per_player.py`. One adapter per player, trained on top of
`sft/merged/`. Each captures that player's betting tendencies and style.

Players: `Bill`, `Eddie`, `MrBlue`, `MrOrange`, `MrPink`, `MrWhite`, `Pluribus`

### `adapters/<player>/` — final adapter weights

| File | Size | Description |
|------|------|-------------|
| `adapter_model.safetensors` | 71 MB | LoRA delta matrices: rank=16, alpha=32, all 7 projection layers |
| `adapter_config.json` | 1.1 KB | LoRA config: r=16, lora_alpha=32, base=`sft/merged` |
| `tokenizer.json` | 11 MB | Tokenizer copy |
| `tokenizer_config.json` | 691 B | Tokenizer config |
| `chat_template.jinja` | 2.5 KB | Chat template |

The 71 MB size comes from rank=16 × 7 target modules × 28 layers × 2 matrices
× 2 bytes (fp16) × average projection dimension.

**Runtime use:** Required for HuggingFace fp16 inference via `inference.py`.
Also the source for GGUF adapter conversion via `convert_adapters_to_gguf.py`.

### `adapters/<player>/checkpoints/` — Stage 2 training checkpoints

Same structure as Stage 1 checkpoints but smaller (Stage 2 trains fewer steps).

| File per checkpoint | Size | Description |
|--------------------|------|-------------|
| `adapter_model.safetensors` | 36 MB | Adapter weights at this checkpoint |
| `optimizer.pt` | 37 MB | Adam optimizer state |
| `rng_state.pth` | 15 KB | RNG state |
| `scheduler.pt` | 1.5 KB | LR scheduler state |
| `trainer_state.json` | 2–5 KB | Loss history |
| `training_args.bin` | 5.7 KB | Training hyperparameters |
| `tokenizer.*` | 11 MB | Tokenizer copy |

**Runtime use:** None. Safe to delete once the final adapter is saved.

---

## GGUF merged mode — `gguf/`

Produced by `convert_to_gguf.py`. Each file contains the Stage 1 base +
one player's Stage 2 adapter merged together and quantized. Self-contained —
no separate adapter file needed at inference.

| File | Size | Description |
|------|------|-------------|
| `<player>.Q4_K_M.gguf` | 941 MB | All 7 players. Q4_K_M quantization (~4.5 bits/weight) |
| `MrWhite.Q8_0.gguf` | 1.6 GB | MrWhite at Q8_0 — added because Q4_K_M quantization noise flips his marginal flop decision |

**Q4_K_M** uses a mixed quantization: most layers at 4-bit with K-quant grouping,
sensitive layers (output projection) at 6-bit. Average ~4.5 bits per weight.

**Q8_0** is 8-bit linear quantization — 1 byte per weight, minimal accuracy loss.

**Runtime use:** Used by `inference_gguf.py` in merged mode. One file loaded
per player, so 7 separate loads for all players.

---

## GGUF adapter mode — `gguf-adapters/`

Produced by `convert_adapters_to_gguf.py`. Separates the base from the
per-player adapters so the base can be shared across all players.

### `gguf-adapters/` — shared base GGUFs

| File | Size | Description |
|------|------|-------------|
| `base.f16.gguf` | 2.9 GB | fp16 base — matches HF fp16 in weight precision |
| `base.Q8_0.gguf` | 1.6 GB | Q8_0 quantized base — best accuracy/size balance |
| `base.Q4_K_M.gguf` | 941 MB | Q4_K_M quantized base — smallest, some marginal decision instability |

The fp16 GGUF is numerically slightly different from the HF fp16 model due to
GGUF storing RMSNorm weights as F32 and different CUDA attention kernels.

### `gguf-adapters/adapters/` — per-player adapter GGUFs

| File | Size | Description |
|------|------|-------------|
| `<player>.gguf` | 36 MB | fp16 LoRA adapter in GGUF format — all 7 players |

Smaller than the safetensors adapters (71 MB) because GGUF adapter format omits
the tokenizer copies and config files that PEFT saves alongside the weights.

**Runtime use:** Load one base GGUF once, then apply per-player adapter at
inference time via `llama_set_adapter_lora()`. Supports hot-swapping between
players without reloading the base.

---

## Storage summary

| Directory | Total size | Required for |
|-----------|-----------|--------------|
| `base/` | ~3.0 GB | Stage 1 training, can skip if `sft/merged/` exists |
| `sft/adapter/` | ~165 MB | Regenerating `sft/merged/` only |
| `sft/merged/` | ~2.9 GB | Stage 2 training, HF inference |
| `sft/checkpoints/` | ~350 MB | Resume Stage 1 training only |
| `adapters/` (7×) | ~580 MB weights + ~580 MB checkpoints | HF inference, GGUF conversion |
| `gguf/` (7×Q4 + 1×Q8) | ~7.3 GB | Merged mode GGUF inference |
| `gguf-adapters/` | ~5.3 GB | Adapter mode GGUF inference |

**Minimum footprint for adapter-mode inference (Q8_0):**
`sft/merged/` + `gguf-adapters/base.Q8_0.gguf` + `gguf-adapters/adapters/` = ~4.8 GB

**Minimum footprint for adapter-mode inference (Q4_K_M):**
`sft/merged/` + `gguf-adapters/base.Q4_K_M.gguf` + `gguf-adapters/adapters/` = ~4.2 GB

(`sft/merged/` is needed only if using HF inference or retraining Stage 2.
For GGUF-only inference it can be excluded, dropping to ~1.9 GB for Q8_0.)
