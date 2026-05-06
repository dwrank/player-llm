# Environment Setup

## Package Manager: uv (recommended)

Both `uv` and `pixi` are installed. Use **uv** for this project — it is
simpler and PyTorch pip wheels bundle their own CUDA runtime, so a conda
environment is not needed.

Use **pixi** instead only if you need to manage the CUDA toolkit itself
(e.g., building custom CUDA extensions or using a CUDA version not bundled
by PyTorch wheels).

## Virtual Environment Location

The `.venv` lives at `~/dev/poker/.venv` — one level above this project.
All sub-projects under `~/dev/poker/` share it, so you only install
dependencies once.

## First-Time Setup

```bash
cd ~/dev/poker

# Create the shared virtual environment
uv venv --python 3.11

# Activate it
source .venv/bin/activate

# Install all dependencies (reads pyproject.toml)
# For CUDA 12.1 (most common on modern GPUs):
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# For CPU-only (or if unsure):
# uv pip install torch torchvision

# Install remaining dependencies
uv pip install -r pyproject.toml

# Verify GPU is visible (if applicable)
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU only')"
```

## Daily Use

```bash
# Activate the shared env from anywhere under ~/dev/poker/
source ~/dev/poker/.venv/bin/activate

# Or run a script directly without activating
~/dev/poker/.venv/bin/python player-llm/scripts/download_model.py
```

## Download the Base Model

```bash
# From ~/dev/poker/player-llm/
python scripts/download_model.py
```

This saves to `/data/models/qwen2.5-1.5b-instruct/base/`.

You can also download manually via HuggingFace CLI:
```bash
huggingface-cli download Qwen/Qwen2.5-1.5B-Instruct \
    --local-dir /data/models/qwen2.5-1.5b-instruct/base
```

## Data Paths

| Resource | Path |
|----------|------|
| Pluribus PHH hands | `/data/poker/phh-dataset/data/pluribus/` |
| HandHQ amateur hands | `/data/poker/phh-dataset/data/handhq/` |
| WSOP pro hands | `/data/poker/phh-dataset/data/wsop/2023/43/5/` |
| Base model | `/data/models/qwen2.5-1.5b-instruct/base/` |
| DAPT checkpoint | `/data/models/qwen2.5-1.5b-instruct/dapt/` |
| SFT checkpoint | `/data/models/qwen2.5-1.5b-instruct/sft/` |
| Per-player adapters | `/data/models/qwen2.5-1.5b-instruct/adapters/<name>/` |

## Hardware Requirements

| Task | Minimum | Recommended |
|------|---------|-------------|
| Stage 0 DAPT | 8 GB GPU | 16 GB GPU |
| Stage 1 SFT (QLoRA) | 6 GB GPU | 8–16 GB GPU |
| Stage 2 LoRA adapters | 6 GB GPU | 8 GB GPU |
| Inference (4-bit) | CPU or 2 GB GPU | 4 GB GPU |
| Inference (fp16) | 4 GB GPU | 8 GB GPU |

All training uses QLoRA (4-bit quantized base + LoRA adapters) via
`bitsandbytes`, which keeps GPU memory well within 8–16 GB even during
training.

## Adding New Python Packages

```bash
# Add to pyproject.toml and install
cd ~/dev/poker
uv add <package-name>
```

All sub-projects under `~/dev/poker/` automatically pick up the new package
since they share the `.venv`.
