# Action Classes — 22-Class Discretized Output

## Overview

The model outputs a single token representing the chosen action.  The output
space is 22 classes, each mapped to a single ASCII character (token IDs 0–21
in the Qwen2.5 vocabulary).

This avoids generating free-form text and makes the model output directly
usable for action selection without postprocessing.

## Class Table

| Class | Char | Token ID | Label      | Context     |
|------:|------|---------|------------|-------------|
| 0  | `!` | 0  | fold       | any         |
| 1  | `"` | 1  | check      | any         |
| 2  | `#` | 2  | call       | any         |
| 3  | `$` | 3  | 2bb        | preflop     |
| 4  | `%` | 4  | 2.5bb      | preflop     |
| 5  | `&` | 5  | 3bb        | preflop     |
| 6  | `'` | 6  | 4bb        | preflop     |
| 7  | `(` | 7  | 7bb        | preflop     |
| 8  | `)` | 8  | 8bb        | preflop     |
| 9  | `*` | 9  | 9bb        | preflop     |
| 10 | `+` | 10 | 10bb       | preflop     |
| 11 | `,` | 11 | 11bb       | preflop     |
| 12 | `-` | 12 | 12bb       | preflop     |
| 13 | `.` | 13 | 13bb       | preflop     |
| 14 | `/` | 14 | 1/3 pot    | postflop    |
| 15 | `0` | 15 | 1/2 pot    | postflop    |
| 16 | `1` | 16 | 2/3 pot    | postflop    |
| 17 | `2` | 17 | pot        | postflop    |
| 18 | `3` | 18 | 1.25x pot  | postflop    |
| 19 | `4` | 19 | 1.5x pot   | postflop    |
| 20 | `5` | 20 | 1.75x pot  | postflop    |
| 21 | `6` | 21 | all-in     | any         |

Classes 0–21 use ASCII characters 33–54 (`!` through `6`).  These are
ordinary printable characters with no special meaning in the Qwen2.5
vocabulary.

## Binning Rules

### Preflop raises (BB multiples)

| Range (BBs)         | Class  |
|---------------------|--------|
| < 2.25              | 2bb    |
| 2.25 – 2.74         | 2.5bb  |
| 2.75 – 3.49         | 3bb    |
| 3.5 – 5.49          | 4bb    |
| 5.5 – 7.49          | 7bb    |
| 7.5 – 8.49          | 8bb    |
| 8.5 – 9.49          | 9bb    |
| 9.5 – 10.49         | 10bb   |
| 10.5 – 11.49        | 11bb   |
| 11.5 – 12.49        | 12bb   |
| ≥ 12.5              | 13bb   |

### Postflop raises (pot fractions)

| Fraction range   | Class     |
|------------------|-----------|
| < 0.45x pot      | 1/3 pot   |
| 0.45 – 0.59x     | 1/2 pot   |
| 0.60 – 0.84x     | 2/3 pot   |
| 0.85 – 1.19x     | pot       |
| 1.20 – 1.37x     | 1.25x pot |
| 1.375 – 1.62x    | 1.5x pot  |
| ≥ 1.625x         | 1.75x pot |

### All-in override

Any raise that is ≥ 80% of the acting player's remaining stack is classified
as all-in regardless of the pot fraction or BB multiple.

## GGUF Compatibility

The full Qwen2.5 vocabulary has ~150k tokens.  During Stage 1 SFT training,
`modules_to_save=["lm_head"]` is set so the classification head is part of
the LoRA adapter and is fully trained alongside the class token rows.

After training and merging, rows 22 and above in `lm_head.weight` are zeroed
before the model is saved:

```python
merged.lm_head.weight[22:] = 0.0
```

This keeps the GGUF-exported weight matrix at the full vocabulary size while
making all non-class tokens unreachable.  At inference, logit bias
`{i: -1e9 for i in range(32000) if i not in CLASS_VOCAB_IDS}` is applied to
ensure the model always outputs one of the 22 class characters.

## Training Oversampling

Raise examples (classes 3–21) are duplicated 2× in the training JSONL to
compensate for their low base rate (~20% of decisions) and high error rate
(~48% of classification errors).  This is applied in `parse_phh.py main()`
and is transparent to the trainer.
