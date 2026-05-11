"""
action_classes.py — Shared action class definitions for poker decision models.

22 output classes, mapped to token IDs 0–21 in the Qwen2.5 vocabulary
(characters '!' through '6', ASCII 33–54).  These are ordinary printable tokens
with no special meaning in the base vocabulary.

Preflop raises use BB multiples; postflop raises use pot fractions.  The two
label sets share 'all-in', and postflop can fall back to BB labels for micro
raises (< 0.25x pot) that are effectively min-raises.
"""

# ── Class definitions ─────────────────────────────────────────────────────────

CLASSES: list[str] = [
    "fold",       # 0  →  '!'
    "check",      # 1  →  '"'
    "call",       # 2  →  '#'
    "2bb",        # 3  →  '$'
    "2.5bb",      # 4  →  '%'
    "3bb",        # 5  →  '&'
    "4bb",        # 6  →  "'"
    "7bb",        # 7  →  '('
    "8bb",        # 8  →  ')'
    "9bb",        # 9  →  '*'
    "10bb",       # 10 →  '+'
    "11bb",       # 11 →  ','
    "12bb",       # 12 →  '-'
    "13bb",       # 13 →  '.'
    "1/3 pot",    # 14 →  '/'
    "1/2 pot",    # 15 →  '0'
    "2/3 pot",    # 16 →  '1'
    "pot",        # 17 →  '2'
    "1.25x pot",  # 18 →  '3'
    "1.5x pot",   # 19 →  '4'
    "1.75x pot",  # 20 →  '5'
    "all-in",     # 21 →  '6'
]

N_CLASSES       = len(CLASSES)                         # 22
CLASS_VOCAB_IDS = list(range(N_CLASSES))               # token IDs 0–21
CLASS_CHARS     = [chr(33 + i) for i in range(N_CLASSES)]  # '!' … '6'

CLASS_TO_CHAR  : dict[int, str] = {i: chr(33 + i)  for i in range(N_CLASSES)}
CHAR_TO_CLASS  : dict[str, int] = {chr(33 + i): i  for i in range(N_CLASSES)}
CLASS_TO_LABEL : dict[int, str] = dict(enumerate(CLASSES))
LABEL_TO_CLASS : dict[str, int] = {v: k for k, v in CLASS_TO_LABEL.items()}

CLASS_TOKENS_LIST = " | ".join(CLASS_CHARS)

# ── Binning parameters ────────────────────────────────────────────────────────

_SHOVE_STACK = 0.80  # raise ≥ 80 % of remaining stack → all-in

# Preflop: (lo_bb_inclusive, hi_bb_exclusive, label)
_PREFLOP_BINS: list[tuple[float, float, str]] = [
    (0.0,   2.25,  "2bb"),
    (2.25,  2.75,  "2.5bb"),
    (2.75,  3.5,   "3bb"),
    (3.5,   5.5,   "4bb"),
    (5.5,   7.5,   "7bb"),
    (7.5,   8.5,   "8bb"),
    (8.5,   9.5,   "9bb"),
    (9.5,  10.5,   "10bb"),
    (10.5, 11.5,   "11bb"),
    (11.5, 12.5,   "12bb"),
    (12.5, float("inf"), "13bb"),
]

# Postflop: (lo_frac_inclusive, hi_frac_exclusive, label)
_POSTFLOP_BINS: list[tuple[float, float, str]] = [
    (0.0,   0.45,  "1/3 pot"),   # absorbs micro < 0.25x
    (0.45,  0.60,  "1/2 pot"),
    (0.60,  0.85,  "2/3 pot"),
    (0.85,  1.20,  "pot"),
    (1.20,  1.375, "1.25x pot"),
    (1.375, 1.625, "1.5x pot"),
    (1.625, float("inf"), "1.75x pot"),
]


# ── Conversion functions ──────────────────────────────────────────────────────

def raise_to_class(amt: float, pot: float, stack: float,
                   street: str, bb: float) -> int:
    """Map a raise amount (dollars) to a class index."""
    if stack > 0 and amt / stack >= _SHOVE_STACK:
        return LABEL_TO_CLASS["all-in"]

    if street == "Preflop":
        bb_amt = amt / bb
        for lo, hi, label in _PREFLOP_BINS:
            if lo <= bb_amt < hi:
                return LABEL_TO_CLASS[label]
        return LABEL_TO_CLASS["13bb"]

    if pot > 0:
        frac = amt / pot
        for lo, hi, label in _POSTFLOP_BINS:
            if lo <= frac < hi:
                return LABEL_TO_CLASS[label]
    return LABEL_TO_CLASS["pot"]


def action_to_class(action: str, pot: float, stack: float,
                    street: str, bb: float = 100.0) -> int:
    """Parse a raw action string and return its class index."""
    a = action.strip().lower()
    if a == "fold":
        return LABEL_TO_CLASS["fold"]
    if a == "check":
        return LABEL_TO_CLASS["check"]
    if a == "call":
        return LABEL_TO_CLASS["call"]
    if a.startswith("raise"):
        parts = a.split()
        if len(parts) >= 2:
            try:
                amt = float(parts[1].replace(",", ""))
                return raise_to_class(amt, pot, stack, street, bb)
            except ValueError:
                pass
    return LABEL_TO_CLASS["fold"]


def class_to_display(class_idx: int) -> str:
    """Human-readable label for a class index."""
    return CLASSES[class_idx]


def char_to_display(ch: str) -> str:
    """Convert a single output character to its human-readable label."""
    idx = CHAR_TO_CLASS.get(ch)
    return CLASSES[idx] if idx is not None else f"<unknown:{ch!r}>"
