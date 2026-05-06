# Data Format: PHH to Training Examples

## PHH File Structure (reference)

```toml
variant = 'NT'                          # NT = No-Limit Texas Hold'em
ante_trimming_status = true
antes = [0, 0, 0, 0, 0, 0]
blinds_or_straddles = [50, 100, 0, 0, 0, 0]
min_bet = 100
starting_stacks = [10000, 10000, 10000, 10000, 10000, 10000]
actions = [
  'd dh p1 TcQc',   # deal hole cards to player 1
  'd dh p2 8s4c',
  ...
  'p3 f',           # player 3 folds
  'p4 cbr 210',     # player 4 raises to 210
  'p5 f',
  'p1 cc',          # player 1 calls/checks
  'd db 7d5h9d',    # deal board (flop)
  'p1 cbr 230',     # player 1 bets 230
  'p4 f',
]
players = ['MrBlue', 'MrBlonde', 'MrWhite', 'MrPink', 'MrBrown', 'Pluribus']
finishing_stacks = [10310, 9900, 10000, 9790, 10000, 10000]
```

### Action tokens
| Token | Meaning |
|-------|---------|
| `d dh pN CARDS` | Deal hole cards to player N |
| `d db CARDS` | Deal board cards |
| `pN f` | Player N folds |
| `pN cc` | Player N checks or calls |
| `pN cbr AMOUNT` | Player N raises/bets to AMOUNT (total, not additional) |
| `pN sm CARDS` | Player N shows cards at showdown |

---

## Stage 0 Format (DAPT — no hole cards needed)

Serialize each hand as a single text string. Omit hole cards entirely.
One hand = one training document.

```
NT 6max blinds=50/100 stacks=10000/10000/10000/10000/10000/10000
PREFLOP: p3 f | p4 cbr 210 | p5 f | p6 f | p1 cc | p2 f
FLOP 7d5h9d: p1 cc | p4 cc
TURN 7c: p1 cc | p4 cc
RIVER Qh: p1 cbr 230 | p4 f
```

Keep it compact. The LM loss applies to every token — the model learns
the grammar of poker actions, bet sizes, and board sequences.

---

## Stage 1 Format (SFT — instruction tuning with hole cards)

One training example per player decision. For each action `pN ACTION` in the
actions list, reconstruct the full game state visible to player N at that moment
and format it as a chat turn.

### Position mapping

Seat order rotates each hand. Derive position from seat relative to the button:
- Seat with blind = 50 → Small Blind (SB)
- Seat with blind = 100 → Big Blind (BB)
- Next seat UTG (Under the Gun), then UTG+1, HJ (Hijack), CO (Cutoff), BTN (Button)

In 6-max: p1=SB, p2=BB, p3=UTG, p4=HJ, p5=CO, p6=BTN

### Game state at decision point

Track incrementally through the action list:
- `pot` = sum of all contributions so far
- `to_call` = largest current bet minus player's contribution this street
- `street` = Preflop / Flop / Turn / River (increments on `d db`)
- `board` = community cards dealt so far
- `active_players` = players who have not folded
- `stacks` = starting_stacks minus contributions

### Training example (chat format)

```json
{
  "messages": [
    {
      "role": "system",
      "content": "You are MrBlue, playing 6-handed no-limit Texas Hold'em ($50/$100). Make decisions as MrBlue would. Reply with exactly one action: fold, check, call, or raise <amount>."
    },
    {
      "role": "user",
      "content": "Street: Preflop\nPosition: Seat 1/6 (SB)\nYour hole cards: TcQc\nBoard: (none)\nPot: $150\nYour stack: $9,950\nTo call: $50\nAction so far: p3 fold, p4 raise $210, p5 fold, p6 fold\nYour action?"
    },
    {
      "role": "assistant",
      "content": "call"
    }
  ]
}
```

### Action normalization

Normalize the assistant response to one of:
- `fold`
- `check`
- `call`
- `raise <amount>` (total chips going in, same convention as PHH `cbr`)

Map PHH actions:
- `pN f` → `fold`
- `pN cc` with to_call=0 → `check`
- `pN cc` with to_call>0 → `call`
- `pN cbr AMOUNT` → `raise <AMOUNT>`

### What to skip

- `d dh` and `d db` actions — these are dealer actions, not player decisions
- `pN sm` showdown actions — not decisions
- Decisions by players whose hole cards are `????` — can't train with unknown cards

---

## Stage 2 Format (per-player LoRA)

Identical format to Stage 1. Filter the dataset to only include decisions
made by the target player.

Example: for the Pluribus adapter, keep only rows where the acting player
is `Pluribus`.

---

## Dataset Split

Before any training, split per player:
- 90% train
- 10% held-out evaluation

Split by hand (not by individual decision) to avoid data leakage —
all decisions from a given hand stay in the same split.

---

## Estimated Dataset Sizes

| Stage | Hands | Examples | Tokens (est.) |
|-------|-------|----------|---------------|
| Stage 0 DAPT | 500K–1M HandHQ | 500K–1M sequences | ~200M–400M |
| Stage 1 SFT | 10,000 Pluribus + 12 pro | ~91,400 | ~18M |
| Stage 2 LoRA (per player) | varies | 800–15,000 | varies |
