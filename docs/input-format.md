# Model Input Format

The model receives a chat-formatted prompt built by `make_input()` in
`scripts/parse_phh.py`.  The function accepts raw game parameters and
computes all derived fields automatically.

## System Message

```
You are {player_name}, playing 6-handed no-limit Texas Hold'em
($50/$100 blinds, $10,000 starting stacks).
Make the decision {player_name} would make.
Reply with exactly one of: ! | " | # | $ | % | & | ' | ( | ) | * | + | , | - | . | / | 0 | 1 | 2 | 3 | 4 | 5 | 6
```

The token list corresponds to the 22 action classes.  See `docs/action-classes.md`.

## User Message

```
Street: {Preflop|Flop|Turn|River}
Position: {SB|BB|UTG|HJ|CO|BTN}
Your hole cards: {e.g. Ah Kd}
Board: {e.g. Qh Jh 2c | none}
Board texture: {e.g. Unpaired, Two-tone, Connected}   ← postflop only
Your hand: {e.g. Ace-King offsuit | Two pair, aces and kings}
Your draws: {e.g. Flush draw (9 outs)}                ← when applicable
Board danger: {e.g. Flush possible (not in your hand)} ← when applicable
Pot: ${amount}
Your stack: ${amount}
SPR: {stack/pot, 1 decimal}
To call: ${amount | free (check)}
Other active players: {pos $stack, ...}
Action so far this street: {pos action, ... | first to act}
Your action?
```

### Field notes

**Board texture** (postflop only) — three comma-separated descriptors:
- Pairing: `Unpaired` / `Paired` / `Trips on board` / `Quads on board`
- Suits: `Rainbow` / `Two-tone` / `Flush possible`
- Connectivity: `Unconnected` / `Connected`

**Your hand** — preflop shows `{rank}-{rank} suited|offsuit` or `Pocket {ranks}`.
Postflop uses poker-eval's hand evaluator (pair, two pair, straight, flush, etc.).
Omitted if hole cards are unknown.

**Your draws** — present on flop and turn when hero has a flush draw (9 outs),
open-ended straight draw (8 outs), or gutshot (4 outs).  Omitted on the river.

**Board danger** — fires when the board creates a threat (flush draw, straight
draw, paired board enabling full house/quads) that hero is NOT part of.
Helps the model recognize spots where opponents may hold better hands.

**SPR (Stack-to-Pot Ratio)** — `stack / pot`.  Encodes commitment pressure:
- SPR < 4: commit-or-fold territory; thin value bets, all-ins standard
- SPR 4–13: standard postflop play
- SPR > 13: deep stack; implied odds and big-hand potential matter more

**Action history** — raises are annotated with the pot fraction at the time of
the raise, e.g. `BTN raise 300 (0.50x pot)`.  The pot shown is the pot
*before* the raise was made so the fraction matches how a player would
calculate it.

## `make_input()` Signature

```python
make_input(
    player_name:    str,
    street:         str,           # "Preflop" | "Flop" | "Turn" | "River"
    position:       str,           # "BTN", "SB", "BB", "UTG", "HJ", "CO"
    hole_cards:     list[str],     # ["Ah", "Kd"], or [] if unknown
    board:          list[str],     # ["Qh", "Jh", "2c"], or []
    pot:            float,
    stack:          float,
    to_call:        float,         # 0 if check option
    others:         list[tuple[str, float]],   # [(pos, stack), ...]
    street_actions: list[tuple[str, str]],     # [(pos, action_str), ...]
    bb:             float = 100.0,
) -> dict[str, str]               # {"system": ..., "user": ...}
```

Raise strings in `street_actions` may include pot-fraction annotations
(`"raise 500 (0.50x pot)"`).  When building from `decision_to_sft()` this
annotation is added automatically.

## Example Output (Flop, BTN)

```
Street: Flop
Position: BTN
Your hole cards: Ah Kh
Board: Qh Jh 2c
Board texture: Unpaired, Two-tone, Connected
Your hand: Ace high
Your draws: Flush draw (9 outs), Open-ended straight draw (8 outs)
Pot: $520
Your stack: $9,790
SPR: 18.8
To call: free (check)
Other active players: SB $9,480, BB $9,900
Action so far this street: first to act
Your action?
```

## Training Data Generation

`parse_phh.py` assembles these prompts from Pluribus hand histories via
`decision_to_sft()`, which calls `make_input()` and appends the ground-truth
action as the assistant turn (a single class character).

Run with:

```bash
python scripts/parse_phh.py --mode sft \
    --input /data/poker/phh-dataset/data/pluribus \
    --output data/sft_pluribus.jsonl
```
