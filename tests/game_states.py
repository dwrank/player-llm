# Shared game states used across all inference tests.
#
# Both are real decisions from the Pluribus dataset, formatted exactly as
# make_input() produces (including SPR, board texture, pot-fraction history).

ALL_PLAYERS = ["Pluribus", "MrBlue", "MrOrange", "Bill", "MrPink", "Eddie", "MrWhite"]

# Preflop: BTN 67s facing HJ open. Easy fold for everyone.
PREFLOP_FOLD = """\
Street: Preflop
Position: BTN
Your hole cards: 6c 7s
Board: none
Your hand: Seven-six offsuit
Pot: $360
Your stack: $10,000
SPR: 27.8
To call: $210
Other active players: SB $9,950, BB $9,900, HJ $9,790
Action so far this street: UTG fold, HJ raise 210 (1.40x pot), CO fold
Your action?"""

PREFLOP_FOLD_LABEL = "fold"

# Flop: UTG KJo on 7s9cTc (gutshot, board danger). Action is raise.
# Pluribus raises ~235 into $470 pot = 0.5x → "1/2 pot" class.
FLOP_RAISE = """\
Street: Flop
Position: UTG
Your hole cards: Kc Jh
Board: 7s 9c Tc
Board texture: Unpaired, Two-tone, Connected
Your hand: King high
Your draws: Gutshot straight draw (4 outs)
Board danger: Flush draw possible (not in your hand)
Pot: $470
Your stack: $9,790
SPR: 20.8
To call: free (check)
Other active players: BB $9,790
Action so far this street: BB check
Your action?"""

FLOP_RAISE_LABEL = "1/2 pot"   # Pluribus; other players vary in class
