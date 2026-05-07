# Shared game states used across all inference tests.
#
# Both are real decisions from the Pluribus dataset.

ALL_PLAYERS = ["Pluribus", "MrBlue", "MrOrange", "Bill", "MrPink", "Eddie", "MrWhite"]

# Preflop: BTN 67s facing HJ open. Easy fold for everyone.
PREFLOP_FOLD = """\
Street: Preflop
Position: BTN
Your hole cards: 6c 7s
Board: none
Pot: $360
Your stack: $10,000
To call: $210
Other active players: SB $9,950, BB $9,900, HJ $9,790
Action so far this street: UTG fold, HJ raise 210, CO fold
Your action?"""

PREFLOP_FOLD_LABEL = "fold"

# Flop: UTG KJo on 7s9cTc (open-ended straight draw). Action is raise.
# Label is raise 235; player sizing varies.
FLOP_RAISE = """\
Street: Flop
Position: UTG
Your hole cards: Kc Jh
Board: 7s 9c Tc
Pot: $470
Your stack: $9,790
To call: free (check)
Other active players: BB $9,790
Action so far this street: BB check
Your action?"""

FLOP_RAISE_LABEL = "raise 235"  # Pluribus label; others vary in sizing
