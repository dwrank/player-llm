#!/usr/bin/env python3
"""
Parse PHH/PHHS poker hand history files into training datasets.

Two output modes:
  sft   -- one JSON line per player decision, with hole cards (Pluribus data)
  dapt  -- one compact text block per hand, no hole cards (HandHQ data)

Usage:
    # SFT dataset from all Pluribus hands
    python parse_phh.py --mode sft \\
        --input /data/poker/phh-dataset/data/pluribus \\
        --output ~/dev/poker/player-llm/data/sft_pluribus.jsonl

    # SFT for a single player only
    python parse_phh.py --mode sft \\
        --input /data/poker/phh-dataset/data/pluribus \\
        --output ~/dev/poker/player-llm/data/sft_pluribus.jsonl \\
        --players Pluribus MrBlue

    # DAPT dataset from high-stakes HandHQ hands
    python parse_phh.py --mode dapt \\
        --input /data/poker/phh-dataset/data/handhq \\
        --output ~/dev/poker/player-llm/data/dapt_handhq.txt \\
        --stakes 400NLH 600NLH 1000NLH \\
        --limit 500000
"""

import ast
import re
import json
import argparse
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Optional

from action_classes import (
    CLASS_TO_CHAR, CLASS_TOKENS_LIST,
    action_to_class, raise_to_class,
)

# ── Data classes ──────────────────────────────────────────────────────────────

@dataclass
class PHHHand:
    variant: str
    antes: list
    blinds: list           # blinds_or_straddles
    min_bet: float
    starting_stacks: list
    actions: list          # raw action strings from the file
    players: list
    finishing_stacks: list
    source: str = ""


@dataclass
class GameState:
    hand: PHHHand
    street: str            # Preflop / Flop / Turn / River
    board: list            # community cards dealt so far
    hole_cards: dict       # 0-based player index → "Ah Kd"
    stacks: list           # current stack per player
    pot: float
    contrib: list          # chips committed this street per player
    current_bet: float     # largest bet/raise total this street
    folded: set
    street_actions: list   # (player_idx, readable_str) for actions so far this street
    positions: dict        # 0-based player index → position name e.g. "BTN"


@dataclass
class Decision:
    player_idx: int
    player_name: str
    position: str
    state: GameState
    action: str            # normalized: fold / check / call / raise <amount>


# ── PHH parsing ───────────────────────────────────────────────────────────────

def _strip_comments(s: str) -> str:
    """Remove # comments that fall outside quoted strings."""
    lines = []
    for line in s.split("\n"):
        in_str, sc = False, None
        for i, c in enumerate(line):
            if in_str:
                if c == sc:
                    in_str = False
            else:
                if c in ('"', "'"):
                    in_str, sc = True, c
                elif c == "#":
                    line = line[:i]
                    break
        lines.append(line)
    return "\n".join(lines)


def _parse_block(text: str, source: str = "") -> Optional[PHHHand]:
    """Parse one TOML-ish hand block into a PHHHand."""

    def get_str(name: str) -> Optional[str]:
        m = re.search(rf"{name}\s*=\s*['\"]([^'\"]+)['\"]", text)
        return m.group(1) if m else None

    def get_float(name: str, default: float = 0.0) -> float:
        m = re.search(rf"{name}\s*=\s*([0-9.]+)", text)
        return float(m.group(1)) if m else default

    def get_list(name: str) -> list:
        # Match from 'name = [' to the first unmatched ']', across newlines
        m = re.search(rf"{name}\s*=\s*(\[.*?\])", text, re.DOTALL)
        if not m:
            return []
        try:
            return ast.literal_eval(_strip_comments(m.group(1)))
        except Exception:
            return []

    variant = get_str("variant")
    if not variant:
        return None

    hand = PHHHand(
        variant=variant,
        antes=get_list("antes"),
        blinds=get_list("blinds_or_straddles"),
        min_bet=get_float("min_bet"),
        starting_stacks=get_list("starting_stacks"),
        actions=get_list("actions"),
        players=get_list("players"),
        finishing_stacks=get_list("finishing_stacks"),
        source=source,
    )

    if not hand.players or not hand.actions or not hand.starting_stacks:
        return None

    return hand


def parse_phh_file(path: Path) -> list[PHHHand]:
    text = path.read_text(errors="replace")
    hand = _parse_block(text, str(path))
    return [hand] if hand else []


def parse_phhs_file(path: Path) -> list[PHHHand]:
    text = path.read_text(errors="replace")
    # Split on section headers like "[1]\n", "[42]\n" at start of line
    blocks = re.split(r"(?m)^\[(\d+)\]\n", text)
    # blocks = [preamble, idx, content, idx, content, ...]
    hands = []
    for i in range(1, len(blocks), 2):
        content = blocks[i + 1] if i + 1 < len(blocks) else ""
        hand = _parse_block(content, str(path))
        if hand:
            hands.append(hand)
    return hands


def iter_hands(paths: list[Path], nt_only: bool = True) -> Iterator[PHHHand]:
    """Yield PHHHand objects from a list of paths, skipping non-NLH if nt_only."""
    for path in paths:
        try:
            if path.suffix == ".phhs":
                hands = parse_phhs_file(path)
            else:
                hands = parse_phh_file(path)
        except Exception:
            continue

        for hand in hands:
            if nt_only and hand.variant != "NT":
                continue
            n = len(hand.players)
            if n < 2 or not hand.actions or len(hand.starting_stacks) < n:
                continue
            yield hand


# ── Position names ────────────────────────────────────────────────────────────

_POS_TEMPLATES: dict[int, list[str]] = {
    2:  ["SB", "BB"],
    3:  ["SB", "BB", "BTN"],
    4:  ["SB", "BB", "UTG", "BTN"],
    5:  ["SB", "BB", "UTG", "CO", "BTN"],
    6:  ["SB", "BB", "UTG", "HJ", "CO", "BTN"],
    7:  ["SB", "BB", "UTG", "UTG+1", "HJ", "CO", "BTN"],
    8:  ["SB", "BB", "UTG", "UTG+1", "UTG+2", "HJ", "CO", "BTN"],
    9:  ["SB", "BB", "UTG", "UTG+1", "UTG+2", "MP", "HJ", "CO", "BTN"],
    10: ["SB", "BB", "UTG", "UTG+1", "UTG+2", "MP", "MP+1", "HJ", "CO", "BTN"],
}


def get_positions(n: int, blinds: list) -> dict[int, str]:
    """Map 0-based player index → position name, based on blind postings."""
    sb_idx = next((i for i, b in enumerate(blinds[:n]) if b > 0), 0)
    template = _POS_TEMPLATES.get(n, [f"P{i+1}" for i in range(n)])
    return {(sb_idx + offset) % n: name for offset, name in enumerate(template)}


# ── Card formatting ───────────────────────────────────────────────────────────

def _fmt_cards(raw: str) -> str:
    """'TcQc' → 'Tc Qc', '7d5h9d' → '7d 5h 9d'"""
    return " ".join(re.findall(r"[2-9TJQKA][cdhs]|\?\?\?\?", raw))


# ── Game state replay ─────────────────────────────────────────────────────────

def _init_state(hand: PHHHand) -> tuple:
    """Return initial (stacks, pot, contrib, current_bet) from blind/ante postings."""
    n = len(hand.players)
    antes  = (hand.antes  + [0.0] * n)[:n]
    blinds = (hand.blinds + [0.0] * n)[:n]

    stacks = [
        hand.starting_stacks[i] - abs(antes[i]) - abs(blinds[i])
        for i in range(n)
    ]
    pot          = sum(abs(a) for a in antes) + sum(abs(b) for b in blinds)
    contrib      = [abs(b) for b in blinds]   # blinds count as preflop contribution
    current_bet  = max((abs(b) for b in blinds), default=0.0)
    return stacks, pot, contrib, current_bet


def iter_decisions(hand: PHHHand) -> Iterator[Decision]:
    """
    Replay a hand action-by-action. Yield one Decision per player action
    (fold / check-call / raise) where that player's hole cards are known.
    """
    n = len(hand.players)
    positions = get_positions(n, (hand.blinds + [0.0] * n)[:n])

    stacks, pot, contrib, current_bet = _init_state(hand)
    hole_cards: dict[int, str] = {}
    board: list[str] = []
    street = "Preflop"
    street_actions: list[tuple] = []
    folded: set[int] = set()

    for raw in hand.actions:
        parts = raw.split()
        if not parts:
            continue
        actor = parts[0]

        # ── Dealer actions ──────────────────────────────────────────────
        if actor == "d":
            if len(parts) < 3:
                continue
            if parts[1] == "dh":
                pidx = int(parts[2][1:]) - 1
                if 0 <= pidx < n and len(parts) > 3 and "????" not in parts[3]:
                    hole_cards[pidx] = _fmt_cards(parts[3])
            elif parts[1] == "db":
                board.extend(_fmt_cards(parts[2]).split())
                if   street == "Preflop": street = "Flop"
                elif street == "Flop":    street = "Turn"
                elif street == "Turn":    street = "River"
                contrib       = [0.0] * n
                current_bet   = 0.0
                street_actions = []
            continue

        # ── Player actions ──────────────────────────────────────────────
        if not (actor.startswith("p") and actor[1:].isdigit()):
            continue
        pidx = int(actor[1:]) - 1
        if not (0 <= pidx < n) or len(parts) < 2:
            continue

        verb = parts[1]
        if verb == "sm":   # showdown reveal, not a decision
            continue

        to_call = max(0.0, current_bet - contrib[pidx])

        # Snapshot state at this decision point
        if pidx in hole_cards:
            yield Decision(
                player_idx=pidx,
                player_name=hand.players[pidx],
                position=positions.get(pidx, f"P{pidx+1}"),
                state=GameState(
                    hand=hand,
                    street=street,
                    board=list(board),
                    hole_cards=dict(hole_cards),
                    stacks=list(stacks),
                    pot=pot,
                    contrib=list(contrib),
                    current_bet=current_bet,
                    folded=set(folded),
                    street_actions=list(street_actions),
                    positions=dict(positions),
                ),
                action=_normalize(verb, parts, to_call),
            )

        # Apply action to mutable state
        if verb == "f":
            folded.add(pidx)
            street_actions.append((pidx, "fold", pot))

        elif verb == "cc":
            pot_before     = pot
            added          = to_call
            stacks[pidx]  -= added
            pot           += added
            contrib[pidx]  = current_bet
            label = "check" if to_call == 0 else f"call {int(current_bet)}"
            street_actions.append((pidx, label, pot_before))

        elif verb == "cbr" and len(parts) > 2:
            pot_before     = pot
            amount         = float(parts[2])
            added          = amount - contrib[pidx]
            stacks[pidx]  -= added
            pot           += added
            contrib[pidx]  = amount
            current_bet    = amount
            street_actions.append((pidx, f"raise {int(amount)}", pot_before))


def _normalize(verb: str, parts: list, to_call: float) -> str:
    if verb == "f":
        return "fold"
    if verb == "cc":
        return "check" if to_call == 0 else "call"
    if verb == "cbr" and len(parts) > 2:
        # Keep one decimal if non-integer (cash games with fractional chips)
        amt = float(parts[2])
        return f"raise {amt:.0f}" if amt == int(amt) else f"raise {amt}"
    return verb


# ── SFT example formatting ────────────────────────────────────────────────────

_SYSTEM = (
    "You are {name}, playing 6-handed no-limit Texas Hold'em "
    "($50/$100 blinds, $10,000 starting stacks). "
    "Make the decision {name} would make. "
    "Reply with exactly one of: " + CLASS_TOKENS_LIST
)


# ── Hand features ─────────────────────────────────────────────────────────────

_POKER_EVAL_DIR = (
    Path(__file__).resolve().parents[3] / "poker-eval" / "poker-eval-cpp" / "python"
)

def _load_evaluate_hand():
    try:
        if str(_POKER_EVAL_DIR) not in sys.path:
            sys.path.insert(0, str(_POKER_EVAL_DIR))
        from poker_eval import evaluate_hand
        return evaluate_hand
    except (ImportError, OSError):
        return None

_evaluate_hand = _load_evaluate_hand()

_RANK_ORDER = "23456789TJQKA"
_RANK_FULL   = {
    '2': 'two',   '3': 'three', '4': 'four',  '5': 'five',
    '6': 'six',   '7': 'seven', '8': 'eight', '9': 'nine',
    'T': 'ten',   'J': 'jack',  'Q': 'queen', 'K': 'king',  'A': 'ace',
}
_RANK_PLURAL = {
    '2': 'twos',   '3': 'threes', '4': 'fours',  '5': 'fives',
    '6': 'sixes',  '7': 'sevens', '8': 'eights', '9': 'nines',
    'T': 'tens',   'J': 'jacks',  'Q': 'queens', 'K': 'kings', 'A': 'aces',
}


def _preflop_hand_desc(hole: list[str]) -> str:
    r1, s1 = hole[0][:-1], hole[0][-1]
    r2, s2 = hole[1][:-1], hole[1][-1]
    if r1 == r2:
        return f"Pocket {_RANK_PLURAL[r1]}"
    if _RANK_ORDER.index(r1) < _RANK_ORDER.index(r2):
        r1, r2 = r2, r1
    suited = "suited" if s1 == s2 else "offsuit"
    return f"{_RANK_FULL[r1].capitalize()}-{_RANK_FULL[r2]} {suited}"


def _postflop_hand_desc(hole: list[str], board: list[str]) -> str | None:
    if _evaluate_hand is None:
        return None
    try:
        _, desc = _evaluate_hand(hole + board)
        return desc
    except Exception:
        return None


def _board_texture(board: list[str]) -> str:
    ranks = [c[:-1] for c in board]
    suits = [c[-1]  for c in board]
    parts = []

    max_rank_count = max(Counter(ranks).values())
    if max_rank_count >= 4:
        parts.append("Quads on board")
    elif max_rank_count == 3:
        parts.append("Trips on board")
    elif max_rank_count == 2:
        parts.append("Paired")
    else:
        parts.append("Unpaired")

    max_suit_count = max(Counter(suits).values())
    if max_suit_count >= 3:
        parts.append("Flush possible")
    elif max_suit_count == 2:
        parts.append("Two-tone")
    else:
        parts.append("Rainbow")

    nums = sorted(set(_RANK_ORDER.index(r) for r in ranks if r in _RANK_ORDER))
    connected = (
        any(nums[i + 1] - nums[i] <= 3 for i in range(len(nums) - 1))
        if len(nums) >= 2 else False
    )
    parts.append("Connected" if connected else "Unconnected")

    return ", ".join(parts)


def _has_straight_potential(all_cards: list[str]) -> bool:
    """True if hero has a made straight or any straight draw (OESD or gutshot)."""
    present = set(_RANK_ORDER.index(c[:-1]) for c in all_cards if c[:-1] in _RANK_ORDER)
    if 12 in present:
        present = present | {-1}
    for start in range(-1, 9):
        if len(present & set(range(start, start + 5))) >= 4:
            return True
    return False


def _draw_desc(hole: list[str], board: list[str]) -> str | None:
    """Hero's flush and straight draws. None on the river (no cards to come)."""
    if len(board) == 5:
        return None

    all_cards = hole + board
    draws = []

    # Flush draw: exactly 4 of one suit (5+ = made flush, shown in hand desc)
    if any(v == 4 for v in Counter(c[-1] for c in all_cards).values()):
        draws.append("Flush draw (9 outs)")

    # Straight draws: scan every 5-rank window
    present = set(_RANK_ORDER.index(c[:-1]) for c in all_cards if c[:-1] in _RANK_ORDER)
    if 12 in present:
        present = present | {-1}

    oesd = gutshot = False
    for start in range(-1, 9):
        window = set(range(start, start + 5))
        n_in   = len(present & window)
        if n_in == 4:
            gap = list(window - present)[0] - start   # 0..4
            if gap in (0, 4):
                oesd = True
            else:
                gutshot = True

    if oesd:
        draws.append("Open-ended straight draw (8 outs)")
    elif gutshot:
        draws.append("Gutshot straight draw (4 outs)")

    return ", ".join(draws) if draws else None


def _board_danger_desc(hole: list[str], board: list[str]) -> str | None:
    """
    Flush/straight board threats that hero is NOT holding or drawing to.
    Only fires when the board creates a danger and the hero has no part of it.
    """
    all_cards = hole + board
    dangers   = []

    # Flush danger: board has 2+ suited and hero has fewer than 4 of that suit
    board_suit_counts = Counter(c[-1] for c in board)
    max_board_suit    = max(board_suit_counts.values())
    if max_board_suit >= 2:
        max_all_suit = max(Counter(c[-1] for c in all_cards).values())
        if max_all_suit < 4:   # hero doesn't have or draw to the flush
            label = "Flush possible" if max_board_suit >= 3 else "Flush draw possible"
            dangers.append(f"{label} (not in your hand)")

    # Straight danger: board is connected and hero has no straight or draw
    board_nums = sorted(
        set(_RANK_ORDER.index(c[:-1]) for c in board if c[:-1] in _RANK_ORDER)
    )
    board_connected = (
        any(board_nums[i + 1] - board_nums[i] <= 3 for i in range(len(board_nums) - 1))
        if len(board_nums) >= 2 else False
    )
    if board_connected and not _has_straight_potential(all_cards):
        dangers.append("Straight draw possible (not in your hand)")

    # Paired board danger: board has pair/trips and hero doesn't benefit from it
    board_rank_counts = Counter(c[:-1] for c in board)
    max_board_rank    = max(board_rank_counts.values())
    if max_board_rank >= 2 and _evaluate_hand is not None and hole:
        try:
            hand_val, _ = _evaluate_hand(hole + board)
            hero_cat    = (hand_val >> 24) & 0xFF
            # hero_cat: 0=HC 1=Pair 2=TwoPair 3=Trips 4=Straight 5=Flush 6=FH 7=Quads 8=SF
            if max_board_rank >= 3:
                # Board has trips — dangerous unless hero has quads
                if hero_cat < 7:
                    dangers.append("Trips on board, quads possible (not in your hand)")
            else:
                # Board has a pair — dangerous unless hero has full house or better
                if hero_cat < 6:
                    dangers.append("Paired board, full house possible (not in your hand)")
        except Exception:
            pass

    return ", ".join(dangers) if dangers else None


def _build_hand_lines(hole_list: list[str], board: list[str]) -> str:
    """Compute derived hand-feature lines for the user message."""
    if not hole_list:
        return ""
    if board:
        hand_desc   = _postflop_hand_desc(hole_list, board)
        texture_str = _board_texture(board)
        draws_str   = _draw_desc(hole_list, board)
        danger_str  = _board_danger_desc(hole_list, board)
    else:
        hand_desc   = _preflop_hand_desc(hole_list)
        texture_str = draws_str = danger_str = None

    lines = ""
    if texture_str:
        lines += f"Board texture: {texture_str}\n"
    if hand_desc:
        lines += f"Your hand: {hand_desc}\n"
    if draws_str:
        lines += f"Your draws: {draws_str}\n"
    if danger_str:
        lines += f"Board danger: {danger_str}\n"
    return lines


def make_input(
    player_name: str,
    street: str,
    position: str,
    hole_cards: list[str],
    board: list[str],
    pot: float,
    stack: float,
    to_call: float,
    others: list[tuple[str, float]],
    street_actions: list[tuple[str, str]],
    bb: float = 100.0,
) -> dict[str, str]:
    """
    Build model input messages from raw game parameters.

    All derived fields (board texture, hand strength, draws, board danger, SPR)
    are computed automatically from hole_cards, board, pot, and stack.

    Args:
        player_name:    Player name for the system prompt.
        street:         "Preflop" | "Flop" | "Turn" | "River"
        position:       Seat label, e.g. "BTN", "SB", "BB", "UTG", "HJ", "CO"
        hole_cards:     Hero's hole cards, e.g. ["Ah", "Kd"]. Empty if unknown.
        board:          Community cards dealt so far. Empty list for preflop.
        pot:            Current pot size in dollars.
        stack:          Hero's remaining stack in dollars.
        to_call:        Amount hero must call (0 if check option).
        others:         Active opponents as [(position_label, stack), ...].
        street_actions: Actions taken this street as [(position_label, action_str), ...].
                        Raise strings may include pot-fraction annotations, e.g.
                        "raise 500 (0.50x pot)".
        bb:             Big blind size in dollars (default 100).

    Returns:
        {"system": str, "user": str}  — ready for chat-template assembly.
    """
    board_str  = " ".join(board) if board else "none"
    call_str   = "free (check)" if to_call == 0 else f"${to_call:,.0f}"
    others_str = ", ".join(f"{pos} ${stk:,.0f}" for pos, stk in others) or "none"
    spr        = stack / pot if pot > 0 else 0.0

    if street_actions:
        history = ", ".join(f"{pos} {act}" for pos, act in street_actions)
    else:
        history = "first to act"

    hand_lines = _build_hand_lines(hole_cards, board)
    cards_str  = " ".join(hole_cards) if hole_cards else "?? ??"

    user_msg = (
        f"Street: {street}\n"
        f"Position: {position}\n"
        f"Your hole cards: {cards_str}\n"
        f"Board: {board_str}\n"
        f"{hand_lines}"
        f"Pot: ${pot:,.0f}\n"
        f"Your stack: ${stack:,.0f}\n"
        f"SPR: {spr:.1f}\n"
        f"To call: {call_str}\n"
        f"Other active players: {others_str}\n"
        f"Action so far this street: {history}\n"
        "Your action?"
    )

    return {
        "system": _SYSTEM.format(name=player_name),
        "user":   user_msg,
    }


def decision_to_sft(d: Decision, bb: float = 100.0) -> dict:
    s = d.state
    n = len(s.hand.players)

    cards   = s.hole_cards.get(d.player_idx, "??")
    to_call = max(0.0, s.current_bet - s.contrib[d.player_idx])

    others = [
        (s.positions.get(i, f"P{i+1}"), s.stacks[i])
        for i in range(n)
        if i != d.player_idx and i not in s.folded
    ]

    street_actions = []
    for item in (s.street_actions or []):
        pidx, act_str = item[0], item[1]
        pot_then = item[2] if len(item) > 2 else None
        pos = s.positions.get(pidx, f"P{pidx+1}")
        if act_str.startswith("raise") and pot_then and pot_then > 0:
            parts_a = act_str.split()
            if len(parts_a) >= 2:
                try:
                    amt  = float(parts_a[1].replace(",", ""))
                    frac = amt / pot_then
                    act_str = f"raise {int(amt):,} ({frac:.2f}x pot)"
                except ValueError:
                    pass
        street_actions.append((pos, act_str))

    msgs = make_input(
        player_name   = d.player_name,
        street        = s.street,
        position      = d.position,
        hole_cards    = cards.split() if "?" not in cards else [],
        board         = list(s.board),
        pot           = s.pot,
        stack         = s.stacks[d.player_idx],
        to_call       = to_call,
        others        = others,
        street_actions= street_actions,
        bb            = bb,
    )

    # Convert raw action to single class character
    cls  = action_to_class(d.action, s.pot, s.stacks[d.player_idx], s.street, bb)
    char = CLASS_TO_CHAR[cls]

    return {
        "messages": [
            {"role": "system",    "content": msgs["system"]},
            {"role": "user",      "content": msgs["user"]},
            {"role": "assistant", "content": char},
        ],
        "player": d.player_name,
        "street": s.street,
        "source": s.hand.source,
    }


# ── DAPT text formatting ──────────────────────────────────────────────────────

def hand_to_dapt(hand: PHHHand) -> str:
    """
    Serialize a hand as compact text for domain-adaptive pretraining.
    No hole cards. Uses position names and abbreviated actions.

    Example output:
        NT 6p bb=100 eff=10000
        PRE pot=150: UTG f | HJ r210 | CO f | BTN f | SB c | BB f
        FLP 7d 5h 9d pot=520: SB x | HJ x
        TRN 7c pot=520: SB x | HJ x
        RVR Qh pot=520: SB r230 | HJ f
    """
    n       = len(hand.players)
    blinds  = (hand.blinds + [0.0] * n)[:n]
    antes   = (hand.antes  + [0.0] * n)[:n]
    positions = get_positions(n, blinds)

    bb   = max((abs(b) for b in blinds), default=0)
    eff  = max(hand.starting_stacks[:n]) if hand.starting_stacks else 0
    header = f"NT {n}p bb={bb:.0f} eff={eff:.0f}"

    # Replay state
    stacks, pot, contrib, current_bet = _init_state(hand)
    board: list[str] = []
    street = "Preflop"

    # Collect per-street action tokens
    seqs: dict[str, list[str]] = {s: [] for s in ("Preflop", "Flop", "Turn", "River")}
    boards: dict[str, list[str]] = {}
    pots_at_street: dict[str, float] = {"Preflop": pot}

    for raw in hand.actions:
        parts = raw.split()
        if not parts:
            continue
        actor = parts[0]

        if actor == "d":
            if len(parts) < 3:
                continue
            if parts[1] == "db":
                new_cards = _fmt_cards(parts[2]).split()
                board.extend(new_cards)
                if   street == "Preflop": street = "Flop"
                elif street == "Flop":    street = "Turn"
                elif street == "Turn":    street = "River"
                boards[street] = new_cards
                contrib = [0.0] * n
                current_bet = 0.0
                pots_at_street[street] = pot
            continue

        if not (actor.startswith("p") and actor[1:].isdigit()):
            continue
        pidx = int(actor[1:]) - 1
        if not (0 <= pidx < n) or len(parts) < 2:
            continue

        verb = parts[1]
        if verb == "sm":
            continue

        pos     = positions.get(pidx, f"P{pidx+1}")
        to_call = max(0.0, current_bet - contrib[pidx])

        if verb == "f":
            seqs[street].append(f"{pos} f")
        elif verb == "cc":
            seqs[street].append(f"{pos} {'x' if to_call == 0 else 'c'}")
            contrib[pidx] = current_bet
            pot += to_call
            stacks[pidx] -= to_call
        elif verb == "cbr" and len(parts) > 2:
            amount = float(parts[2])
            added  = amount - contrib[pidx]
            seqs[street].append(f"{pos} r{amount:.0f}")
            pot          += added
            stacks[pidx] -= added
            contrib[pidx] = amount
            current_bet   = amount

    # Build output lines
    lines = [header]
    labels = {
        "Preflop": "PRE",
        "Flop":    "FLP",
        "Turn":    "TRN",
        "River":   "RVR",
    }
    for s in ("Preflop", "Flop", "Turn", "River"):
        acts = seqs[s]
        if not acts:
            continue
        board_str = " ".join(boards.get(s, []))
        pot_str   = f"pot={pots_at_street.get(s, 0):.0f}"
        label     = f"{labels[s]} {board_str}".strip()
        lines.append(f"{label} {pot_str}: {' | '.join(acts)}")

    return "\n".join(lines)


# ── File collection ───────────────────────────────────────────────────────────

def collect_paths(root: Path, stakes: list[str]) -> list[Path]:
    if root.is_file():
        return [root]
    paths = []
    for ext in ("*.phh", "*.phhs"):
        for p in root.rglob(ext):
            if stakes and not any(s in str(p) for s in stakes):
                continue
            paths.append(p)
    return sorted(paths)


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert PHH hand histories to LLM training data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--mode",    choices=["sft", "dapt"], required=True)
    parser.add_argument("--input",   type=Path, required=True)
    parser.add_argument("--output",  type=Path, required=True)
    parser.add_argument("--stakes",  nargs="*", default=[],
                        help="Filter paths by stake string, e.g. 1000NLH 600NLH")
    parser.add_argument("--players", nargs="*", default=[],
                        help="SFT only: restrict to these player names")
    parser.add_argument("--limit",   type=int, default=0,
                        help="Max examples to write (0 = no limit)")
    args = parser.parse_args()

    paths = collect_paths(args.input, args.stakes)
    print(f"Found {len(paths):,} input files", file=sys.stderr)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    count = 0

    with open(args.output, "w") as out:
        for hand in iter_hands(paths):
            if args.mode == "sft":
                for decision in iter_decisions(hand):
                    if args.players and decision.player_name not in args.players:
                        continue
                    ex   = decision_to_sft(decision)
                    line = json.dumps(ex) + "\n"
                    out.write(line)
                    count += 1
                    if args.limit and count >= args.limit:
                        break
                    # Oversample raise examples 2x to reduce fold bias
                    char = ex["messages"][-1]["content"]
                    if len(char) == 1 and ord(char) > ord('#'):
                        out.write(line)
                        count += 1
                        if args.limit and count >= args.limit:
                            break
            else:
                text = hand_to_dapt(hand)
                out.write(text + "\n\n")
                count += 1

            if count % 10_000 == 0 and count > 0:
                print(f"  {count:,} written...", file=sys.stderr)

            if args.limit and count >= args.limit:
                break

    print(f"Done: {count:,} examples → {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
