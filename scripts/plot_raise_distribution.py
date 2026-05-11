#!/usr/bin/env python3
"""
Analyse and plot raise sizing distributions from the SFT training data.

Preflop raises are expressed in BB multiples (rounded to nearest 0.5BB).
Postflop raises (flop/turn/river) are expressed as pot fractions.

Usage:
    python plot_raise_distribution.py
    python plot_raise_distribution.py --data /path/to/sft.jsonl --out raises.png
    python plot_raise_distribution.py --bb 100
"""

import argparse
import json
import re
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

# ── Config ────────────────────────────────────────────────────────────────────

DEFAULT_DATA = Path(__file__).parent.parent / "data" / "sft_pluribus.jsonl"
DEFAULT_OUT  = Path(__file__).parent.parent / "logs" / "raise_distribution.png"
DEFAULT_BB   = 100  # $100 big blind in Pluribus data

POSTFLOP_BINS = [
    ("micro",    0.00, 0.25),
    ("1/3 pot",  0.25, 0.45),
    ("1/2 pot",  0.45, 0.60),
    ("2/3 pot",  0.60, 0.85),
    ("pot",      0.85, 1.20),
    ("overbet",  1.20, 2.00),
    ("all-in",   2.00, float("inf")),
]
POSTFLOP_LABELS = [b[0] for b in POSTFLOP_BINS]
SHOVE_STACK     = 0.80  # raise ≥ 80% of stack → all-in regardless of pot fraction

STREETS          = ["Flop", "Turn", "River"]
POSTFLOP_COLORS  = ["#dd8452", "#55a868", "#c44e52"]

# ── Parsing ───────────────────────────────────────────────────────────────────

def parse_raises(path: Path, bb: int) -> tuple[list[dict], list[dict]]:
    """Return (preflop_raises, postflop_raises). Each entry is a dict."""
    preflop, postflop = [], []
    with open(path) as f:
        for line in f:
            ex   = json.loads(line)
            resp = ex["messages"][2]["content"]
            if not resp.startswith("raise"):
                continue
            msg = ex["messages"][1]["content"]

            pot_m    = re.search(r"Pot: \$([0-9,]+)", msg)
            amt_m    = re.search(r"raise (\d+)", resp)
            stack_m  = re.search(r"Your stack: \$([0-9,]+)", msg)
            street_m = re.search(r"Street: (\w+)", msg)

            if not (pot_m and amt_m and stack_m):
                continue

            pot    = int(pot_m.group(1).replace(",", ""))
            amt    = int(amt_m.group(1))
            stack  = int(stack_m.group(1).replace(",", ""))
            street = street_m.group(1) if street_m else "Unknown"

            if pot <= 0 or stack <= 0:
                continue

            stack_frac = amt / stack

            if street == "Preflop":
                bb_amt = round((amt / bb) * 2) / 2  # nearest 0.5BB
                preflop.append({"amt": amt, "bb": bb_amt, "stack_frac": stack_frac,
                                "stack": stack, "pot": pot})
            else:
                frac     = amt / pot
                all_in   = stack_frac >= SHOVE_STACK
                bin_name = "all-in" if all_in else POSTFLOP_LABELS[-1]
                if not all_in:
                    for label, lo, hi in POSTFLOP_BINS[:-1]:
                        if lo <= frac < hi:
                            bin_name = label
                            break
                postflop.append({"amt": amt, "frac": frac, "bin": bin_name,
                                 "street": street, "stack_frac": stack_frac})
    return preflop, postflop


# ── Preflop plot ──────────────────────────────────────────────────────────────

def plot_preflop(ax_hist: plt.Axes, ax_bar: plt.Axes, raises: list[dict]) -> None:
    bbs = [r["bb"] for r in raises]
    n   = len(raises)

    # Histogram: clip at 20BB, bin every 0.5BB
    clipped = [min(b, 20.5) for b in bbs]
    edges   = np.arange(0, 21.5, 0.5)
    ax_hist.hist(clipped, bins=edges, color="#4c72b0", edgecolor="white", linewidth=0.4)
    ax_hist.set_xlabel("Raise size (BB)")
    ax_hist.set_ylabel("Count")
    ax_hist.set_title(f"Preflop raise distribution  (n={n:,})")
    ax_hist.set_xlim(0, 21)
    ax_hist.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.0fBB"))

    # Annotate the top 3 peaks
    counts_by_bb = Counter(bbs)
    for bb_val, cnt in counts_by_bb.most_common(3):
        if bb_val <= 20:
            ax_hist.annotate(f"{bb_val:.1f}BB\n{100*cnt/n:.0f}%",
                             xy=(bb_val, cnt),
                             xytext=(bb_val + 0.8, cnt * 0.95),
                             fontsize=7, color="#4c72b0",
                             arrowprops=dict(arrowstyle="-", color="#4c72b0", lw=0.8))

    # Bar chart: group into named buckets
    buckets = [
        ("2BB",       lambda b: b == 2.0),
        ("2.5BB",     lambda b: b == 2.5),
        ("3BB",       lambda b: b == 3.0),
        ("3.5–5BB",   lambda b: 3.5 <= b <= 5.0),
        ("5.5–9BB",   lambda b: 5.5 <= b <= 9.0),
        ("9.5–15BB",  lambda b: 9.5 <= b <= 15.0),
        (">15BB",     lambda b: b > 15.0),
    ]
    labels = [b[0] for b in buckets]
    vals   = [sum(1 for b in bbs if fn(b)) for _, fn in buckets]
    pcts   = [100 * v / n for v in vals]

    bars = ax_bar.bar(labels, pcts, color="#4c72b0", edgecolor="white")
    ax_bar.set_ylabel("% of preflop raises")
    ax_bar.set_title("Preflop raises by bucket")
    ax_bar.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))
    ax_bar.tick_params(axis="x", labelrotation=20)
    for bar, pct in zip(bars, pcts):
        if pct > 0.5:
            ax_bar.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                        f"{pct:.1f}%", ha="center", va="bottom", fontsize=8)


# ── Postflop plot ─────────────────────────────────────────────────────────────

def plot_postflop(ax_hist: plt.Axes, ax_bar: plt.Axes, raises: list[dict]) -> None:
    fracs = [r["frac"] for r in raises]
    bins  = [r["bin"]  for r in raises]
    n     = len(raises)

    # Histogram: clip at 3x pot, bin boundaries at standard sizes
    clipped = [min(f, 3.0) for f in fracs]
    edges   = np.concatenate([
        np.arange(0, 0.25, 0.05),
        [0.25, 0.33, 0.45, 0.50, 0.60, 0.67, 0.75, 0.85, 1.00, 1.20, 1.50, 2.00, 3.0],
    ])
    edges = np.unique(np.sort(edges))
    ax_hist.hist(clipped, bins=edges, color="#55a868", edgecolor="white", linewidth=0.4)
    for _, lo, _ in POSTFLOP_BINS[1:]:
        if lo < 3.0:
            ax_hist.axvline(lo, color="red", linewidth=0.8, linestyle="--", alpha=0.5)
    ax_hist.set_xlabel("Raise amount / pot  (clipped at 3x)")
    ax_hist.set_ylabel("Count")
    ax_hist.set_title(f"Postflop raise distribution  (n={n:,})")
    ax_hist.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.2fx"))

    # Bar chart: by bin, split by street
    street_data   = {s: Counter(r["bin"] for r in raises if r["street"] == s) for s in STREETS}
    street_totals = {s: sum(c.values()) for s, c in street_data.items()}

    x     = np.arange(len(POSTFLOP_LABELS))
    width = 0.22
    for i, (street, color) in enumerate(zip(STREETS, POSTFLOP_COLORS)):
        total = street_totals[street] or 1
        pcts  = [100 * street_data[street].get(b, 0) / total for b in POSTFLOP_LABELS]
        bars  = ax_bar.bar(x + (i - 1) * width, pcts, width, label=street,
                           color=color, edgecolor="white")

    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(POSTFLOP_LABELS, rotation=20, ha="right")
    ax_bar.set_ylabel("% of postflop raises (per street)")
    ax_bar.set_title("Postflop raises by bucket and street")
    ax_bar.legend(fontsize=8)
    ax_bar.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))


# ── Summary ───────────────────────────────────────────────────────────────────

def summary(preflop: list[dict], postflop: list[dict]) -> None:
    print(f"\n── Preflop  (n={len(preflop):,}) ──────────────────────────")
    counts = Counter(r["bb"] for r in preflop)
    n = len(preflop)
    for bb, cnt in sorted(counts.items()):
        if 100 * cnt / n >= 0.5 or bb <= 4:
            print(f"  {bb:>5.1f}BB  {cnt:>5,}  {100*cnt/n:>5.1f}%")

    print(f"\n── Postflop (n={len(postflop):,}) ──────────────────────────")
    counts = Counter(r["bin"] for r in postflop)
    n = len(postflop)
    street_counts = {s: Counter(r["bin"] for r in postflop if r["street"] == s)
                     for s in STREETS}
    for label in POSTFLOP_LABELS:
        c   = counts.get(label, 0)
        pct = 100 * c / n
        scts = "  ".join(
            f"{s[0]}:{100*street_counts[s].get(label,0)/max(sum(street_counts[s].values()),1):.0f}%"
            for s in STREETS
        )
        print(f"  {label:<10} {c:>5,}  {pct:>5.1f}%   {scts}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", type=Path, default=DEFAULT_DATA)
    p.add_argument("--out",  type=Path, default=DEFAULT_OUT)
    p.add_argument("--bb",   type=int,  default=DEFAULT_BB,
                   help="Big blind size in dollars (default: 100)")
    args = p.parse_args()

    print(f"Loading {args.data} ...")
    preflop, postflop = parse_raises(args.data, args.bb)
    print(f"Preflop raises: {len(preflop):,}   Postflop raises: {len(postflop):,}")

    summary(preflop, postflop)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Raise sizing distribution — Preflop (BB) vs Postflop (pot fraction)",
                 fontsize=13, fontweight="bold")

    plot_preflop(axes[0, 0], axes[0, 1], preflop)
    plot_postflop(axes[1, 0], axes[1, 1], postflop)

    plt.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150)
    print(f"\nSaved → {args.out}")


if __name__ == "__main__":
    main()
