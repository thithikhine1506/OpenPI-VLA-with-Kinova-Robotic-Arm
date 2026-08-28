#!/usr/bin/env python3
"""
Log evaluation trials for the Gen3 policy.

Run this in a SECOND terminal, alongside gen3_policy_client.py in the first.
After each rollout, record what happened. Writes eval_log.csv.

    python3 log_trial.py --checkpoint 5000

Outcomes:
    s = success   block ended up in the bowl
    p = partial   grasped it but dropped it or missed the bowl
    f = failure   never got a grasp
    x = void      operator error, tunnel drop, etc -- excluded from the rate

Positions (a 3x3 grid over the workspace, from the camera's point of view):
    1 2 3     1=far-left   2=far-centre   3=far-right
    4 5 6     4=mid-left   5=mid-centre   6=mid-right
    7 8 9     7=near-left  8=near-centre  9=near-right
"""

import argparse
import csv
import datetime
import os
import sys

FIELDS = ["trial", "timestamp", "checkpoint", "position", "outcome", "note"]
OUTCOMES = {
    "s": "success",
    "p": "partial",
    "f": "failure",
    "x": "void",
}


def load(path):
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return list(csv.DictReader(f))


def summarize(rows):
    scored = [r for r in rows if r["outcome"] != "void"]
    if not scored:
        print("\nno scored trials yet")
        return

    n = len(scored)
    counts = {k: sum(1 for r in scored if r["outcome"] == v)
              for k, v in OUTCOMES.items() if k != "x"}

    print(f"\n{'=' * 46}")
    print(f"  {n} scored trials ({len(rows) - n} void)")
    print(f"{'=' * 46}")
    for k in ("s", "p", "f"):
        v = OUTCOMES[k]
        print(f"  {v:<10} {counts[k]:3d}   {100 * counts[k] / n:5.1f}%")
    print(f"  {'-' * 30}")
    print(f"  success rate           {100 * counts['s'] / n:5.1f}%")
    print(f"  grasp rate (s+p)       {100 * (counts['s'] + counts['p']) / n:5.1f}%")

    # by position -- shows which parts of the workspace the data covered
    by_pos = {}
    for r in scored:
        by_pos.setdefault(r["position"], []).append(r["outcome"])
    if len(by_pos) > 1:
        print(f"\n  by position:")
        for pos in sorted(by_pos):
            o = by_pos[pos]
            ok = sum(1 for x in o if x == "success")
            print(f"    pos {pos}:  {ok}/{len(o)}  ({100 * ok / len(o):.0f}%)")

    # by checkpoint -- for A/B comparison
    by_ckpt = {}
    for r in scored:
        by_ckpt.setdefault(r["checkpoint"], []).append(r["outcome"])
    if len(by_ckpt) > 1:
        print(f"\n  by checkpoint:")
        for c in sorted(by_ckpt):
            o = by_ckpt[c]
            ok = sum(1 for x in o if x == "success")
            print(f"    ckpt {c}:  {ok}/{len(o)}  ({100 * ok / len(o):.0f}%)")
    print()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="5000")
    ap.add_argument("--out", default="eval_log.csv")
    ap.add_argument("--summary", action="store_true",
                    help="print the summary and exit")
    args = ap.parse_args()

    rows = load(args.out)

    if args.summary:
        summarize(rows)
        return

    n = len(rows)
    print(f"logging to {args.out}   checkpoint {args.checkpoint}   "
          f"{n} trials already recorded")
    print(__doc__.split("Outcomes:")[1].split('"""')[0])
    print("Enter blank outcome to finish.\n")

    new = []
    try:
        while True:
            n += 1
            pos = input(f"[trial {n}] block position (1-9): ").strip()
            if not pos:
                n -= 1
                break
            out = input(f"[trial {n}] outcome (s/p/f/x): ").strip().lower()
            if not out:
                n -= 1
                break
            if out not in OUTCOMES:
                print(f"  ? expected one of {list(OUTCOMES)} -- not recorded")
                n -= 1
                continue
            note = input(f"[trial {n}] note (optional): ").strip()

            new.append({
                "trial": n,
                "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
                "checkpoint": args.checkpoint,
                "position": pos,
                "outcome": OUTCOMES[out],
                "note": note,
            })
            print(f"  -> {OUTCOMES[out]}\n")
    except (KeyboardInterrupt, EOFError):
        print()

    if new:
        write_header = not os.path.exists(args.out)
        with open(args.out, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=FIELDS)
            if write_header:
                w.writeheader()
            w.writerows(new)
        print(f"wrote {len(new)} trials to {args.out}")

    summarize(load(args.out))


if __name__ == "__main__":
    main()
