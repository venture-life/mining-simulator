#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

# Ensure project root is on sys.path for local package imports
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(THIS_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from honest_mining import simulate_mining_eventqV2


OUTPUT_DIR = "out"
TRACE_PATH = os.path.join(OUTPUT_DIR, "trace_run.json")
CONTEXT_PATH = os.path.join(OUTPUT_DIR, "trace_context.json")


def run_and_capture() -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Run the simulator with the requested parameters and return (trace, meta)."""
    # Parameters from the user request
    steps = 500
    groups = 5
    attacker_share = 0.4
    k = 3
    window = 5.0  # tau
    seed = 215
    work_shares = 16
    random_tie_break = False

    # CLI parity:
    # - Lambda_eff = rate * work_shares (default rate = 1/120)
    # - deterministic tie-break by default (unless --random-tie-break is requested)
    # - track_times enabled to match --track-times
    rate = 1.0 / 120.0
    Lambda_eff = rate * work_shares
    res = simulate_mining_eventqV2(
        steps=steps,
        groups=groups,
        shares=None,
        Lambda=Lambda_eff,
        D=window,
        max_prop_delay=2.5,
        k=k,
        deterministic_selection=(not random_tie_break),
        seed=seed,
        trace=True,
        trace_limit=None,
        attacker_share=attacker_share,
        selfish_policy=None,
        work_shares=work_shares,
    )

    trace = res.trace or []
    meta = res.to_dict()
    return trace, meta


def finde_sequence(trace: List[Dict[str, Any]], deliver_only: bool, miner_id: int, include_workshares: bool) -> Dict[str, Any]:
    """Return only filtered lists from the full trace (no slice indices or diagnosis).

    Behavior:
    - deliver_only=True:
      - deliver_context: all deliveries to `miner_id` across the full trace.
        - Types: ["DELIVER", "DELIVER_WS"] if include_workshares else ["DELIVER"].
      - mine_context: []

    - deliver_only=False:
      - deliver_context: []
      - mine_context: all mining events by `miner_id` across the full trace.
        - Types: ["MINE", "MINE_WS"] if include_workshares else ["MINE"].
    """
    # Event type filters for context (full-trace filtering)
    deliver_types = ("DELIVER", "DELIVER_WS") if include_workshares else ("DELIVER",)
    mine_types_ctx = ("MINE", "MINE_WS") if include_workshares else ("MINE",)

    # Build filtered lists only
    deliver_context: List[Dict[str, Any]] = []
    if deliver_only:
        deliver_context = [
            e for e in trace
            if (e.get("type") in deliver_types) and (int(e.get("to", -1)) == miner_id)
        ]

    mine_context: List[Dict[str, Any]] = []
    if not deliver_only:
        mine_context = [
            e for e in trace
            if (e.get("type") in mine_types_ctx) and (int(e.get("miner", -1)) == miner_id)
        ]

    return {
        "deliver_context": deliver_context,
        "mine_context": mine_context,
    }


def main() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    trace, meta = run_and_capture()

    # Save full trace for offline inspection
    with open(TRACE_PATH, "w", encoding="utf-8") as f:
        json.dump(trace, f, indent=2)
    print(f"Saved full trace to: {TRACE_PATH} with {len(trace)} events")

    # Find the requested sequence per miner; when deliver_only=True, filter to deliveries to miner[0]
    ctx = finde_sequence(trace, deliver_only=True, miner_id=5, include_workshares=False)

    # Save context slice (drop full trace from meta for compact context)
    meta_lite = dict(meta)
    meta_lite.pop("trace", None)
    with open(CONTEXT_PATH, "w", encoding="utf-8") as f:
        json.dump({
            "meta": meta_lite,
            "sequence_context": ctx,
        }, f, indent=2)
    print(f"Saved context slice to: {CONTEXT_PATH}")

    # Print a concise summary of counts
    print("\nContext summary:")
    print(json.dumps({
        "deliver_count": len(ctx.get("deliver_context", [])),
        "mine_count": len(ctx.get("mine_context", [])),
    }, indent=2))



if __name__ == "__main__":
    main()
