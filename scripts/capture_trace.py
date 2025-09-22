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
    steps = 400
    groups = 5
    attacker_share = 0.4
    k = 1
    window = 0.0  # tau
    seed = 215

    # We explicitly set deterministic_selection=False to match --random-tie-break
    res = simulate_mining_eventqV2(
        steps=steps,
        groups=groups,
        shares=None,
        Lambda=1.0 / 120.0,
        D=window,
        max_prop_delay=2.5,
        k=k,
        deterministic_selection=False,
        seed=seed,
        track_times=False,
        trace=True,
        trace_limit=None,
        attacker_share=attacker_share,
        selfish_policy=None,
    )

    trace = res.trace or []
    meta = res.to_dict()
    return trace, meta


def find_sequence(trace: List[Dict[str, Any]], deliver_height: int = 97, miner_id: int = 0) -> Dict[str, Any]:
    """Find the first DELIVER to miner `miner_id` at height `deliver_height` and the next MINE by that miner.

    Returns a dictionary with indices and event snippets for context.
    """
    deliver_idx: Optional[int] = None
    for i, ev in enumerate(trace):
        if ev.get("type") == "DELIVER" and int(ev.get("to", -1)) == miner_id:
            try:
                h = int(ev.get("height"))
            except Exception:
                continue
            if h == deliver_height:
                deliver_idx = i
                break

    mine_idx: Optional[int] = None
    if deliver_idx is not None:
        for j in range(deliver_idx + 1, len(trace)):
            ev = trace[j]
            if ev.get("type") == "MINE" and int(ev.get("miner", -1)) == miner_id:
                mine_idx = j
                break

    # Prepare context slices around each index
    ctx: Dict[str, Any] = {
        "deliver_idx": deliver_idx,
        "mine_idx": mine_idx,
        "deliver_event": (trace[deliver_idx] if deliver_idx is not None else None),
        "mine_event": (trace[mine_idx] if mine_idx is not None else None),
        "deliver_context": [],
        "mine_context": [],
    }

    def slice_ctx(idx: int, radius: int = 8) -> List[Dict[str, Any]]:
        lo = max(0, idx - radius)
        hi = min(len(trace), idx + radius + 1)
        return trace[lo:hi]

    if deliver_idx is not None:
        ctx["deliver_context"] = slice_ctx(deliver_idx)
    if mine_idx is not None:
        ctx["mine_context"] = slice_ctx(mine_idx)

    # Add a short diagnosis if we can compute parent height from the MINE event
    diagnosis: Dict[str, Any] = {}
    if mine_idx is not None:
        me = trace[mine_idx]
        try:
            new_h = int(me.get("height"))
            parent_h = new_h - 1
        except Exception:
            new_h = None
            parent_h = None
        diagnosis.update({
            "t_mine": me.get("t"),
            "new_block_height": new_h,
            "parent_id": me.get("parent_id"),
            "inferred_parent_height": parent_h,
        })
    if deliver_idx is not None:
        de = trace[deliver_idx]
        diagnosis.update({
            "t_deliver": de.get("t_deliver"),
            "delivered_block_id": de.get("block_id"),
            "delivered_height": de.get("height"),
            "delivered_parent_id": de.get("parent_id"),
        })
    ctx["diagnosis"] = diagnosis

    return ctx


def main() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    trace, meta = run_and_capture()

    # Save full trace for offline inspection
    with open(TRACE_PATH, "w", encoding="utf-8") as f:
        json.dump(trace, f, indent=2)
    print(f"Saved full trace to: {TRACE_PATH} with {len(trace)} events")

    # Find the requested sequence: miner[0] receives height 97, then next MINE by miner[0]
    ctx = find_sequence(trace, deliver_height=97, miner_id=0)

    # Save context slice
    with open(CONTEXT_PATH, "w", encoding="utf-8") as f:
        json.dump({
            "meta": meta,
            "sequence_context": ctx,
        }, f, indent=2)
    print(f"Saved context slice to: {CONTEXT_PATH}")

    # Print a concise diagnosis to the console
    diag = ctx.get("diagnosis", {})
    print("\nDiagnosis (concise):")
    print(json.dumps(diag, indent=2))
    if ctx.get("deliver_idx") is None:
        print("Note: No DELIVER to miner 0 at height 97 was found in this run.")
    elif ctx.get("mine_idx") is None:
        print("Note: No subsequent MINE by miner 0 after the height-97 delivery was found.")


if __name__ == "__main__":
    main()
