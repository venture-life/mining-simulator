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
        track_times=True,
        time_bins=50,
        trace=True,
        trace_limit=None,
        attacker_share=attacker_share,
        selfish_policy=None,
        work_shares=work_shares,
    )

    trace = res.trace or []
    meta = res.to_dict()
    return trace, meta


def finde_sequence(trace: List[Dict[str, Any]], deliver_only: bool, miner_id: int) -> Dict[str, Any]:
    """Build a miner-specific event context from the trace.

    - When deliver_only is True:
      deliver_context = all events of type {DELIVER, DELIVER_WS} with `to == miner_id` across the entire trace;
      mine_context = [].

    - When deliver_only is False:
      deliver_context = a local slice around the first delivery (DELIVER or DELIVER_WS) to `miner_id`;
      mine_context = the next MINE by `miner_id` after that delivery, with a local slice for context when present.
    """
    deliver_idx: Optional[int] = None
    for i, ev in enumerate(trace):
        if ev.get("type") in ("DELIVER", "DELIVER_WS") and int(ev.get("to", -1)) == miner_id:
            deliver_idx = i
            break

    mine_idx: Optional[int] = None
    if (deliver_idx is not None) and (not deliver_only):
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
        if deliver_only:
            # Collect the entire filtered list of deliveries to miner_id across the whole trace
            ctx["deliver_context"] = [
                e for e in trace
                if (e.get("type") in ("DELIVER", "DELIVER_WS")) and (int(e.get("to", -1)) == miner_id)
            ]
        else:
            ctx["deliver_context"] = slice_ctx(deliver_idx)
    if (mine_idx is not None) and (not deliver_only):
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
            "ws_included_mine": me.get("ws_included"),
        })
    if deliver_idx is not None:
        de = trace[deliver_idx]
        diagnosis.update({
            "t_deliver": de.get("t_deliver"),
            "delivered_block_id": de.get("block_id"),
            "delivered_height": de.get("height"),
            "delivered_parent_id": de.get("parent_id"),
            "ws_included_deliver": de.get("ws_included"),
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

    # Find the requested sequence per miner; when deliver_only=True, filter to deliveries to miner[0]
    ctx = finde_sequence(trace, deliver_only=True, miner_id=5)

    # Save context slice (drop full trace from meta for compact context)
    meta_lite = dict(meta)
    meta_lite.pop("trace", None)
    with open(CONTEXT_PATH, "w", encoding="utf-8") as f:
        json.dump({
            "meta": meta_lite,
            "sequence_context": ctx,
        }, f, indent=2)
    print(f"Saved context slice to: {CONTEXT_PATH}")

    # Print a concise diagnosis to the console
    diag = ctx.get("diagnosis", {})
    print("\nDiagnosis (concise):")
    print(json.dumps(diag, indent=2))



if __name__ == "__main__":
    main()
