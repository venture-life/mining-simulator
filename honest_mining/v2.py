from __future__ import annotations

from typing import List, Optional, Dict
import heapq
import math

try:
    import numpy as np
except ImportError as e:
    raise SystemExit(
        "numpy is required. Please install dependencies: `pip install -r requirements.txt`"
    ) from e

from .simulator import HonestEventqResult
from .miner import Miner, Block
from .selfish_miner import SelfishMiner


def simulate_mining_eventqV2(
    steps: int,
    *,
    groups: int = 3,
    shares: Optional[List[float]] = None,
    Lambda: float = 1.0 / 60.0,
    D: float = 5.0,
    k: int = 3,
    seed: Optional[int] = None,
    track_times: bool = False,
    time_bins: int = 50,
    trace: bool = False,
    trace_limit: Optional[int] = None,
    attacker_share: Optional[float] = None,
) -> HonestEventqResult:
    """
    Continuous-time simulation using a single global Poisson process of block arrivals with rate Λ.
    Each arrival is assigned to a miner via thinning (probabilities 'shares'), and that miner mints
    a block at that event time. The block is then gossiped to other miners via random per-delivery
    delays with parent repair to ensure parents arrive before children.

    By default (attacker_share is None), the model is honest-only with 'groups' miners.
    If attacker_share in (0,1) is provided, we instantiate one SelfishMiner as the LAST miner and
    allocate shares as follows: if 'shares' is provided, treat them as RELATIVE shares for the honest
    groups and rescale them to sum to (1 - attacker_share); otherwise, allocate honest miners uniformly
    with (1 - attacker_share)/groups each. The attacker gets attacker_share. Total number of miners
    becomes G = groups + 1 in that case.

    Local in-time classification and chain-weight selection are handled by the Miner logic. We set
    the in-time window τ = D, use fork-resolution parameter k for longest-chain dominance, and
    model propagation with random per-delivery delays capped at D/2 (interpreting D ≈ 2×MAX_PROPAGATION_TIME).

    The simulation runs until the canonical chain (as seen by miner 0) reaches 'steps' blocks beyond
    genesis. At the end, we compute summary metrics from miner 0's local view.

    If track_times=True, we compute the first rival timing per height from miner 0's view and produce
    a histogram with 'time_bins' bins over [0, D/2].
    """
    if steps <= 0:
        raise ValueError("steps must be > 0")
    if groups <= 0:
        raise ValueError("groups must be > 0")
    if Lambda <= 0.0:
        raise ValueError("Lambda must be > 0")
    if D < 0.0:
        raise ValueError("D must be >= 0")

    # Configure miners and shares
    miners: List[Miner]  # SelfishMiner also conforms to Miner-like API
    attacker_idx: Optional[int] = None
    if attacker_share is None:
        # Honest-only: shares default equal
        if shares is None:
            shares = [1.0 / groups] * groups
        if len(shares) != groups:
            raise ValueError("shares length must equal groups")
        if any(s < 0 for s in shares):
            raise ValueError("shares must be non-negative")
        total_share = float(sum(shares))
        if total_share <= 0.0:
            raise ValueError("sum(shares) must be > 0")
        shares = [float(s) / total_share for s in shares]
        miners = [Miner(miner_id=i, k=k, tau=D) for i in range(groups)]
    else:
        a = float(attacker_share)
        if not (0.0 < a < 1.0):
            raise ValueError("attacker_share must be in (0,1)")
        if groups <= 0:
            raise ValueError("groups must be > 0 when attacker_share is set (number of honest groups)")
        # Honest shares: if provided, rescale to sum to (1 - a); else uniform over 'groups'
        if shares is None:
            honest_share = (1.0 - a) / float(groups)
            honest_shares = [honest_share for _ in range(groups)]
        else:
            if len(shares) != groups:
                raise ValueError("shares length must equal groups (honest groups) when attacker_share is set")
            if any(s < 0 for s in shares):
                raise ValueError("shares must be non-negative")
            total_share = float(sum(shares))
            if total_share <= 0.0:
                raise ValueError("sum(shares) must be > 0")
            norm = [float(s) / total_share for s in shares]
            honest_shares = [(1.0 - a) * p for p in norm]
        # Append attacker as last index
        shares = honest_shares + [a]
        attacker_idx = groups  # attacker is last
        miners = [Miner(miner_id=i, k=k, tau=D) for i in range(groups)] + [SelfishMiner(miner_id=attacker_idx, k=k, tau=D, alpha=a)]

    max_prop_delay = 0.5 * D
    if track_times:
        if time_bins <= 0:
            raise ValueError("time_bins must be > 0 when track_times=True")
        bins = int(time_bins)
        first_rival_count = 0
        first_rival_sum = 0.0
        first_rival_hist = [0 for _ in range(bins)]

    rng = np.random.default_rng(seed)

    # Network delay sampling (right-skew) and parent-repair parameters
    # - sample_delay(): lognormal with mean ~= max_prop_delay/2, capped at max_prop_delay (0..D/2)
    # - T_REQ: small request/response overhead per missing ancestor during repair
    def sample_delay() -> float:
        if max_prop_delay <= 0.0:
            return 0.0
        sigma = 0.6  # right-skew
        mu = math.log(max(max_prop_delay / 2.0, 1e-9)) - 0.5 * sigma * sigma
        d = float(rng.lognormal(mean=mu, sigma=sigma))
        return d if d <= max_prop_delay else max_prop_delay

    T_REQ = 0.1  # seconds

    # Global block store to enable ancestor fetch during parent repair
    block_store: Dict[str, Block] = {}

    # Optional event trace for visualization
    trace_events: List[Dict] = [] if trace else None  # type: ignore[assignment]
    def _trace_append(ev: Dict) -> None:
        if trace_events is not None:
            # Enforce optional limit on trace length
            if trace_limit is not None and len(trace_events) >= trace_limit:
                return
            trace_events.append(ev)

    # Helper: create a fresh Block instance for delivery to a miner to avoid shared mutation
    def _clone_block(b: Block) -> Block:
        return Block(
            id=b.id,
            parent_id=b.parent_id,
            miner_id=b.miner_id,
            height=0,  # will be set by receiver based on local parent
            uncles=list(b.uncles),
            created_time=b.created_time,
        )

    # Helper: broadcast a freshly mined (or newly published) block from miner gi at time t
    def _broadcast_block(blk: Block, gi: int, t: float) -> None:
        # Record globally for potential ancestor fetches
        block_store[blk.id] = blk
        # Schedule deliveries to all other miners with random per-delivery delay (ordered per miner)
        G = len(miners)
        if G > 1:
            for m in miners:
                if m.miner_id == gi:
                    continue
                delay = sample_delay()
                _push_event(t + delay, 0, (m.miner_id, blk, False))
    # miners already initialized above

    # Event queue: (time, kind, seq, payload)
    # kind: 0 = DELIVER, 1 = MINE (DELIVERs at same time process first).
    # seq is a monotonically increasing tie-breaker to avoid comparing payloads (e.g., Blocks).
    # payload for DELIVER: (miner_id, block, repair_flag)
    events: List[tuple] = []
    seq = 0
    def _push_event(ev_t: float, ev_kind: int, ev_payload) -> None:
        nonlocal seq
        seq += 1
        heapq.heappush(events, (ev_t, ev_kind, seq, ev_payload))
    t = 0.0
    mine_events = 0
    # Seed first global mining event
    _push_event(t + float(rng.exponential(1.0 / Lambda)), 1, None)

    while events:
        t, kind, _, payload = heapq.heappop(events)
        if kind == 1:  # MINE
            # Assign to group via thinning
            G = len(miners)
            gi = int(rng.choice(G, p=shares))
            # Winning miner mints immediately at time t
            new_block = miners[gi].on_mine(now=t)
            # Count mining events
            mine_events += 1
            # If the miner returned a block, we broadcast it immediately (honest-compatible).
            if isinstance(new_block, Block):
                if trace:
                    _trace_append({
                        "type": "MINE",
                        "t": t,
                        "miner": gi,
                        "block_id": new_block.id,
                        "parent_id": new_block.parent_id,
                        "height": new_block.height,
                        "weight": miners[gi].cum_block_weight.get(new_block.id, None),
                        "uncles": list(new_block.uncles) if getattr(new_block, "uncles", None) else [],
                    })
                _broadcast_block(new_block, gi, t)
            else:
                # SelfishMiner withholds: still emit a MINE trace so origin lane shows the box at mine time
                if trace and isinstance(miners[gi], SelfishMiner):
                    try:
                        wlist = getattr(miners[gi], "_withheld", None)
                        if wlist:
                            b = wlist[-1]
                            _trace_append({
                                "type": "MINE",
                                "t": t,
                                "miner": gi,
                                "block_id": b.id,
                                "parent_id": b.parent_id,
                                "height": b.height,
                                # Use PRIVATE view weight at mine time
                                "weight": getattr(miners[gi], "private").cum_block_weight.get(b.id, None),
                                "uncles": list(getattr(b, "uncles", []) or []),
                            })
                    except Exception:
                        pass
            # Policy hook (only for the miner that mined)
            decide = getattr(miners[gi], "decide_action", None)
            act = getattr(miners[gi], "act", None)
            if callable(decide) and callable(act):
                try:
                    action = decide(t)
                    to_publish = act(action, t)
                    if to_publish:
                        for pb in to_publish:
                            _broadcast_block(pb, gi, t)
                except Exception:
                    # Keep simulator running even if a custom policy misbehaves
                    pass
            # Schedule next global mining arrival
            _push_event(t + float(rng.exponential(1.0 / Lambda)), 1, None)
        else:  # DELIVER
            mid, blk, repair = payload
            m = miners[mid]
            # Network-level parent repair: ensure parents are known before delivering child
            if blk.parent_id is not None and blk.parent_id not in m.blocks:
                # Identify missing ancestor chain up to the nearest known ancestor
                missing: List[str] = []
                pid = blk.parent_id
                while pid is not None and pid not in m.blocks:
                    missing.append(pid)
                    pb = block_store.get(pid)
                    if pb is None:
                        break
                    # type: ignore[attr-defined]
                    pid = getattr(pb, "parent_id", None)  # walk up using stored blocks

                # Schedule deliveries for missing ancestors in order, each incurring T_REQ + delay
                t_cur = t
                for bid in reversed(missing):
                    pb = block_store.get(bid)
                    if pb is None:
                        continue
                    t_cur = t_cur + T_REQ + sample_delay()
                    _push_event(t_cur, 0, (mid, pb, True))

                # Reschedule the child to arrive after its parents
                epsilon = 1e-9
                t_child = max(t, t_cur + epsilon)
                _push_event(t_child, 0, (mid, blk, False))
            else:
                # Deliver a fresh clone to avoid cross-miner state contamination
                m.on_receive(_clone_block(blk), received_time=t)
                if trace:
                    _trace_append({
                        "type": "DELIVER",
                        "t_mine": blk.created_time,
                        "t_deliver": t,
                        "from": int(blk.miner_id) if blk.miner_id is not None else None,
                        "to": mid,
                        "block_id": blk.id,
                        "parent_id": blk.parent_id,
                        "repair": bool(repair),
                    })
                # Policy hook (only for the miner that received)
                decide = getattr(m, "decide_action", None)
                act = getattr(m, "act", None)
                if callable(decide) and callable(act):
                    try:
                        action = decide(t)
                        to_publish = act(action, t)
                        if to_publish:
                            for pb in to_publish:
                                _broadcast_block(pb, mid, t)
                    except Exception:
                        # Keep simulator running even if a custom policy misbehaves
                        pass

        # Note: policy hooks are invoked only for the miner whose local state just changed
        # (via MINE or successful DELIVER). No global polling to avoid information leakage.

        # Stop when miner 0's canonical head reaches target height (use cached selected head)
        head_id = miners[0].selected_head_id
        if miners[0].blocks[head_id].height >= steps:
            break

    # Compute results from miner 0's local view
    m0 = miners[0]
    final_head = m0._select_head()
    path = m0._iter_path_from_head(final_head.id)

    G = len(miners)
    canonical_counts = [0 for _ in range(G)]
    uncle_counts = [0 for _ in range(G)]

    # Count canonical production per group (exclude genesis)
    for b in path:
        if b.height == 0 or b.miner_id is None:
            continue
        canonical_counts[int(b.miner_id)] += 1

    # Count in-time uncles referenced on the canonical path by producing group
    for b in path:
        for uid in b.uncles:
            ub = m0.blocks.get(uid)
            if ub is not None and ub.miner_id is not None:
                uncle_counts[int(ub.miner_id)] += 1

    # Fork histogram and counts: size per height = number of UNIQUE groups present in-time at that height
    G = len(miners)
    S_size_hist = [0 for _ in range(G + 1)]
    fork_heights = 0
    for h, bids in m0.blocks_by_height.items():
        if h == 0:
            continue  # skip genesis height
        present_groups = set()
        for bid in bids:
            if bid in m0.in_time_blocks:
                prod = m0.blocks[bid].miner_id
                if prod is not None:
                    present_groups.add(int(prod))
        size = len(present_groups)
        if size > G:
            size = G
        S_size_hist[size] += 1
        if size > 1:
            fork_heights += 1

    timing: Optional[Dict] = None
    if track_times:
        # First rival timing per height from miner 0's view (in-time only)
        for h, bids in m0.blocks_by_height.items():
            if h == 0:
                continue
            times = [m0.block_first_seen[bid] for bid in bids if bid in m0.in_time_blocks and bid in m0.block_first_seen]
            if not times:
                continue
            times.sort()
            if len(times) > 1:
                t0 = times[0]
                t1 = times[1]
                first_rival_count += 1
                delta = max(0.0, t1 - t0)
                first_rival_sum += delta
                idx = int((delta / max_prop_delay) * bins)
                if idx >= bins:
                    idx = bins - 1
                first_rival_hist[idx] += 1
        timing = {
            "enabled": True,
            "first_rival_fraction": first_rival_count / float(steps),
            "mean_first_rival_time": (first_rival_sum / first_rival_count) if first_rival_count > 0 else None,
            "first_rival_hist": first_rival_hist,
            "first_rival_bin_edges": [ (i * max_prop_delay) / float(bins) for i in range(bins + 1) ],
        }


    return HonestEventqResult(
        groups=len(miners),
        shares=shares,
        Lambda=Lambda,
        D=D,
        steps=steps,
        canonical_counts=canonical_counts,
        uncle_counts=uncle_counts,
        fork_heights=fork_heights,
        S_size_hist=S_size_hist,
        max_prop_delay=max_prop_delay,
        timing=timing,
        elapsed_time=t,
        mine_events=mine_events,
        trace=trace_events if trace else None,
        attacker_index=attacker_idx,
    )

# Backward-compatible alias
simulate_honest_eventqV2 = simulate_mining_eventqV2
