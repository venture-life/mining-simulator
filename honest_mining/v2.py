from __future__ import annotations

from typing import List, Optional, Dict, Callable, Any, Union
import heapq
import math

try:
    import numpy as np
except ImportError as e:
    raise SystemExit(
        "numpy is required. Please install dependencies: `pip install -r requirements.txt`"
    ) from e

from .simulator import HonestEventqResult
from .miner import Miner, Block, WorkShare
from .selfish_miner import SelfishMiner


def simulate_mining_eventqV2(
    steps: int,
    *,
    groups: int = 3,
    shares: Optional[List[float]] = None,
    Lambda: float = 1.0 / 60.0,
    D: float = 5.0,
    max_prop_delay: float = 2.5,
    k: int = 3,
    deterministic_selection: bool = True,
    seed: Optional[int] = None,
    track_times: bool = False,
    time_bins: int = 50,
    trace: bool = False,
    trace_limit: Optional[int] = None,
    attacker_share: Optional[float] = None,
    selfish_policy: Optional[Callable[["SelfishMiner", float], Any]] = None,
    work_shares: Optional[int] = None,
    burnout_window: int = 3, # end of simulation, steps - burnout_window: Aggressively publish all withheld and behave honestly
    attacker_connectivity_edge: float = 250.0, # eg, 250x faster propagation
) -> HonestEventqResult:
    """
    Continuous-time simulation with a single global Poisson process of block arrivals (rate Λ).
    Each arrival is assigned to a miner via thinning by shares; the winner mints immediately. The
    resulting block or work-share is broadcast to all miners via per-delivery delays. DAG-gating
    at each recipient ensures parents (and prior versions) arrive before dependents; objects that
    arrive early are held pending until prerequisites are delivered. Broadcasting includes the
    origin miner; self-delivery uses a tiny epsilon delay to serialize local publish sub-events.

    Policy hooks (Publish-or-Perish):
    - After each MINE or successful DELIVER to miner i, we call i.act(t) at most once.
    - i.act(t) returns either None or a single Block (publish-one). If a Block is returned, it is
      broadcast immediately and will be delivered (including to i) via the event queue. Subsequent
      publish decisions happen on subsequent DELIVER events; there is no inner streaming loop.

    Attacker mode:
    - If attacker_share ∈ (0,1) is provided, a SelfishMiner is appended as the last miner. If
      honest shares are provided, they are rescaled to sum to (1−α); otherwise, they default to
      uniform (1−α)/groups. The attacker gets share α. Total miners become groups + 1.

    Miner-local behavior:
    - In-time classification window τ = D. Fork-resolution dominance parameter k for longest-chain
    - rule. Per-delivery propagation delays are capped at max_prop_delay (default 2.5s).

    Work-shares (optional):
    - If work_shares = N > 1 is provided, miners mine a real Block with probability 1/N on each
      global mining event and a WorkShare otherwise. WorkShares reference the chosen parent tip and
      carry a per-parent version starting at 0. The global event rate Λ should be scaled by N at the
      CLI level to preserve an average block interval of ~1/Λ_base.

    The simulation stops when miner 0's canonical head reaches 'steps' beyond genesis. Summary
    metrics are computed from miner 0's local view. If track_times=True, we also compute first-rival
    timing histograms over [0, max_prop_delay] using 'time_bins' bins.

    Burn-out transition:
    - If attacker_share is set and burnout_window > 0, when miner 0's canonical height enters the
      final `burnout_window` blocks before cutoff, the SelfishMiner switches to burn-out mode:
        - Immediately starts publishing withheld blocks (one-by-one via act()), and adopts once empty.
        - New mines from the attacker act honestly (mine on PUBLIC head and publish immediately).
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
        miners = []
        for i in range(groups):
            tb_seed = None if seed is None else int(seed) + int(i)
            miners.append(Miner(miner_id=i, k=k, tau=D, deterministic_selection=deterministic_selection, tie_break_seed=tb_seed, work_shares=work_shares))
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
        miners = []
        for i in range(groups):
            tb_seed = None if seed is None else int(seed) + int(i)
            miners.append(Miner(miner_id=i, k=k, tau=D, deterministic_selection=deterministic_selection, tie_break_seed=tb_seed, work_shares=work_shares))
        # Attacker tie-break seed also derived from simulator seed + attacker index
        atk_tb_seed = None if seed is None else int(seed) + int(attacker_idx)
        miners.append(SelfishMiner(miner_id=attacker_idx, k=k, tau=D, deterministic_selection=deterministic_selection, tie_break_seed=atk_tb_seed, alpha=a, policy=selfish_policy, work_shares=work_shares))

    # max propagation delay is now an independent parameter (default 2.5s), not tied to D
    max_prop_delay = float(max_prop_delay)
    if track_times:
        if time_bins <= 0:
            raise ValueError("time_bins must be > 0 when track_times=True")
        bins = int(time_bins)
        first_rival_count = 0
        first_rival_sum = 0.0
        first_rival_hist = [0 for _ in range(bins)]
        # Re-org tracking (miner 0): count head rollbacks by depth via LCA
        reorg_count = 0
        reorg_len_counts: Dict[int, int] = {}
        # Split counters: block-induced vs work-share-induced reorgs
        reorg_count_block = 0
        reorg_len_counts_block: Dict[int, int] = {}
        reorg_count_ws = 0
        reorg_len_counts_ws: Dict[int, int] = {}
        # Initialize previous head to miner 0's current selection (genesis initially)
        try:
            prev_head_id = None
            if len(locals().get('miners', [])) > 0:
                prev_head_id = miners[0].selected_head_id
        except Exception:
            prev_head_id = None

    rng = np.random.default_rng(seed)

    # Optional: record the winner (group index) of each MINE event for streak sanity checks
    winners = [] if track_times else None

    # Network delay sampling (right-skew)
    # - sample_delay(connectivity_boost, is_workshare): lognormal with mean ~= eff_max/2, capped at eff_max
    #   where eff_max = (max_prop_delay/2.5 if is_workshare else max_prop_delay), and further divided by
    #   attacker_connectivity_edge when connectivity_boost=True (e.g., when an attacker endpoint is involved).
    EPSILON = 1e-9  
    def sample_delay(connectivity_boost: bool = False, is_workshare: bool = False) -> float:
        base_max = max_prop_delay / 2.5 if is_workshare else max_prop_delay
        eff_max = base_max / attacker_connectivity_edge if connectivity_boost else base_max
        if eff_max <= 0.0:
            return 0.0
        sigma = 0.6  # right-skew
        mu = math.log(max(eff_max / 2.0, EPSILON)) - 0.5 * sigma * sigma
        d = float(rng.lognormal(mean=mu, sigma=sigma))
        return d if d <= eff_max else eff_max


    # Optional event trace for visualization
    trace_events: List[Dict] = [] if trace else None  # type: ignore[assignment]
    # De-duplication guard for trace rendering
    # We suppress duplicate trace entries to keep visualization edges consistent.
    _trace_seen: Dict[str, set] = {"DELIVER": set(), "MINE": set(), "DELIVER_WS": set(), "MINE_WS": set()} if trace else {}  # type: ignore[assignment]
    def _trace_append(ev: Dict) -> None:
        if trace_events is not None:
            ev_type = str(ev.get("type"))
            # Compute a conservative uniqueness key per event type
            key = None
            if ev_type == "DELIVER":
                # Uniqueness by (to, block_id)
                key = (ev.get("to"), ev.get("block_id"))
            elif ev_type == "MINE":
                # A block is mined once; use block_id for safety
                key = (ev.get("block_id"),)
            elif ev_type == "DELIVER_WS":
                # Uniqueness by (to, parent_id, version)
                key = (ev.get("to"), ev.get("parent_id"), ev.get("version"))
            elif ev_type == "MINE_WS":
                # A work-share is unique by its id
                key = (ev.get("ws_id"),)
            
            # If we have a seen-set for this type, skip duplicates
            if isinstance(_trace_seen, dict) and ev_type in _trace_seen and key is not None:
                s = _trace_seen[ev_type]
                if key in s:
                    return
                s.add(key)
            # Enforce optional limit on trace length (after de-dup decision)
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
            included_ws_ids=list(getattr(b, "included_ws_ids", []) or []),
        )

    # Helper: create a fresh WorkShare instance for delivery to a miner
    def _clone_workshare(ws: WorkShare) -> WorkShare:
        return WorkShare(
            id=ws.id,
            parent_id=ws.parent_id,
            miner_id=ws.miner_id,
            version=ws.version,
            created_time=ws.created_time,
        )


    # Helper: broadcast a freshly mined or published work object (Block or WorkShare)
    def _broadcast_workobject(obj: Union[Block, WorkShare], gi: int, t: float) -> None:
        # Schedule deliveries to all miners; self gets tiny epsilon delay
        G = len(miners)
        if G >= 1:
            for m in miners:
                if m.miner_id == gi:
                    delay = EPSILON
                else:
                    # Apply connectivity boost when either endpoint is the attacker
                    boost = bool(attacker_idx is not None and (gi == attacker_idx or m.miner_id == attacker_idx))
                    delay = max(sample_delay(connectivity_boost=boost, is_workshare=isinstance(obj, WorkShare)), 2*EPSILON)
                _push_event(t + delay, 0, (m.miner_id, obj))

    # Helper: compute LCA height between two heads in miner m's local view
    def _lca_height(m: Miner, id_a: str, id_b: str) -> int:
        if id_a == id_b:
            return m.blocks[id_a].height
        a = m.blocks.get(id_a)
        b = m.blocks.get(id_b)
        if a is None or b is None:
            return 0
        # Lift the deeper one to the same height
        while a.height > b.height and a.parent_id is not None:
            a = m.blocks.get(a.parent_id)
            if a is None:
                return 0
        while b.height > a.height and b.parent_id is not None:
            b = m.blocks.get(b.parent_id)
            if b is None:
                return 0
        # Walk up together until they meet
        while a.id != b.id:
            if a.parent_id is None or b.parent_id is None:
                return 0
            a = m.blocks.get(a.parent_id)
            b = m.blocks.get(b.parent_id)
            if a is None or b is None:
                return 0
        return a.height

    # Event queue: (time, kind, seq, payload)
    # kind: 0 = DELIVER, 1 = MINE (DELIVERs at same time process first).
    # seq is a monotonically increasing tie-breaker to avoid comparing payloads (e.g., Blocks).
    # payload for DELIVER: (miner_id, obj)
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

    # ---------------- DAG-gating state ----------------
    # Arrival time maps per miner
    arr_block_time: List[Dict[str, float]] = []
    arr_ws_pv_time: List[Dict[tuple, float]] = []  # key = (parent_id, version)
    for _ in range(len(miners)):
        arr_block_time.append({})
        arr_ws_pv_time.append({})

    # Pending dependency tracking
    # Unique pending key per (miner, obj):
    #  - Block: ("B", mid, block_id)
    #  - WorkShare: ("WS", mid, ws_id)
    class _Pending:
        __slots__ = ("obj", "base_time", "dep_keys")
        def __init__(self, obj, base_time: float, dep_keys: set):
            self.obj = obj
            self.base_time = float(base_time)
            self.dep_keys = set(dep_keys)

    pending_items: Dict[tuple, _Pending] = {}
    # Map dependency key -> set of pending keys waiting on it
    waiters_by_dep: Dict[tuple, set] = {}

    def _pending_key(mid: int, obj: Union[Block, WorkShare]):
        if isinstance(obj, Block):
            return ("B", int(mid), obj.id)
        else:
            return ("WS", int(mid), obj.id)

    def _dep_keys_for(mid: int, obj: Union[Block, WorkShare]) -> set:
        deps = set()
        if isinstance(obj, Block):
            pid = getattr(obj, "parent_id", None)
            if pid is not None and pid not in miners[mid].blocks:
                deps.add(("B", mid, pid))
        else:
            pid = getattr(obj, "parent_id", None)
            ver = int(getattr(obj, "version", 0) or 0)
            if pid is not None and pid not in miners[mid].blocks:
                deps.add(("B", mid, pid))
            if ver > 0:
                if (pid, ver - 1) not in arr_ws_pv_time[mid]:
                    deps.add(("WS", mid, pid, ver - 1))
        return deps

    def _register_pending(mid: int, obj: Union[Block, WorkShare], base_time: float, dep_keys: set) -> None:
        uid = _pending_key(mid, obj)
        if uid in pending_items:
            # Keep earliest base_time and union deps
            p = pending_items[uid]
            if base_time < p.base_time:
                p.base_time = float(base_time)
            p.dep_keys |= dep_keys
        else:
            pending_items[uid] = _Pending(obj, base_time, dep_keys)
        # Register in waiters for each dep key
        p = pending_items[uid]
        for dk in dep_keys:
            s = waiters_by_dep.get(dk)
            if s is None:
                s = set()
                waiters_by_dep[dk] = s
            s.add(uid)

    def _unblock_dependents(mid: int, dep_key: tuple, now: float) -> None:
        # Fetch and clear waiters for this dependency
        uids = waiters_by_dep.pop(dep_key, None)
        if not uids:
            return
        for uid in list(uids):
            p = pending_items.get(uid)
            if p is None:
                continue
            # Remove this dependency from the pending item
            if dep_key in p.dep_keys:
                p.dep_keys.discard(dep_key)
            # If no more dependencies, schedule delivery
            if not p.dep_keys:
                # Remove this pending from any other dep wait lists
                for other_dk, other_waiters in list(waiters_by_dep.items()):
                    if uid in other_waiters:
                        other_waiters.discard(uid)
                        if not other_waiters:
                            waiters_by_dep.pop(other_dk, None)
                # Pop from pending and schedule delivery at max(base_time, now + epsilon)
                pending_items.pop(uid, None)
                deliver_t = max(float(p.base_time), float(now) + EPSILON)
                # uid encodes mid already; reconstruct payload
                kind, mid2, _ = uid
                _push_event(deliver_t, 0, (int(mid2), p.obj))

    # Sanity for burnout_window
    try:
        burnout_window = int(burnout_window)
    except Exception:
        burnout_window = 0
    if burnout_window < 0:
        burnout_window = 0

    while events:
        t, kind, _, payload = heapq.heappop(events)
        if kind == 1:  # MINE
            # Assign to group via thinning
            G = len(miners)
            gi = int(rng.choice(G, p=shares))
            # Record winner for streak analysis (if enabled)
            if winners is not None:
                winners.append(gi)
            # Winning miner mints immediately at time t
            new_obj = miners[gi].on_mine(now=t)
            # Count mining events
            mine_events += 1
            # If the miner returned a block, we broadcast it immediately (honest-compatible).
            if isinstance(new_obj, Block):
                if trace:
                    _trace_append({
                        "type": "MINE",
                        "t": t,
                        "miner": gi,
                        "block_id": new_obj.id,
                        "parent_id": new_obj.parent_id,
                        "height": new_obj.height,
                        "weight": miners[gi].cum_block_weight.get(new_obj.id, None),
                        "uncles": list(new_obj.uncles) if getattr(new_obj, "uncles", None) else [],
                        "ws_included": len(getattr(new_obj, "included_ws_ids", []) or []),
                    })
                _broadcast_workobject(new_obj, gi, t)
            elif isinstance(new_obj, WorkShare):
                if trace:
                    _trace_append({
                        "type": "MINE_WS",
                        "t": t,
                        "miner": gi,
                        "ws_id": new_obj.id,
                        "parent_id": new_obj.parent_id,
                        "version": new_obj.version,
                    })
                _broadcast_workobject(new_obj, gi, t)
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
                                "ws_included": len(getattr(b, "included_ws_ids", []) or []),
                            })
                    except Exception:
                        pass
            # Policy hook (only for the miner that mined)
            act = getattr(miners[gi], "act", None)
            if callable(act):
                try:
                    out = act(t)
                    if isinstance(out, (Block, WorkShare)):
                        _broadcast_workobject(out, gi, t)
                except Exception:
                    # Keep simulator running even if a custom policy misbehaves
                    pass
            # Schedule next global mining arrival
            _push_event(t + float(rng.exponential(1.0 / Lambda)), 1, None)
        else:  # DELIVER
            mid, obj = payload
            m = miners[mid]
            # DAG-gating: deliver only when prerequisites are already satisfied; else register pending
            deps = _dep_keys_for(mid, obj)
            if deps:
                _register_pending(mid, obj, t, deps)
            else:
                # Capture miner 0's head before delivery to split reorgs by event type
                pre_head_for_split = None
                if track_times and mid == 0:
                    try:
                        pre_head_for_split = miners[0].selected_head_id
                    except Exception:
                        pre_head_for_split = None
                if isinstance(obj, Block):
                    # Deliver a fresh clone to avoid cross-miner state contamination
                    blk = obj
                    m.on_receive(_clone_block(blk), received_time=t)
                    # Record arrival and unblock dependents
                    arr_block_time[mid][blk.id] = t
                    _unblock_dependents(mid, ("B", mid, blk.id), t)
                    if trace:
                        _trace_append({
                            "type": "DELIVER",
                            "t_mine": blk.created_time,
                            "t_deliver": t,
                            "from": int(blk.miner_id) if blk.miner_id is not None else None,
                            "to": mid,
                            "block_id": blk.id,
                            "parent_id": blk.parent_id,
                            # Receiver-local metrics after delivery
                            "height": m.blocks[blk.id].height,
                            "weight": m.cum_block_weight.get(blk.id, None),
                            "ws_included": len(getattr(blk, "included_ws_ids", []) or []),
                        })
                elif isinstance(obj, WorkShare):
                    ws = obj
                    m.on_receive_workshare(_clone_workshare(ws), received_time=t)
                    # Record arrival and unblock dependents
                    pid = ws.parent_id
                    arr_ws_pv_time[mid][(pid, ws.version)] = t
                    _unblock_dependents(mid, ("WS", mid, pid, ws.version), t)
                    if trace:
                        _trace_append({
                            "type": "DELIVER_WS",
                            "t_mine": ws.created_time,
                            "t_deliver": t,
                            "from": int(ws.miner_id) if ws.miner_id is not None else None,
                            "to": mid,
                            "ws_id": ws.id,
                            "parent_id": ws.parent_id,
                            "version": ws.version,
                        })
                # Check burnout
                h0 = miners[0].blocks[miners[0].selected_head_id].height
                if h0 >= max(0, steps - burnout_window):
                    if isinstance(m, SelfishMiner) and not getattr(m, 'burnout', False):
                        m.burnout = True
                # Policy hook (only for the miner that received)
                act = getattr(m, "act", None)
                if callable(act):
                    out = act(t)
                    if isinstance(out, (Block, WorkShare)):
                        _broadcast_workobject(out, mid, t)
                # Split re-org detection for this specific delivery to miner 0
                if track_times and mid == 0 and pre_head_for_split is not None:
                    cur_head_id_local = miners[0].selected_head_id
                    if cur_head_id_local != pre_head_for_split:
                        m0 = miners[0]
                        old_h_local = m0.blocks.get(pre_head_for_split).height if pre_head_for_split in m0.blocks else 0
                        lca_h_local = _lca_height(m0, pre_head_for_split, cur_head_id_local)
                        depth_local = max(0, int(old_h_local) - int(lca_h_local))
                        if depth_local > 0:
                            if isinstance(obj, Block):
                                reorg_count_block += 1
                                reorg_len_counts_block[depth_local] = reorg_len_counts_block.get(depth_local, 0) + 1
                            elif isinstance(obj, WorkShare):
                                reorg_count_ws += 1
                                reorg_len_counts_ws[depth_local] = reorg_len_counts_ws.get(depth_local, 0) + 1

        # Re-org detection: track miner 0 head changes and measure rollback depth
        if track_times:
            if prev_head_id is None:
                prev_head_id = miners[0].selected_head_id
            else:
                cur_head_id = miners[0].selected_head_id
                if cur_head_id != prev_head_id:
                    m0 = miners[0]
                    old_h = m0.blocks.get(prev_head_id).height if prev_head_id in m0.blocks else 0
                    lca_h = _lca_height(m0, prev_head_id, cur_head_id)
                    depth = max(0, int(old_h) - int(lca_h))
                    if depth > 0:
                        reorg_count += 1
                        reorg_len_counts[depth] = reorg_len_counts.get(depth, 0) + 1
                    prev_head_id = cur_head_id

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
        # Compute winner streak histogram across all miners and compare to expectation
        streaks = None
        if winners is not None and len(winners) > 0:
            M = len(winners)
            # Observed histogram: total across all miners
            obs_total: Dict[int, int] = {}
            # Per-miner observed histograms
            G = len(miners)
            obs_per_miner: List[Dict[int, int]] = [{} for _ in range(G)]
            # Single pass to accumulate runs
            cur = winners[0]
            run_len = 1
            for w in winners[1:]:
                if w == cur:
                    run_len += 1
                else:
                    obs_total[run_len] = obs_total.get(run_len, 0) + 1
                    d = obs_per_miner[cur]
                    d[run_len] = d.get(run_len, 0) + 1
                    cur = w
                    run_len = 1
            # Final run
            obs_total[run_len] = obs_total.get(run_len, 0) + 1
            d = obs_per_miner[cur]
            d[run_len] = d.get(run_len, 0) + 1

            maxL = max(obs_total.keys()) if obs_total else 0
            # Expected counts: sum over i of expected number of EXACT runs of length L for symbol i
            # Boundary-aware formula:
            # For L < M: E_i(L) = max(M-L-1,0) * p^L * (1-p)^2 + 2 * p^L * (1-p)
            # For L == M: E_i(L) = p^M
            exp_total: Dict[int, float] = {}
            for L in range(1, maxL + 1):
                Etot = 0.0
                for p in shares:
                    p = float(p)
                    if L == M:
                        Ei = p ** M
                    else:
                        interior = max(M - L - 1, 0)
                        Ei = (p ** L) * ((1.0 - p) ** 2) * interior + 2.0 * (p ** L) * (1.0 - p)
                    Etot += Ei
                exp_total[L] = Etot

            # Normalize into ordered lists for stable printing (index L-1 corresponds to length L)
            obs_list = [obs_total.get(L, 0) for L in range(1, maxL + 1)]
            exp_list = [exp_total.get(L, 0.0) for L in range(1, maxL + 1)]

            # Quick sanity diagnostics: per-bin p-values for bins with enough expected mass, aggregate the tail
            MIN_E = 10.0  # minimum expected count per bin for reliable per-bin normal approximation

            # Helper: two-sided p-value via normal approximation (with continuity correction)
            def _normal_two_sided_p(o: float, e: float) -> float:
                if e <= 0.0:
                    return 1.0 if o == 0.0 else 0.0
                sd = math.sqrt(e)
                diff = o - e
                if diff > 0.0:
                    adj = diff - 0.5
                elif diff < 0.0:
                    adj = diff + 0.5
                else:
                    adj = 0.0
                z = adj / sd if sd > 0.0 else 0.0
                # two-sided from standard normal
                return float(math.erfc(abs(z) / math.sqrt(2.0)))

            # Helper: exact Poisson CDF for small lambdas (sums series)
            def _poisson_cdf(k: int, lam: float) -> float:
                if lam < 0.0:
                    return 0.0
                if k < 0:
                    return 0.0
                # Start with P(X=0)
                term = math.exp(-lam)
                s = term
                for i in range(1, k + 1):
                    term = term * lam / float(i)
                    s += term
                return float(s)

            def _poisson_two_sided_p(o: int, lam: float) -> float:
                if lam <= 0.0:
                    return 1.0 if o == 0 else 0.0
                # For small lambda, compute exact tail; otherwise fall back to normal approx
                if lam < 50.0:
                    if o <= lam:
                        p_one = _poisson_cdf(o, lam)
                        return float(min(1.0, 2.0 * p_one))
                    else:
                        # P(X >= o) = 1 - P(X <= o-1)
                        p_one = 1.0 - _poisson_cdf(o - 1, lam)
                        return float(min(1.0, 2.0 * p_one))
                else:
                    return _normal_two_sided_p(float(o), lam)

            # Helper: upper-tail p-value for chi-square via Wilson–Hilferty approximation
            def _chi2_sf_wh(x: float, k: int) -> float:
                if k <= 0:
                    return 1.0
                if x <= 0.0:
                    return 1.0
                z = ((x / float(k)) ** (1.0 / 3.0) - (1.0 - 2.0 / (9.0 * float(k)))) / math.sqrt(2.0 / (9.0 * float(k)))
                return 0.5 * math.erfc(z / math.sqrt(2.0))

            # Helper: Binomial survival function P(K >= k) for K ~ Binom(n,p)
            def _binom_sf(k: int, n: int, p: float) -> float:
                if n <= 0:
                    return 1.0
                if k <= 0:
                    return 1.0
                if k > n:
                    return 0.0
                # Exact sum for moderate n (here n is small: number of per-bin tests up to L0)
                one_minus_p = 1.0 - p
                s = 0.0
                for i in range(k, n + 1):
                    s += math.comb(n, i) * (p ** i) * (one_minus_p ** (n - i))
                return float(min(max(s, 0.0), 1.0))

            # Determine cutoff L0: largest L with expected >= MIN_E (monotone decreasing in L)
            L0 = 0
            for L in range(1, maxL + 1):
                if exp_total.get(L, 0.0) >= MIN_E:
                    L0 = L

            # Per-bin p-values up to L0 (Poisson-based; exact for small lambda, normal approx for large)
            p_values = []  # index L-1 corresponds to length L, only for L=1..L0
            for L in range(1, L0 + 1):
                E = float(exp_total.get(L, 0.0))
                Oi = int(obs_total.get(L, 0))
                p_values.append(_poisson_two_sided_p(Oi, E))

            # Aggregate the tail L >= L0+1 (include expected mass beyond maxL until negligible)
            tail_Lmin = L0 + 1
            tail_obs = 0
            if tail_Lmin <= maxL:
                for L in range(tail_Lmin, maxL + 1):
                    tail_obs += int(obs_total.get(L, 0))
            # Sum expected tail to convergence (geometric decay), capped by M
            tail_exp = 0.0
            tol = EPSILON 
            max_iter = 10000
            L = tail_Lmin
            iters = 0
            while L <= M and iters < max_iter:
                # Expected exact runs of length L across all miners
                Etot = 0.0
                for p in shares:
                    p = float(p)
                    if L == M:
                        Ei = p ** M
                    else:
                        interior = max(M - L - 1, 0)
                        Ei = (p ** L) * ((1.0 - p) ** 2) * interior + 2.0 * (p ** L) * (1.0 - p)
                    Etot += Ei
                if Etot < tol:
                    break
                tail_exp += Etot
                L += 1
                iters += 1

            # Compute a p-value for the aggregated tail
            tail_method = "poisson_exact" if tail_exp < 50.0 else "normal_approx_cc"
            tail_p = _poisson_two_sided_p(int(tail_obs), float(tail_exp))

            # Global chi-square across L=1..L0 and tail
            chi2_stat = 0.0
            df = 0
            for L in range(1, L0 + 1):
                E = float(exp_total.get(L, 0.0))
                if E > 0.0:
                    O = float(obs_total.get(L, 0))
                    chi2_stat += (O - E) * (O - E) / E
                    df += 1
            if tail_exp > 0.0:
                chi2_stat += (float(tail_obs) - float(tail_exp)) ** 2 / float(tail_exp)
                df += 1
            chi2_p = _chi2_sf_wh(chi2_stat, df) if df > 0 else 1.0

            # Per-miner local diagnostics
            per_miner = []
            for i in range(G):
                obs_i = obs_per_miner[i]
                if not obs_i:
                    continue
                maxL_i = max(obs_i.keys())
                # Expected exact runs for miner i
                p_i = float(shares[i])
                exp_i: Dict[int, float] = {}
                for L in range(1, maxL_i + 1):
                    if L == M:
                        Ei = p_i ** M
                    else:
                        interior = max(M - L - 1, 0)
                        Ei = (p_i ** L) * ((1.0 - p_i) ** 2) * interior + 2.0 * (p_i ** L) * (1.0 - p_i)
                    exp_i[L] = Ei
                # Cutoff for miner i
                L0_i = 0
                for L in range(1, maxL_i + 1):
                    if exp_i.get(L, 0.0) >= MIN_E:
                        L0_i = L
                # Per-bin p-values for miner i
                pvals_i = []
                for L in range(1, L0_i + 1):
                    Ei = float(exp_i.get(L, 0.0))
                    Oi = int(obs_i.get(L, 0))
                    pvals_i.append(_poisson_two_sided_p(Oi, Ei))
                # Tail for miner i
                tail_Lmin_i = L0_i + 1
                tail_obs_i = 0
                if tail_Lmin_i <= maxL_i:
                    for L in range(tail_Lmin_i, maxL_i + 1):
                        tail_obs_i += int(obs_i.get(L, 0))
                tail_exp_i = 0.0
                L = tail_Lmin_i
                iters_i = 0
                while L <= M and iters_i < max_iter:
                    if L == M:
                        Ei = p_i ** M
                    else:
                        interior = max(M - L - 1, 0)
                        Ei = (p_i ** L) * ((1.0 - p_i) ** 2) * interior + 2.0 * (p_i ** L) * (1.0 - p_i)
                    if Ei < tol:
                        break
                    tail_exp_i += Ei
                    L += 1
                    iters_i += 1
                tail_method_i = "poisson_exact" if tail_exp_i < 50.0 else "normal_approx_cc"
                tail_p_i = _poisson_two_sided_p(int(tail_obs_i), float(tail_exp_i))
                # Chi-square for miner i
                chi2_i = 0.0
                df_i = 0
                for L in range(1, L0_i + 1):
                    Ei = float(exp_i.get(L, 0.0))
                    if Ei > 0.0:
                        Oi = float(obs_i.get(L, 0))
                        chi2_i += (Oi - Ei) * (Oi - Ei) / Ei
                        df_i += 1
                if tail_exp_i > 0.0:
                    chi2_i += (float(tail_obs_i) - float(tail_exp_i)) ** 2 / float(tail_exp_i)
                    df_i += 1
                chi2_p_i = _chi2_sf_wh(chi2_i, df_i) if df_i > 0 else 1.0
                per_miner.append({
                    "miner": i,
                    "share": p_i,
                    "max_observed_length": maxL_i,
                    "L0_cutoff": L0_i,
                    "p_values": pvals_i,
                    "tail": {
                        "L_min": tail_Lmin_i,
                        "observed": int(tail_obs_i),
                        "expected": float(tail_exp_i),
                        "p_value": float(tail_p_i),
                        "method": tail_method_i,
                    },
                    "chi2": {
                        "stat": float(chi2_i),
                        "df": int(df_i),
                        "p_value": float(chi2_p_i),
                        "method": "wilson_hilferty_approx",
                    },
                })

            # Compact PASS/FAIL sanity summary
            ALPHA_GLOBAL = 0.1
            ALPHA_BIN = 0.1
            ALPHA_TAIL = 0.1
            ALPHA_BINOM = 0.1

            num_bins = len(p_values)
            small_bins = sum(1 for pv in p_values if pv < ALPHA_BIN)
            binom_p = _binom_sf(small_bins, num_bins, ALPHA_BIN) if num_bins > 0 else 1.0

            global_pass = bool(chi2_p >= ALPHA_GLOBAL)
            tail_pass = bool(tail_p >= ALPHA_TAIL)
            per_miner_pass = all(m["chi2"]["p_value"] >= ALPHA_GLOBAL for m in per_miner)
            binom_pass = bool(binom_p >= ALPHA_BINOM)

            verdict_pass = global_pass and tail_pass and per_miner_pass and binom_pass

            min_miner_p = min((m["chi2"]["p_value"] for m in per_miner), default=1.0)

            sanity = {
                "alpha": {
                    "global": ALPHA_GLOBAL,
                    "bin": ALPHA_BIN,
                    "tail": ALPHA_TAIL,
                    "binom": ALPHA_BINOM,
                },
                "global_pass": global_pass,
                "tail_pass": tail_pass,
                "per_miner_pass": per_miner_pass,
                "binom_pass": binom_pass,
                "verdict": "PASS" if verdict_pass else "FAIL",
                "stats": {
                    "global_p": float(chi2_p),
                    "tail_p": float(tail_p),
                    "min_miner_chi2_p": float(min_miner_p),
                    "k_small": int(small_bins),
                    "N_bins": int(num_bins),
                    "binom_p": float(binom_p),
                },
            }

            streaks = {
                "enabled": True,
                "mine_events": M,
                "max_observed_length": maxL,
                "observed_total": obs_list,
                "expected_total": exp_list,
                "min_expected_per_bin": MIN_E,
                "L0_cutoff": L0,
                "p_values": p_values,  # lengths 1..L0
                "tail": {
                    "L_min": tail_Lmin,
                    "observed": int(tail_obs),
                    "expected": float(tail_exp),
                    "p_value": float(tail_p),
                    "method": tail_method,
                },
                "p_value_method_per_bin": "poisson_exact_or_normal",
                "global_chi2": {
                    "stat": float(chi2_stat),
                    "df": int(df),
                    "p_value": float(chi2_p),
                    "method": "wilson_hilferty_approx",
                },
                "per_miner": per_miner,
                "sanity": sanity,
            }

        # Build re-org histogram results
        if track_times:
            # Aggregate totals from split counters (block + ws) for primary 'reorgs'
            agg_len_counts: Dict[int, int] = {}
            for L, c in reorg_len_counts_block.items():
                agg_len_counts[L] = agg_len_counts.get(L, 0) + int(c)
            for L, c in reorg_len_counts_ws.items():
                agg_len_counts[L] = agg_len_counts.get(L, 0) + int(c)
            agg_count = int(reorg_count_block) + int(reorg_count_ws)
            max_reorg_len = max(agg_len_counts.keys(), default=0)
            reorg_len_hist = [agg_len_counts.get(L, 0) for L in range(1, max_reorg_len + 1)] if max_reorg_len > 0 else []
            # Integer-length bins: edges at [0, 1, ..., max_len+1]
            reorg_len_bin_edges = [i for i in range(0, (max_reorg_len + 1) + 1)] if max_reorg_len > 0 else [0, 1]
            total_reorg_len = sum(L * c for L, c in agg_len_counts.items())
            mean_reorg_len = (total_reorg_len / float(agg_count)) if agg_count > 0 else None

            # Block-induced reorgs
            max_reorg_len_block = max(reorg_len_counts_block.keys(), default=0)
            reorg_len_hist_block = [reorg_len_counts_block.get(L, 0) for L in range(1, max_reorg_len_block + 1)] if max_reorg_len_block > 0 else []
            reorg_len_bin_edges_block = [i for i in range(0, (max_reorg_len_block + 1) + 1)] if max_reorg_len_block > 0 else [0, 1]
            total_reorg_len_block = sum(L * c for L, c in reorg_len_counts_block.items())
            mean_reorg_len_block = (total_reorg_len_block / float(reorg_count_block)) if reorg_count_block > 0 else None

            # Work-share-induced reorgs
            max_reorg_len_ws = max(reorg_len_counts_ws.keys(), default=0)
            reorg_len_hist_ws = [reorg_len_counts_ws.get(L, 0) for L in range(1, max_reorg_len_ws + 1)] if max_reorg_len_ws > 0 else []
            reorg_len_bin_edges_ws = [i for i in range(0, (max_reorg_len_ws + 1) + 1)] if max_reorg_len_ws > 0 else [0, 1]
            total_reorg_len_ws = sum(L * c for L, c in reorg_len_counts_ws.items())
            mean_reorg_len_ws = (total_reorg_len_ws / float(reorg_count_ws)) if reorg_count_ws > 0 else None

        timing = {
            "enabled": True,
            "first_rival_fraction": (first_rival_count / float(max(1, steps))) if first_rival_count > 0 else 0.0,
            "mean_first_rival_time": (first_rival_sum / first_rival_count) if first_rival_count > 0 else None,
            "first_rival_hist": first_rival_hist,
            "first_rival_bin_edges": [ (i * max_prop_delay) / float(bins) for i in range(bins + 1) ],
            "reorgs": {
                "count": int(agg_count) if track_times else 0,
                "mean_length": float(mean_reorg_len) if track_times and (mean_reorg_len is not None) else None,
                "length_hist": reorg_len_hist if track_times else [],
                "length_bin_edges": reorg_len_bin_edges if track_times else [],
            },
            "reorgs_block": {
                "count": int(reorg_count_block) if track_times else 0,
                "mean_length": float(mean_reorg_len_block) if track_times and (mean_reorg_len_block is not None) else None,
                "length_hist": reorg_len_hist_block if track_times else [],
                "length_bin_edges": reorg_len_bin_edges_block if track_times else [],
            },
            "reorgs_ws": {
                "count": int(reorg_count_ws) if track_times else 0,
                "mean_length": float(mean_reorg_len_ws) if track_times and (mean_reorg_len_ws is not None) else None,
                "length_hist": reorg_len_hist_ws if track_times else [],
                "length_bin_edges": reorg_len_bin_edges_ws if track_times else [],
            },
            "streaks": streaks,
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
