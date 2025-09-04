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

    # Optional: record the winner (group index) of each MINE event for streak sanity checks
    winners = [] if track_times else None

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
            # Record winner for streak analysis (if enabled)
            if winners is not None:
                winners.append(gi)
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
            tol = 1e-9
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

        timing = {
            "enabled": True,
            "first_rival_fraction": first_rival_count / float(steps),
            "mean_first_rival_time": (first_rival_sum / first_rival_count) if first_rival_count > 0 else None,
            "first_rival_hist": first_rival_hist,
            "first_rival_bin_edges": [ (i * max_prop_delay) / float(bins) for i in range(bins + 1) ],
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
