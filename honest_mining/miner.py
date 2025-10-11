from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple, Callable
import hashlib
import random


@dataclass
class Block:
    """
    Minimal block object used by Miner.

    Notes
    -----
    - Height is derived from parent (genesis height = 0).
    - Cumulative chain weight is miner-local and stored in Miner.cum_block_weight,
      not on the Block itself. Step weight = 1 if this block is in-time + number
      of in-time uncles it references at creation time; 0 if the block is late.
    - Uncles are references to block ids that are siblings of this block's parent
      ("N-1" relative to this block): siblings of parent at parent's height that
      were in-time from the miner's perspective at creation time.
    """
    id: str
    parent_id: Optional[str]
    miner_id: Optional[int] = None
    height: int = 0
    uncles: List[str] = field(default_factory=list)
    created_time: float = 0.0
    # Explicit list of work-share ids the producer includes for this block (same parent). Treated as a set.
    included_ws_ids: List[str] = field(default_factory=list)


@dataclass
class WorkShare:
    """
    Lightweight work-share object propagated by the simulator when using N work-shares.

    Notes
    -----
    - Work-shares reference the current tip/parent by id and carry a sequential version per parent
      from the miner's local perspective at creation time, starting at version 0.
    - They do not alter local chain state/weights; miners only count them per parent to derive
      the next version when they mine a subsequent work-share on the same parent.
    """
    id: str
    parent_id: Optional[str]
    miner_id: Optional[int] = None
    version: int = 0
    created_time: float = 0.0


class Miner:
    """
    Honest miner maintaining a local block tree and selecting a mining head using
    the specified fork-resolution rules.

    Fork resolution (evaluated on known heads):
    1) If one chain is longer than all others by at least k blocks, mine on the
       longest chain.
    2) Otherwise, restrict to heads within (k−1) of the TOP HEIGHT and choose the
       chain with the largest weight among those.
    3) If multiple chains share the largest weight:
       - If deterministic_selection=True (default), choose deterministically by
         hashing the sorted candidate ids (BLAKE2b) and selecting index = H % len(candidates).
       - Otherwise, choose uniformly at random among the tied heads.

    Definitions (evaluated from this miner's local perspective):
    - In-time block:
        * First block seen at a given height h (sets t_first[h]).
        * Any additional block at height h that arrives within τ of t_first[h].
          Equivalently, if a block extends a strictly higher height than any block
          seen so far, it will also be the first-seen at that height and be in-time.
    - Uncle of B2: a competing in-time block of B2's parent block (i.e., a sibling
      of B2's parent sharing its grandparent).
    - Weight of a chain (path from genesis to head):
        number of in-time blocks on the path + number of in-time uncle hashes
        embedded in the in-time blocks along that path.
    """

    def __init__(self, miner_id: int, k: int, tau: float, *, genesis_id: str = "GENESIS", deterministic_selection: bool = True, tie_break_seed: Optional[int] = None, work_shares: Optional[int] = None) -> None:
        self.miner_id = miner_id
        # Sanity check: enforce k >= 1 (k=0 behaves like k=1 for 'strictly dominant longest')
        self.k = max(1, int(k))
        self.tau = float(tau)
        self.deterministic_selection: bool = bool(deterministic_selection)
        self.tie_break_seed: Optional[int] = (int(tie_break_seed) if tie_break_seed is not None else None)
        # Per-miner RNG for random tie-breaking (seeded by simulator when provided)
        self._rng = random.Random(self.tie_break_seed)

        # Work-share configuration (N). If N <= 1 or unset, behave as standard blocks-only mining.
        try:
            self.work_shares: int = int(work_shares) if work_shares is not None else 1
        except Exception:
            self.work_shares = 1
        if self.work_shares <= 0:
            self.work_shares = 1

        # Local block tree
        self.blocks: Dict[str, Block] = {}
        self.children: Dict[str, List[str]] = {}

        # Maintain current leaf set incrementally (ids of blocks with no known children)
        self.leaves: Set[str] = set()
        # Maintain leaves grouped by height for fast k-dominance checks
        self.leaves_by_height: Dict[int, Set[str]] = {}
        # Track current maximum height among known blocks
        self.max_height: int = 0

        # Per-miner in-time bookkeeping
        # first_seen_time_by_height[h] = earliest wall-clock time we first saw any block at height h
        self.first_seen_time_by_height: Dict[int, float] = {}
        self.in_time_blocks: Set[str] = set()
        # Track first time we saw each block id (used for in-time classification and metrics)
        self.block_first_seen: Dict[str, float] = {}
        # Maintain the set of known block ids at each height (used for metrics/analysis)
        self.blocks_by_height: Dict[int, Set[str]] = {}

        # Miner-local cumulative weight per block id (float to support WS-based weights)
        self.cum_block_weight: Dict[str, float] = {}
        # Per-block step weight (local contribution at the block)
        self.step_weight: Dict[str, float] = {}

        # Simple local id generator for self-mined blocks
        self._next_seq = 0

        # Work-share tracking:
        # - workshare_count_by_parent: per-parent NEXT version index (max seen version + 1)
        #   Used when mining a new work-share to assign `version` for that parent.
        # - _seen_workshare_ids: de-dup set by work-share id to ignore exact duplicates.
        self.workshare_count_by_parent: Dict[str, int] = {}
        self._seen_workshare_ids: Set[str] = set()
        # WS arrival tuples by parent: (arrival_time, ws_id, version)
        # Used to build included_ws_ids at mine-time and to evaluate pre-block/late WS inclusion
        self._ws_arrivals_by_parent: Dict[str, List[Tuple[float, str, int]]] = {}
        self._ws_counted_by_block: Dict[str, Set[str]] = {}
        # WS continuity caches to avoid repeated scans/sorts
        # - _ws_earliest_id_by_parent[pid][ver] = first-seen ws_id for that version
        # - _ws_contig_count_by_parent[pid] = largest m such that versions [0..m-1] are all present
        # - _ws_arrival_time_by_id[ws_id] = local delivery time for fast cutoff checks
        self._ws_earliest_id_by_parent: Dict[str, Dict[int, str]] = {}
        self._ws_contig_count_by_parent: Dict[str, int] = {}
        self._ws_arrival_time_by_id: Dict[str, float] = {}

        # Genesis
        genesis = Block(id=genesis_id, parent_id=None, miner_id=None, height=0, uncles=[], created_time=0.0)
        # Record first-seen time for genesis and connect
        self.block_first_seen[genesis.id] = 0.0
        self._add_block_connected(genesis, received_time=0.0)
        # Selected head tracking (updated on every event thereafter)
        self.selected_head_id: str = genesis.id

    # ------------------------- public API -------------------------
    def on_receive(self, block: Block, received_time: float) -> None:
        """
        Receive a block from the network at received_time.
        Assumes ordered per-miner delivery: if parent_id is set, parent must already be known.
        """
 
        # Fast path: duplicate delivery → keep earliest first-seen and return
        if block.id in self.blocks:
            return
 
        # Record earliest first-seen timestamp
        prev_seen = self.block_first_seen.get(block.id)
        if prev_seen is None or received_time < prev_seen:
            self.block_first_seen[block.id] = received_time
 
        # Enforce parent-known invariant under ordered delivery
        if block.parent_id is not None and block.parent_id not in self.blocks:
            raise AssertionError("Out-of-order delivery: parent unknown for block {bid} (parent {pid})".format(
                bid=block.id, pid=block.parent_id
            ))

        # Connect immediately
        self._add_block_connected(block, received_time)

        # Update selected head after processing this receive
        self._update_selected_head()

    def on_mine(self, now: float):
        """
        Called when this miner wins the lottery at time `now`.

        When work_shares = N > 1, with probability 1/N we mine a real block (legacy behavior).
        Otherwise, we mine a work-share that references the current parent tip and assign a
        version equal to the number of work-shares we have already seen for that parent.

        Returns
        -------
        Block | WorkShare
            A Block when a real block is mined, else a WorkShare instance.

        Note
        ----
        Do not deliver to self here; the simulator will schedule deliveries (including to self
        with a tiny epsilon delay) via its event queue.
        """
        N = int(self.work_shares) if isinstance(self.work_shares, int) else 1
        is_block = (N <= 1) or (self._rng.random() <= (1.0 / float(N)))
        # Choose parent once, WS-aware when mining a work-share
        parent = self.choose_parent_to_mine_upon(now, for_workshare=(N > 1 and not is_block))
        # Track the selected parent head at the time of mining
        self.selected_head_id = parent.id

        if is_block:
            # Produce a real block (legacy path)
            new_id = f"{self.miner_id}:{self._next_seq}"
            self._next_seq += 1

            # Determine uncles: siblings of parent that are in-time (from this miner's view)
            uncles: List[str] = []
            if parent.parent_id is not None:
                gp = parent.parent_id
                for sib_id in self.children.get(gp, []):
                    if sib_id == parent.id:
                        continue
                    if sib_id in self.in_time_blocks:
                        uncles.append(sib_id)

            # Build included WS set at mine time: earliest per version, continuous from v=0 (up to now)
            included_ids: List[str] = []
            if N > 1:
                _cnt, included_ids = self._continuous_ws_for_parent(parent.id, now)
            new_block = Block(
                id=new_id,
                parent_id=parent.id,
                miner_id=self.miner_id,
                height=parent.height + 1,
                uncles=uncles,
                created_time=now,
                included_ws_ids=included_ids,
            )
            return new_block

        # Produce a work-share for the selected parent (WS-mode)
        if N > 1:
            pid = parent.id
            cur = self.workshare_count_by_parent.get(pid, 0)
            ws_id = f"WS:{self.miner_id}:{pid}:{cur}"
            ws = WorkShare(id=ws_id, parent_id=pid, miner_id=self.miner_id, version=cur, created_time=now)
            return ws

    def on_receive_workshare(self, ws: WorkShare, received_time: float) -> None:
        """
        Receive a work-share and update per-parent counts. Duplicate deliveries are ignored.
        """
        if ws.id in self._seen_workshare_ids:
            return
        self._seen_workshare_ids.add(ws.id)
        pid = ws.parent_id or "GENESIS"
        # Update next-version index as max(current, ws.version + 1) to avoid over-counting
        # when multiple distinct work-shares exist at the same version for this parent.
        cur_next = self.workshare_count_by_parent.get(pid, 0)
        v = int(getattr(ws, "version", 0) or 0)
        self.workshare_count_by_parent[pid] = max(cur_next, v + 1)
        # Record arrival for future blocks referencing this parent
        self._ws_arrivals_by_parent.setdefault(pid, []).append((received_time, ws.id, v))
        self._ws_arrival_time_by_id[ws.id] = float(received_time)
        # Update earliest-per-version cache and contiguous count for this parent
        emap = self._ws_earliest_id_by_parent.setdefault(pid, {})
        if v not in emap:
            emap[v] = ws.id
        # Increment contiguous count while next version exists
        cur_cc = int(self._ws_contig_count_by_parent.get(pid, 0))
        while emap.get(cur_cc) is not None:
            cur_cc += 1
        self._ws_contig_count_by_parent[pid] = cur_cc
        # If WS-based weighting is enabled (N>1), apply late WS within tau to eligible child blocks
        if self.work_shares and int(self.work_shares) > 1:
            changed = self._apply_late_ws_to_children(ws, received_time)
            if changed:
                self._update_selected_head()

    # ------------------------- internal helpers -------------------------
    def _add_block_connected(self, block: Block, received_time: float) -> None:
        """Insert a block whose parent (if any) is already known; set height and in-time state.

        Notes
        -----
        - In-time is determined by the earliest first-seen timestamp at this height (t0).
        - With ordered, fixed-delay delivery, once t0 for a height is set it will not move earlier.
          We therefore classify each new block against the current t0 without re-scanning others.
        """
        if block.id in self.blocks:
            return  # already known

        # Set height from parent
        if block.parent_id is None:
            block.height = 0
        else:
            parent = self.blocks[block.parent_id]
            block.height = parent.height + 1

        self.blocks[block.id] = block
        self.children.setdefault(block.id, [])  # ensure presence for leaf tracking
        if block.parent_id is not None:
            self.children.setdefault(block.parent_id, []).append(block.id)
            # Parent is no longer a leaf once it gains a child
            self.leaves.discard(block.parent_id)
            # Keep leaves_by_height in sync for the parent removal
            ph = parent.height
            s = self.leaves_by_height.get(ph)
            if s is not None:
                s.discard(parent.id)
                if not s:
                    self.leaves_by_height.pop(ph, None)
        # New block starts as a leaf
        self.leaves.add(block.id)
        # Track leaf by height for fast queries
        self.leaves_by_height.setdefault(block.height, set()).add(block.id)
        # Update max height
        if block.height > self.max_height:
            self.max_height = block.height

        # Track membership at this height
        h = block.height
        self.blocks_by_height.setdefault(h, set()).add(block.id)

        # Prune old leaves far below the tip to keep head enumeration bounded
        self._prune_old_leaves()

        # Use the earliest known first-seen time for this block (fallback to provided)
        t_recv = self.block_first_seen.get(block.id, received_time)

        # In-time classification based on earliest first-seen at this height (t0)
        t0 = self.first_seen_time_by_height.get(h)
        if t0 is None:
            # First block seen at this height establishes t0 and is in-time
            self.first_seen_time_by_height[h] = t_recv
            self.in_time_blocks.add(block.id)
        else:
            # Classify this block against existing t0
            if (t_recv - t0) <= self.tau:
                self.in_time_blocks.add(block.id)
            else:
                # Ensure it's not incorrectly marked in-time
                self.in_time_blocks.discard(block.id)

        # Compute and store step weight and cumulative weight
        if self.work_shares and int(self.work_shares) > 1:
            # WS-based weighting: ignore uncles. Gate by in-time classification.
            self.step_weight[block.id] = float(self._initial_step_weight_for_block(block, t_recv))
        else:
            # Legacy weighting: block counts only if in-time; include in-time uncles
            step_w_f = 0.0
            if block.id in self.in_time_blocks:
                step_w_f = 1.0
                for u in block.uncles:
                    if u in self.in_time_blocks:
                        step_w_f += 1.0
            self.step_weight[block.id] = float(step_w_f)
        parent_w = 0.0 if block.parent_id is None else float(self.cum_block_weight.get(block.parent_id, 0.0))
        self.cum_block_weight[block.id] = parent_w + float(self.step_weight.get(block.id, 0.0))

    # ------------------------- helper methods -------------------------
    def _is_in_time(self, new_height: int, at_time: float) -> bool:
        """Return True if a prospective block at height new_height would be in-time at at_time."""
        t0 = self.first_seen_time_by_height.get(int(new_height))
        return (t0 is None) or ((float(at_time) - float(t0)) <= self.tau)

    def _continuous_ws_for_parent(self, parent_id: str, cutoff_time: Optional[float]) -> Tuple[int, List[str]]:
        """Return (count, ids) of earliest-per-version WS for parent_id up to cutoff_time, continuous from v=0.

        Note: We maintain earliest-per-version incrementally at delivery time; thus, all cached
        entries reflect WS whose arrival time <= now. For callers passing `cutoff_time`=now, this
        is equivalent to filtering by time.
        """
        emap = self._ws_earliest_id_by_parent.get(parent_id, {})
        cc = int(self._ws_contig_count_by_parent.get(parent_id, 0))
        ids: List[str] = []
        for v in range(cc):
            wsid = emap.get(v)
            if wsid is None:
                break
            # If a historical cutoff_time earlier than arrival is requested, skip (rare case)
            if (cutoff_time is not None):
                t_ws = self._ws_arrival_time_by_id.get(wsid)
                if t_ws is None or float(t_ws) > float(cutoff_time):
                    break
            ids.append(wsid)
        return (len(ids), ids)

    def _prospective_step_if_mined_on(self, cand: Block, at_time: float) -> float:
        """Compute the immediate step weight for a new child mined on cand at at_time under WS-mode."""
        N = int(self.work_shares) if isinstance(self.work_shares, int) else 1
        if N <= 1:
            return 1.0
        h_new = int(cand.height) + 1
        if not self._is_in_time(h_new, at_time):
            return 0.0
        cnt, _ = self._continuous_ws_for_parent(cand.id, at_time)
        base = 1.0 / float(N)
        return float(base + base * float(cnt))

    def choose_parent_to_mine_upon(self, now: float, for_workshare: bool = False) -> Block:
        """WS-aware parent selection between current head and its parent.

        - Maximizes immediate prospective cumulative weight if we mine on the candidate now.
        - When for_workshare=True and N>1, adds a marginal +base if the new WS would extend
          the contiguous prefix for that parent at time now.
        - On ties, prefers the current head to avoid self-forks.
        """
        parent = self._select_head()
        N = int(self.work_shares) if isinstance(self.work_shares, int) else 1
        if N > 1 and parent is not None:
            candidates: List[Block] = [parent]
            if parent.parent_id is not None:
                par = self.blocks.get(parent.parent_id)
                if par is not None:
                    candidates.append(par)
            best_val = None
            best_cands: List[Block] = []
            base = 1.0 / float(N)
            for cand in candidates:
                step = self._prospective_step_if_mined_on(cand, now)
                if for_workshare and self._is_in_time(int(cand.height) + 1, now):
                    cnt, _ = self._continuous_ws_for_parent(cand.id, now)
                    next_ver = int(self.workshare_count_by_parent.get(cand.id, 0))
                    if next_ver == cnt:
                        step += base
                prospective = float(self.cum_block_weight.get(cand.id, 0.0)) + float(step)
                if (best_val is None) or (prospective > best_val):
                    best_val = prospective
                    best_cands = [cand]
                elif prospective == best_val:
                    best_cands.append(cand)
            if best_cands:
                parent = best_cands[0]
        return parent

    def _initial_step_weight_for_block(self, block: Block, t_recv: float) -> float:
        """Compute WS-mode initial step weight for a just-received block at t_recv and update counted set."""
        N = int(self.work_shares) if isinstance(self.work_shares, int) else 1
        base = 1.0 / float(N)
        if block.id not in self.in_time_blocks:
            return 0.0
        step_w = base
        pid = block.parent_id or "GENESIS"
        included: Set[str] = set(getattr(block, "included_ws_ids", []) or [])
        if included:
            counted = self._ws_counted_by_block.setdefault(block.id, set())
            # Count only included WS that arrived before this block
            for ws_id in list(included):
                t_ws = self._ws_arrival_time_by_id.get(ws_id)
                if (t_ws is not None) and (float(t_ws) <= float(t_recv)) and (ws_id not in counted):
                    step_w += base
                    counted.add(ws_id)
        return float(step_w)

    def _apply_late_ws_to_children(self, ws: WorkShare, received_time: float) -> bool:
        """Apply late WS within tau to eligible child blocks; returns True if any weights changed."""
        pid = ws.parent_id or "GENESIS"
        N = int(self.work_shares) if isinstance(self.work_shares, int) else 1
        base = 1.0 / float(max(1, N))
        changed = False
        for bid in list(self.children.get(pid, [])):
            if bid not in self.blocks:
                continue
            t_block = self.block_first_seen.get(bid)
            if t_block is None:
                continue
            blk_obj = self.blocks.get(bid)
            if blk_obj is None:
                continue
            if (received_time >= t_block) and (received_time <= t_block + self.tau) and (bid in self.in_time_blocks) and (ws.id in getattr(blk_obj, "included_ws_ids", [])):
                counted = self._ws_counted_by_block.setdefault(bid, set())
                if (ws.id not in counted):
                    counted.add(ws.id)
                    self.step_weight[bid] = float(self.step_weight.get(bid, 0.0)) + base
                    self._propagate_weight_delta_from(bid, base)
                    changed = True
        return changed

    def _propagate_weight_delta_from(self, bid: str, delta: float) -> None:
        """Propagate a step-weight increment delta from block bid to all its descendants."""
        if not delta:
            return
        # Update this block and all descendants cumulatively
        stack = [bid]
        while stack:
            x = stack.pop()
            self.cum_block_weight[x] = float(self.cum_block_weight.get(x, 0.0)) + delta
            stack.extend(self.children.get(x, []))

    def _update_selected_head(self) -> None:
        """Refresh the cached selected mining head according to current fork-resolution rules."""
        head = self._select_head()
        self.selected_head_id = head.id


    def _select_head(self) -> Block:
        """Apply fork-resolution rules to choose the mining head among current leaves."""
        # Fast path: if there is a unique top-height leaf and no competitors within (k-1)
        # heights below it, then by Rule 1 this leaf dominates by >= k and is selected.
        max_h = self.max_height
        top_ids = self.leaves_by_height.get(max_h, set())
        if top_ids:
            if len(top_ids) == 1:
                # Check if any competitor leaf exists in the window [max_h-(k-1), max_h-1]
                start_h = max(0, max_h - (self.k - 1))
                competitor_nearby = any(self.leaves_by_height.get(h) for h in range(start_h, max_h))
                if not competitor_nearby:
                    return self.blocks[next(iter(top_ids))]

        heads = self._current_heads()
        if not heads:
            # Should not happen (at least genesis is present)
            return self.blocks[next(iter(self.blocks))]

        # Rule 1: longest chain by >= k blocks (strictly one dominant longest)
        max_h = max(b.height for b in heads)
        longest = max(heads, key=lambda b: b.height)
        if len(heads) == 1:
            return longest
        second_h = max((b.height for b in heads if b is not longest), default=longest.height)
        if max_h - second_h >= self.k:
            return longest

        # Restrict candidates to heads within (k-1) of the top height when Rule 1 cannot resolve
        start_h = max(0, max_h - (self.k - 1))
        eligible = [b for b in heads if b.height >= start_h]
        # Rule 2: choose by maximal cumulative chain weight among eligible candidates
        max_w = max(self.cum_block_weight.get(b.id, 0) for b in eligible)
        best = [b for b in eligible if self.cum_block_weight.get(b.id, 0) == max_w]
        if len(best) == 1:
            return best[0]

        # Rule 3: break ties deterministically or uniformly at random
        if self.deterministic_selection:
            chosen_id = self._deterministic_choice([b.id for b in best])
            return self.blocks[chosen_id]
        else:
            return self._rng.choice(best)

    def _current_heads(self) -> List[Block]:
        max_h = self.max_height
        start_h = max(0, max_h - (2 * self.k))
        ids: List[str] = []
        for h in range(start_h, max_h + 1):
            ids.extend(self.leaves_by_height.get(h, set()))
        return [self.blocks[bid] for bid in sorted(ids)]

    def _prune_old_leaves(self) -> None:
        cutoff = max(0, int(self.max_height) - (2 * int(self.k)))
        if cutoff <= 0:
            return
        # Remove leaves and their height-index entries below cutoff
        to_drop_heights = [h for h in list(self.leaves_by_height.keys()) if int(h) < cutoff]
        for h in to_drop_heights:
            s = self.leaves_by_height.pop(h, None)
            if s:
                for bid in s:
                    self.leaves.discard(bid)


    def _iter_path_from_head(self, head_id: str) -> List[Block]:
        """Return the path [genesis, ..., head] as a list of Block objects."""
        path: List[Block] = []
        cur = self.blocks[head_id]
        while cur is not None:
            path.append(cur)
            if cur.parent_id is None:
                break
            cur = self.blocks.get(cur.parent_id)
        path.reverse()
        return path

    def _deterministic_choice(self, ids: List[str]) -> str:
        """Choose one id deterministically by hashing the sorted candidate ids (BLAKE2b).

        Steps:
        - Sort the candidate ids lexicographically.
        - Compute H = blake2b( join_with_NUL(sorted_ids) ).
        - idx = int.from_bytes(H, "big") % len(sorted_ids).
        - Return sorted_ids[idx].
        Deterministic across miners given identical candidate sets. 
        """
        if not ids:
            raise ValueError("_deterministic_choice requires a non-empty list")
        sorted_ids = sorted(ids)
        payload = b"\x00".join(s.encode("utf-8") for s in sorted_ids)
        digest = hashlib.blake2b(payload).digest()
        idx = int.from_bytes(digest, byteorder="big", signed=False) % len(sorted_ids)
        return sorted_ids[idx]

