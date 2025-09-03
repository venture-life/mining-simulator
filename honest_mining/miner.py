from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple, Callable
import hashlib


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


class Miner:
    """
    Honest miner maintaining a local block tree and selecting a mining head using
    the specified fork-resolution rules.

    Fork resolution (evaluated on known heads):
    1) If one chain is longer than all others by at least k blocks, mine on the
       longest chain.
    2) Otherwise choose the chain with the largest weight.
    3) If multiple chains share the largest weight, choose deterministically by
       hashing the sorted candidate ids (BLAKE2b) and selecting index = H % len(candidates).

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

    def __init__(self, miner_id: int, k: int, tau: float, *, genesis_id: str = "GENESIS") -> None:
        self.miner_id = miner_id
        self.k = int(k)
        self.tau = float(tau)

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

        # Miner-local cumulative weight per block id
        self.cum_block_weight: Dict[str, int] = {}

        # Simple local id generator for self-mined blocks
        self._next_seq = 0

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

    def on_mine(self, now: float) -> Block:
        """
        Called when this miner wins the lottery at time `now`.
        Select a parent head by fork-resolution rules, create a new block, embed
        eligible uncle references, and return the new block (for broadcasting).
        """
        parent = self._select_head()
        # Track the selected parent head at the time of mining
        self.selected_head_id = parent.id
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

        new_block = Block(id=new_id, parent_id=parent.id, miner_id=self.miner_id, height=parent.height + 1, uncles=uncles, created_time=now)

        # Locally receive our own block immediately
        self.on_receive(new_block, received_time=now)
        return new_block

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

        # Compute and store cumulative weight incrementally
        # Step weight: only if this block is in-time, count 1 + number of in-time uncles it references
        step_w = 0
        if block.id in self.in_time_blocks:
            step_w = 1
            for u in block.uncles:
                if u in self.in_time_blocks:
                    step_w += 1
        parent_w = 0 if block.parent_id is None else self.cum_block_weight.get(block.parent_id, 0)
        self.cum_block_weight[block.id] = parent_w + step_w

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
                competitor_nearby = False
                if self.k > 0:
                    start_h = max(0, max_h - (self.k - 1))
                    for h in range(start_h, max_h):
                        if self.leaves_by_height.get(h):
                            competitor_nearby = True
                            break
                # If k == 0, Rule 1 cannot establish a unique dominance; fall through.
                if self.k > 0 and not competitor_nearby:
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

        # Rule 2: choose by maximal cumulative chain weight
        max_w = max(self.cum_block_weight.get(b.id, 0) for b in heads)
        best = [b for b in heads if self.cum_block_weight.get(b.id, 0) == max_w]
        if len(best) == 1:
            return best[0]

        # Rule 3: deterministic random among ties
        chosen_id = self._deterministic_choice([b.id for b in best])
        return self.blocks[chosen_id]

    def _current_heads(self) -> List[Block]:
        """Return blocks that have no known children (leaves)."""
        return [self.blocks[bid] for bid in self.leaves]


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

