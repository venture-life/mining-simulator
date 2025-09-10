from __future__ import annotations

import math
import random
import logging
from typing import Dict, List, Optional, Set, Tuple, Callable, Any
from .miner import Miner, Block
from . import policy as sm_policy

logger = logging.getLogger(__name__)

class SelfishMiner:
    """
    Wrapper around two Miner instances to support selfish-mining strategies.

    Views:
    - public: what is broadcast/known by the network
    - private: may include withheld blocks mined by this miner

    API:
    - on_mine(now): mines on the PRIVATE head and WITHHOLDS (returns None)
    - on_receive(block, t): deliver a block to the PUBLIC view
    - act(now): calls a unified policy decider (miner, now) -> int in {-1,0,1}
        -1 => adopt (align PRIVATE with PUBLIC and clear withheld)
         0 => hide (no-op)
         1 => publish exactly one withheld block (returns Block)

    Publishing semantics:
    - act(now) returns at most one withheld Block which the simulator broadcasts.
    - The simulator delivers to all miners—INCLUDING SELF—via its event queue.
      Self-delivery uses a tiny epsilon delay to serialize local sub-events.
    - Subsequent publish decisions occur on subsequent DELIVER events (no inner
      tight loop inside a single simulator event).
    """

    def __init__(self, miner_id: int, k: int, tau: float, *, genesis_id: str = "GENESIS", alpha: Optional[float] = None, policy: Optional[Callable[["SelfishMiner", float], Any]] = None) -> None:
        self.miner_id = int(miner_id)
        self.k = int(k)
        self.tau = float(tau)

        # Internal honest miners for the two views
        self.public = Miner(miner_id=self.miner_id, k=self.k, tau=self.tau, genesis_id=genesis_id)
        self.private = Miner(miner_id=self.miner_id, k=self.k, tau=self.tau, genesis_id=genesis_id)

        # Local sequence for ids when crafting blocks directly (if needed later)
        self._next_seq = 0

        # Queue of withheld blocks (kept only in private view until published)
        self._withheld: List[Block] = []


        # Track last event context to drive minimal publishing logic
        self._last_event: str = "init"  # one of {"init","mine","receive"}
        self._last_receive_from_competitor: bool = False

        # Metrics state (Bh, Bs, Diff_w, luck, last, published)
        # - Bh: public head height (honest chain length)
        # - Bs: private head height (selfish chain length)
        # - Diff_w: Wh - Ws (public chain weight minus private chain weight)
        # - last: 'h' or 's' (miner of the last mined block as perceived by this agent)
        # - published: number of withheld blocks we have published so far
        self.Bh: int = 0
        self.Bs: int = 0
        self.diff_w: int = 0
        self.luck: bool = False
        self.last: str = 'h'
        self.published: int = 0

        # Initialize metrics
        self._recompute_state()
        # Optional policy hook (pluggable):
        # - External policy (callable) should return an int in {-1, 0, 1} mapping to:
        #   -1 -> adopt, 0 -> hide, 1 -> publish_one
        # - External stepper: object with .step(miner, now) -> int in {-1,0,1}
        # - If none provided, default to heuristic base policy composed with a StreamingStepper
        #   whose step(miner, now) returns int in {-1,0,1}.
        self.policy: Optional[Callable[["SelfishMiner", float], Any]] = (policy if callable(policy) else sm_policy.heuristic_policy)
        if callable(policy):
            # Use external as-decider directly: fn(self, now)->int or obj.step(self, now)->int
            if hasattr(policy, 'step') and callable(getattr(policy, 'step')):
                self._decide_fn = getattr(policy, 'step')  # type: ignore[assignment]
            else:
                self._decide_fn = policy  # type: ignore[assignment]
            self._stepper = None  # not used for external policies
        else:
            # Internal default: wrap heuristic base policy in a StreamingStepper (int-returning)
            self._stepper = sm_policy.make_streaming_stepper(self.policy)
            self._decide_fn = self._stepper.step
        # Local-event id tracking (used for debugging/trace alignment)
        self._local_event_id: int = 0
        # Attacker's own hashrate share (alpha) if known; influences policy aggressiveness
        try:
            self.alpha: Optional[float] = (float(alpha) if alpha is not None else None)
        except Exception as e:
            logger.exception("Failed to parse alpha '%s'; defaulting to None", alpha)
            self.alpha = None

    # ------------------------- public API (mirror) -------------------------
    def on_receive(self, block: Block, received_time: float) -> None:
        """
        Receive a block from the network at received_time and update both views.
        We must clone the incoming block to avoid cross-view mutation of fields
        like height during connection.
        """
        # Always clone for PUBLIC deliver to avoid cross-view mutation
        pb_pub = self._clone_block(block)
        self.public.on_receive(pb_pub, received_time)
        
        self._last_event = "receive"
        self._last_receive_from_competitor = bool(block.miner_id is not None and block.miner_id != self.miner_id)
        if self._last_receive_from_competitor:
            self.last = 'h'
            # Only mirror into PRIVATE when we are not withholding; this keeps
            # private aligned with public in honest/no-withhold regimes, and avoids
            # contaminating the private view during withholding.
            if not self._withheld:
                # Clone the entire PUBLIC view into PRIVATE to guarantee parent-known invariant.
                # Calling private.on_receive directly can violate ordered-delivery assumptions.
                self.private = self._clone_miner_state(self.public)
            else:
                # While withholding, do NOT deliver competitor blocks into PRIVATE.
                # PRIVATE only tracks our withheld chain; it will be reconciled with PUBLIC
                # upon adopt/publish via _clone_miner_state to maintain parent-known invariants.
                pass
        # Increment local event counter (idempotence key)
        self._local_event_id += 1

    def on_mine(self, now: float) -> Optional[Block]:
        """
        Mine a block on the PRIVATE head and WITHHOLD it (return None).

        Publishing a withheld block is treated as a local PoP sub-event: the
        simulator may call act(now) multiple times at the same simulation time t.
        Each publish immediately updates the public chain and all local state,
        after which the policy is consulted again on the new state to decide
        whether to publish another withheld block.
        """
        # Choose parent:
        # - If withholding, extend the private withheld tip to keep the secret chain contiguous.
        # - Otherwise, mine on the FRP-selected private head (honest behavior when not withholding).
        if self._withheld:
            parent = self._withheld[-1]
        else:
            parent = self.private._select_head()
        new_id = f"{self.miner_id}:{self._next_seq}"
        self._next_seq += 1

        # Determine uncles: siblings of parent that are in-time (from PUBLIC knowledge)
        uncles: List[str] = []
        if parent.parent_id is not None:
            gp = parent.parent_id
            for sib_id in self.public.children.get(gp, []):
                if sib_id == parent.id:
                    continue
                if sib_id in self.public.in_time_blocks:
                    uncles.append(sib_id)
                    
        new_block = Block(
            id=new_id,
            parent_id=parent.id,
            miner_id=self.miner_id,
            height=parent.height + 1,
            uncles=uncles,
            created_time=now,
        )

        # Receive locally into PRIVATE view only; do not broadcast yet
        self.private.on_receive(new_block, received_time=now)
        self._withheld.append(new_block)
        self._last_event = "mine"
        self.last = 's'
        # Increment local event counter (idempotence key)
        self._local_event_id += 1
        return None

    # Backward/alternate naming: mirror "deliver" terminology used at simulator level
    def on_deliver(self, block: Block, received_time: float) -> None:
        """Alias for on_receive for API clarity."""
        return self.on_receive(block, received_time)

    # ------------------------- view forwarding (read-only) -----------------
    @property
    def blocks(self) -> Dict[str, Block]:
        return self.public.blocks

    @property
    def children(self) -> Dict[str, List[str]]:
        return self.public.children

    @property
    def leaves(self) -> Set[str]:
        return self.public.leaves

    @property
    def max_height(self) -> int:
        return self.public.max_height

    @property
    def cum_block_weight(self) -> Dict[str, int]:
        return self.public.cum_block_weight

    @property
    def blocks_by_height(self) -> Dict[int, Set[str]]:
        return self.public.blocks_by_height

    @property
    def in_time_blocks(self) -> Set[str]:
        return self.public.in_time_blocks

    @property
    def block_first_seen(self) -> Dict[str, float]:
        return self.public.block_first_seen

    # ------------------------- utilities for strategies --------------------
    def lead(self) -> int:
        """Return private_head_height - public_head_height."""
        pub_head = self.public.blocks[self.public.selected_head_id]
        # If withholding, measure lead against the current withheld tip;
        # otherwise, use the FRP-selected private head.
        if self._withheld:
            prv_head = self._withheld[-1]
        else:
            prv_head = self.private._select_head()
        return int(prv_head.height) - int(pub_head.height)


    

    # ------------------------- metrics and state ---------------------------
    def _recompute_state(self, *, now: Optional[float] = None) -> None:
        """Recompute the selfish-miner state tuple components.

        Diff_w = Wh - Ws; last and published kept incrementally.
        """
        # Use IDs for weight lookups
        pub_head_id = self.public.selected_head_id
        cf = None  # type: Optional[Miner]
        # Compute Ws as counterfactual "if we published now" when time is provided and we are withholding
        if now is not None and self._withheld:
            prv_head_id = self._withheld[-1].id
            cf = self._build_counterfactual_public(float(now))
            Ws = int(cf.cum_block_weight.get(prv_head_id, 0))
        else:
            prv_head_id = (self._withheld[-1].id if self._withheld else self.private.selected_head_id)
            Ws = int(self.private.cum_block_weight.get(prv_head_id, 0))
        # Public head weight
        Wh = int(self.public.cum_block_weight.get(pub_head_id, 0))
        self.diff_w = Wh - Ws

        # Race tie-break luck (counterfactual): when withholding and diff_w==0,
        # publish-now counterfactual selects OUR head under deterministic FRP.
        self.luck = False
        try:
            if cf is not None and int(self.diff_w) == 0:
                selected = cf._select_head()
                if selected is not None and selected.miner_id is not None:
                    self.luck = (int(selected.miner_id) == self.miner_id)
        except Exception as e:
            logger.exception("luck (counterfactual) computation failed; resetting luck=False")
            self.luck = False

    def _build_counterfactual_public(self, now: float) -> Miner:
        """Clone PUBLIC and deliver all withheld blocks with V2-like timing starting at `now`."""
        cf = self._clone_miner_state(self.public)
        try:
            # # sample_delay() ~ lognormal, capped at max_prop_delay = 0.5 * tau
            # max_prop_delay = 0.5 * float(self.tau)
            # if max_prop_delay <= 0.0:
            #     initial_delay = 0.0
            # else:
            #     sigma = 0.6
            #     mu = math.log(max(max_prop_delay / 2.0, 1e-9)) - 0.5 * sigma * sigma
            #     d = random.lognormvariate(mu, sigma)
            #     initial_delay = d if d <= max_prop_delay else max_prop_delay
            initial_delay = 2.5
            EPSILON_LOCAL_PUBLISH = 0.001
            t_cur = float(now) + float(initial_delay) + EPSILON_LOCAL_PUBLISH

            # If we have previously published part of our secret chain (popped from _withheld)
            # but PUBLIC has not yet seen those blocks (due to self-delivery epsilon in the simulator),
            # then cf (a clone of PUBLIC) may be missing parents of the current first withheld block.
            # Bridge any missing ancestor chain from PRIVATE before delivering the current withheld list.
            delivered_ids: Set[str] = set()
            if self._withheld:
                try:
                    parent_id = self._withheld[0].parent_id
                    missing_ancestors: List[Block] = []
                    while parent_id is not None and parent_id not in cf.blocks:
                        pb = self.private.blocks.get(parent_id)
                        if pb is None:
                            break
                        missing_ancestors.append(pb)
                        parent_id = pb.parent_id
                    # Deliver oldest ancestor first to satisfy parent-known invariant
                    for anc in reversed(missing_ancestors):
                        if anc.id in delivered_ids:
                            continue
                        cf.on_receive(self._clone_block(anc), received_time=t_cur)
                        delivered_ids.add(anc.id)
                        t_cur += EPSILON_LOCAL_PUBLISH
                except Exception:
                    # Best-effort; if we cannot repair ancestors from PRIVATE, proceed to try withheld list
                    pass

            # Now deliver the current withheld list in order
            for wb in self._withheld:
                if wb.id in delivered_ids:
                    continue
                cf.on_receive(self._clone_block(wb), received_time=t_cur)
                delivered_ids.add(wb.id)
                t_cur += EPSILON_LOCAL_PUBLISH
        except Exception as e:
            logger.exception("Counterfactual delivery failed while building cf PUBLIC; proceeding with partial state")
        return cf


    # ========================= PoP action space =============================
    def act(self, now: float) -> Optional[Block]:
        """Perform one streaming decision step (via decider) and execute it.

        Decider interface: returns an int in {-1, 0, 1}
        -1 => adopt (perish)
         0 => hide (no-op)
         1 => publish exactly one withheld block (if any)
        """

        # Recompute policy-relevant metrics at the current decision time
        self._recompute_state(now=now)

        # Call decider with the unified interface (miner, now) -> int
        try:
            decision = int(self._decide_fn(self, now))  # type: ignore[misc]
        except Exception as e:
            logger.exception("Policy decider raised; defaulting decision=0")
            decision = 0

        if decision < 0:
            if self._withheld:
                self._adopt_public()
            return None
        if decision > 0:
            return self._publish_one(now)
        return None

    # ------------------------- action helpers ------------------------------
    def _publish_one(self, now: float) -> Optional[Block]:
        """Publish exactly one withheld block if available; return the block or None."""
        if not self._withheld:
            return None
        b = self._withheld.pop(0)
        # Do not deliver to self here; simulator will broadcast (including self) with epsilon
        self.published += 1
        return b

    def _adopt_public(self) -> None:
        """Give up private chain: align private view to public and clear withheld."""
        self.private = self._clone_miner_state(self.public)
        self._withheld.clear()
        # Reset published counter to reflect "since adopt" semantics
        self.published = 0



    # ------------------------- cloning and selection utils -----------------
    def _clone_miner_state(self, src: Miner) -> Miner:
        """Deep-copy a Miner state (for runtime performance optimization).

        Semantics-preserving: copies all local views without recomputation by replay.
        """
        dst = Miner(miner_id=src.miner_id, k=src.k, tau=src.tau, genesis_id="GENESIS")
        # Clone blocks (dataclass) by value
        dst.blocks = {}
        for bid, b in src.blocks.items():
            dst.blocks[bid] = Block(
                id=b.id,
                parent_id=b.parent_id,
                miner_id=b.miner_id,
                height=b.height,
                uncles=list(b.uncles),
                created_time=b.created_time,
            )
        # Shallow-copy id structures (ids refer to dst.blocks keys)
        dst.children = {pid: list(ch) for pid, ch in src.children.items()}
        dst.leaves = set(src.leaves)
        dst.max_height = int(src.max_height)
        # Copy leaves_by_height index for fast head selection
        dst.leaves_by_height = {h: set(bids) for h, bids in getattr(src, 'leaves_by_height', {}).items()}
        # Copy timing and classification maps/sets
        dst.first_seen_time_by_height = dict(src.first_seen_time_by_height)
        dst.in_time_blocks = set(src.in_time_blocks)
        dst.block_first_seen = dict(src.block_first_seen)
        dst.blocks_by_height = {h: set(bids) for h, bids in src.blocks_by_height.items()}
        # Copy cumulative weights
        dst.cum_block_weight = dict(src.cum_block_weight)
        # Preserve selected head
        dst.selected_head_id = src.selected_head_id
        return dst



    # ------------------------- internals -----------------------------------
    @staticmethod
    def _clone_block(b: Block) -> Block:
        """Create a fresh Block instance to avoid cross-view mutation of fields."""
        return Block(
            id=b.id,
            parent_id=b.parent_id,
            miner_id=b.miner_id,
            height=0,  # receiver will derive
            uncles=list(b.uncles),
            created_time=b.created_time,
        )

    
