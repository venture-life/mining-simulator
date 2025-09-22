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

    def __init__(self, miner_id: int, k: int, tau: float, *, genesis_id: str = "GENESIS", deterministic_selection: bool = True, tie_break_seed: Optional[int] = None, alpha: Optional[float] = None, policy: Optional[Callable[["SelfishMiner", float], Any]] = None) -> None:
        self.miner_id = int(miner_id)
        # Sanity check: enforce k >= 1 (k=0 behaves like k=1 for 'strictly dominant longest')
        self.k = max(1, int(k))
        self.tau = float(tau)
        # Propagate tie-break style to internal honest miners
        self.deterministic_selection: bool = bool(deterministic_selection)
        self.tie_break_seed: Optional[int] = (int(tie_break_seed) if tie_break_seed is not None else None)

        # Internal honest miners for the two views
        self.public = Miner(miner_id=self.miner_id, k=self.k, tau=self.tau, genesis_id=genesis_id, deterministic_selection=self.deterministic_selection, tie_break_seed=self.tie_break_seed)
        self.private = Miner(miner_id=self.miner_id, k=self.k, tau=self.tau, genesis_id=genesis_id, deterministic_selection=self.deterministic_selection, tie_break_seed=self.tie_break_seed)

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
        self.last: str = 'h0'
        self.published: int = 0
        self.lead = 0
        # Burn-out mode: when enabled, transition towards honest behavior near cutoff.
        # - act(): aggressively publishes withheld blocks until empty, then adopts public.
        # - on_mine(): in burnout, mine on PUBLIC head and immediately publish (return Block).
        self.burnout: bool = False

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
        # Policy-controlled toggle: when hiding and not withholding, prefer mining on the
        # second-last block if the selected head is not ours but its parent is ours.
        # Set this from the policy via: miner.prefer_second_last_on_hide = True/False
        # to be explored some day
        # self.prefer_second_last_on_hide: bool = False

    def get_lead(self) -> int:
        """Return private_head_height - public_head_height."""
        pub_head = self.public.blocks[self.public.selected_head_id]
        prv_head = self.private.blocks[self.private.selected_head_id]

        return int(prv_head.height) - int(pub_head.height)        

    # ------------------------- public API (mirror) -------------------------
    def on_receive(self, block: Block, received_time: float) -> None:
        """
        Receive a block from the network at received_time and update both views.
        We must clone the incoming block to avoid cross-view mutation of fields
        like height during connection.
        """

        # Fast path: duplicate delivery → keep earliest first-seen and return
        if block.id in self.public.blocks:
            return

        # Always clone for PUBLIC deliver to avoid cross-view mutation
        pb_pub = self._clone_block(block)
        self.public.on_receive(pb_pub, received_time)
        
        self._last_event = "receive"
        self._last_receive_from_competitor = bool(block.miner_id is not None and block.miner_id != self.miner_id)
        if self._last_receive_from_competitor:
            # only log lead if it was a new block, ie. not one we published ourselfs... that would be doublecounting lead, since it's already in on_mine
            self.lead = self.get_lead()
            if self.public.blocks[block.parent_id].miner_id == self.miner_id:
                # extends us
                self.last = 'h1'
            else:
                # extends rando
                self.last = 'h0'
        else:
            self.last = 's1'



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
        # If in burn-out mode, switch to honest-like behavior: mine on PUBLIC head and publish immediately.
        if self.burnout:
            # Select parent from PUBLIC view
            parent_pub = self.public._select_head()
            new_id = f"{self.miner_id}:{self._next_seq}"
            self._next_seq += 1
            # Determine uncles from PUBLIC siblings of the chosen parent
            uncles_pub: List[str] = []
            if parent_pub.parent_id is not None:
                gp = parent_pub.parent_id
                for sib_id in self.public.children.get(gp, []):
                    if sib_id == parent_pub.id:
                        continue
                    if sib_id in self.public.in_time_blocks:
                        uncles_pub.append(sib_id)
            new_block_pub = Block(
                id=new_id,
                parent_id=parent_pub.id,
                miner_id=self.miner_id,
                height=parent_pub.height + 1,
                uncles=uncles_pub,
                created_time=now,
            )
            # Do not mutate local state here; simulator will broadcast and then deliver (including to self)
            self._last_event = "mine"
            self.last = 's0'
            # lead will be recomputed on delivery
            self._local_event_id += 1
            return new_block_pub

        # Choose parent:
        # - Default: use the PRIVATE view's head per fork rules.
        # - Special case (Rule 3 tie and non-deterministic selection):
        #     prefer a head mined by us; else prefer a head whose parent was mined by us; else uniform random.
        
        parent = None
        if not self.deterministic_selection:
            try:
                heads = self.private._current_heads()
            except Exception:
                heads = []
            if heads:
                # Rule 1: longest chain by >= k blocks (strict dominance)
                max_h = max(b.height for b in heads)
                longest = max(heads, key=lambda b: b.height)
                if len(heads) == 1:
                    parent = longest
                else:
                    second_h = max((b.height for b in heads if b is not longest), default=longest.height)
                    if max_h - second_h >= self.k:
                        parent = longest
                    else:
                        # Restrict candidates to heads within (k-1) of the top height when Rule 1 cannot resolve
                        start_h = max(0, max_h - (self.k - 1))
                        eligible = [b for b in heads if b.height >= start_h]
                        # Rule 2: maximal cumulative chain weight among eligible candidates
                        max_w = max(self.private.cum_block_weight.get(b.id, 0) for b in eligible)
                        best = [b for b in eligible if self.private.cum_block_weight.get(b.id, 0) == max_w]
                        if len(best) > 1:
                            # Rule 3 tie: apply biased tie-break
                            mine_heads = [b for b in best if b.miner_id == self.miner_id]
                            if mine_heads:
                                parent = self.private._rng.choice(mine_heads)
                            else:
                                parent_mine_heads = []
                                for b in best:
                                    pid = b.parent_id
                                    if pid is not None:
                                        pb = self.private.blocks.get(pid)
                                        if pb is not None and pb.miner_id == self.miner_id:
                                            parent_mine_heads.append(b)
                                if parent_mine_heads:
                                    parent = self.private._rng.choice(parent_mine_heads)
                                else:
                                    parent = self.private._rng.choice(best)
                        elif best:
                            parent = best[0]

        if parent is None:
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
        self.last = 's0'
        self.lead = self.get_lead()

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



    # ------------------------- metrics and state ---------------------------
    def _recompute_state(self, *, now: Optional[float] = None) -> None:
        """Recompute the selfish-miner state tuple components.

        Diff_w = Wh - Ws; last and published kept incrementally.
        """
        self.luck = False

        # Use IDs for weight lookups
        if (self.k <= 1 or self.tau <= 0.0) and not self.deterministic_selection:
            self.diff_w = 0
        else:
            pub_head_id = self.public.selected_head_id
            cf = None  # type: Optional[Miner]
            # Compute Ws as counterfactual "if we published now" when time is provided and we are withholding
            if now is not None and self._withheld:
                prv_head_id = self._withheld[-1].id
                cf = self._build_counterfactual_public(float(now),only_first = False)
                Ws = int(cf.cum_block_weight.get(prv_head_id, 0))
            else:
                prv_head_id = self.private.selected_head_id
                Ws = int(self.private.cum_block_weight.get(prv_head_id, 0))

            Wh = int(self.public.cum_block_weight.get(pub_head_id, 0))
            self.diff_w = Wh - Ws


    def _build_counterfactual_public(self, now: float, *, only_first: bool = False) -> Miner:
        """Clone PUBLIC and deliver withheld blocks with V2-like timing starting at `now`.

        If only_first is True, deliver only the first withheld block (plus any missing ancestors).
        Otherwise, deliver the entire withheld list in order (plus any missing ancestors for the first).
        """
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
            initial_delay = 0.01
            EPSILON_LOCAL_PUBLISH = 0.000
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
                    logger.exception("Counterfactual metric computation impaired due to missing ancestors; we could not repair ancestors from PRIVATE, proceeding ...")

            # Now deliver the current withheld set according to mode (first-only or all)
            to_deliver = []
            if self._withheld:
                if only_first:
                    to_deliver = [self._withheld[0]]
                else:
                    to_deliver = list(self._withheld)

            for wb in to_deliver:
                if wb.id in delivered_ids:
                    continue
                cf.on_receive(self._clone_block(wb), received_time=t_cur)
                if self.deterministic_selection and wb.id == self._withheld[0].id:
                    selected = cf._select_head()
                    if selected is not None and selected.miner_id is not None:
                        self.luck = (int(selected.miner_id) == self.miner_id)
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
        # self._recompute_state(now=now)


        # Call decider with the unified interface (miner, now) -> int
        try:
            # decision = int(self._decide_fn(self, now))  # type: ignore[misc]
            # Burn-out mode: aggressively drain withheld, then align to public.
            if self.burnout:
                if self._withheld:
                    decision = 1
                else:
                    decision = -1
            elif self.last == "s0" or self.last == "s1":
                if self.lead == 1 and self.last == "s1":
                    if self._withheld:
                        decision = 1
                    else:
                        decision = 0
                else:
                    decision = 0
            else: # others find a block
                if self.lead < 0:
                    decision = -1
                # elif self.lead == 0:
                #     # publish one, it will trigger s1 with lead = 0 (line 443)
                #     decision = 1
                # elif self.lead == 1:
                #     # publish all, it will trigger s1 with lead = 1 (line 438)
                #     decision = 1

                else: # self.lead >= 2
                    # publish one, it will trigger s1 with lead >= 2 (line 443)
                    decision = 1
                # Longer re-orgs: Only feasible if pure PoW, no PoP (ie, k = 1, D=0 (effectively first-seen))
                # elif self.lead < 2:
                #     decision = 1
                # else:
                #     decision = 0






        except Exception as e:
            logger.exception("Policy decider raised; defaulting decision=0")
            decision = 0

        self.lead = self.get_lead()
        if decision < 0:
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
        self.lead = self.get_lead()



    # ------------------------- cloning and selection utils -----------------
    def _clone_miner_state(self, src: Miner) -> Miner:
        """Deep-copy a Miner state (for runtime performance optimization).

        Semantics-preserving: copies all local views without recomputation by replay.
        """
        dst = Miner(
            miner_id=src.miner_id,
            k=src.k,
            tau=src.tau,
            genesis_id="GENESIS",
            deterministic_selection=getattr(src, 'deterministic_selection', True),
            tie_break_seed=getattr(src, 'tie_break_seed', None),
        )
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
        # Preserve RNG state for random tie-breaking so CF matches continuation
        try:
            if hasattr(src, '_rng') and hasattr(dst, '_rng'):
                dst._rng.setstate(src._rng.getstate())
        except Exception:
            pass
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

    
