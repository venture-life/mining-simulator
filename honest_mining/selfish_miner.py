from __future__ import annotations

import math
import random
import logging
from typing import Dict, List, Optional, Set, Tuple, Callable, Any, Union
from .miner import Miner, Block, WorkShare
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

    Burn-out mode:
    - When the simulator toggles `burnout=True`, `on_mine()` switches to mining on the
      PUBLIC head and returns blocks immediately for publication, while `act()` drains
      any remaining withheld blocks before adopting the public chain.

    Policy integration:
    - External policies can be provided via the `policy` argument (callables returning
      {-1, 0, 1} or objects exposing `.step(miner, now) -> int`). When no policy is
      supplied, `act()` falls back to a minimal built-in heuristic driven by the
      internal bookkeeping fields (`lead`, `last`, `_withheld`, etc.).
    """

    def __init__(self, miner_id: int, k: int, tau: float, *, genesis_id: str = "GENESIS", deterministic_selection: bool = True, tie_break_seed: Optional[int] = None, alpha: Optional[float] = None, policy: Optional[Callable[["SelfishMiner", float], Any]] = None, work_shares: Optional[int] = None) -> None:
        self.miner_id = int(miner_id)
        # Sanity check: enforce k >= 1 (k=0 behaves like k=1 for 'strictly dominant longest')
        self.k = max(1, int(k))
        self.tau = float(tau)
        # Propagate tie-break style to internal honest miners
        self.deterministic_selection: bool = bool(deterministic_selection)
        self.tie_break_seed: Optional[int] = (int(tie_break_seed) if tie_break_seed is not None else None)
        # RNG for local stochastic decisions (e.g., work-share dice)
        self._rng = random.Random(self.tie_break_seed)

        # Internal honest miners for the two views
        self.public = Miner(
            miner_id=self.miner_id,
            k=self.k,
            tau=self.tau,
            genesis_id=genesis_id,
            deterministic_selection=self.deterministic_selection,
            tie_break_seed=self.tie_break_seed,
            work_shares=work_shares,
        )
        self.private = Miner(
            miner_id=self.miner_id,
            k=self.k,
            tau=self.tau,
            genesis_id=genesis_id,
            deterministic_selection=self.deterministic_selection,
            tie_break_seed=self.tie_break_seed,
            work_shares=work_shares,
        )

        # Local sequence for ids when crafting blocks directly (if needed later)
        self._next_seq = 0

        # Queue of withheld blocks (kept only in private view until published)
        self._withheld: List[Block] = []
        # Withheld work-shares by parent id (kept private until publishing the corresponding block)
        self._withheld_ws_by_parent: Dict[str, List[WorkShare]] = {}
        # Release queue for publishing a block and its associated WS in sequence
        self._pending_release: List[object] = []


        # Track last event context to drive minimal publishing logic
        self._last_event: str = "init"  # one of {"init","mine","receive"}
        self._last_receive_from_competitor: bool = False
        # Local-event id tracking (used for debugging/trace alignment)
        self._local_event_id: int = 0
        # Attacker's own hashrate share (alpha) if known; influences policy aggressiveness
        self.alpha: Optional[float] = (float(alpha) if alpha is not None else None)        

        # Metrics state (Diff_w, luck, last, published)
        # - Diff_w: Wh - Ws (public chain weight minus private chain weight)
        # - last: 'h' or 's' (miner of the last mined block as perceived by this agent)
        # - published: number of withheld blocks we have published so far
        self.diff_w: int = 0
        self.luck: bool = False
        self.last: str = 'h0'
        self.published: int = 0
        self.lead: int = 0
        # Burn-out mode: when enabled, transition towards honest behavior near cutoff.
        # - act(): aggressively publishes withheld blocks until empty, then adopts public.
        # - on_mine(): in burnout, mine on PUBLIC head and immediately publish (return Block).
        self.burnout: bool = False

        # Work-share configuration (N); if N<=1, behave as blocks-only
        try:
            self.work_shares: int = int(work_shares) if work_shares is not None else 1
        except Exception:
            self.work_shares = 1
        if self.work_shares <= 0:
            self.work_shares = 1

        # Optional policy hook (pluggable):
        # - External policy (callable) should return an int in {-1, 0, 1} mapping to:
        #   -1 -> adopt, 0 -> hide, 1 -> publish_one
        # - External stepper: object with .step(miner, now) -> int in {-1,0,1}
        # - When no policy is supplied we still cache the heuristic base policy so it can be
        #   inspected or wrapped externally, but `_decide_fn` is left unset so that `act()`
        #   uses its built-in fallback logic.
        self.policy: Optional[Callable[["SelfishMiner", float], Any]] = (policy if callable(policy) else sm_policy.heuristic_policy)
        if callable(policy):
            # Use external as-decider directly: fn(self, now)->int or obj.step(self, now)->int
            if hasattr(policy, 'step') and callable(getattr(policy, 'step')):
                self._decide_fn = getattr(policy, 'step')  # type: ignore[assignment]
            else:
                self._decide_fn = policy  # type: ignore[assignment]
            self._stepper = None  # not used for external policies
        else:
            # Internal fallback: defer to heuristic rules inside act() instead of a stepper.
            self._stepper = None  # StreamingStepper integration is currently opt-in.
            self._decide_fn = None



    # ------------------------- public API (mirror) -------------------------
    def on_receive(self, block: Block, received_time: float) -> None:
        """
        Receive a block from the network at `received_time` and update the PUBLIC view.

        Notes
        -----
        - Callers are expected to provide a fresh `Block` (the simulator already does) so it can
          be passed straight into the honest `Miner` without mutating shared state.
        - The private view remains unchanged here; withheld blocks stay isolated until
          `_adopt_public()` is triggered.
        """

        # Fast path: duplicate delivery → keep earliest first-seen and return
        if block.id in self.public.blocks:
            return

        self.public.on_receive(block, received_time)
        
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


    def on_receive_workshare(self, ws: WorkShare, received_time: float) -> None:
        """Forward work-share deliveries to the PUBLIC view for bookkeeping.

        Selfish miners ignore work-shares for strategy but keep counts per parent for
        versioning semantics when interoperability is needed.
        """
        self.public.on_receive_workshare(ws, received_time)
        if ws.parent_id in self.private.blocks:
            self.private.on_receive_workshare(ws,received_time)
        


    def on_mine(self, now: float):
        """
        Mine according to the current operating mode and work-share setting.

        - With work_shares = N > 1: mine a real block with probability 1/N; otherwise emit a
          WorkShare that references the PUBLIC head. In normal (non-burnout) mode, real blocks
          are mined on the PRIVATE head and withheld (return None).
        - Burn-out mode: extend the PUBLIC head; if a real block is mined, return it immediately.

        Publishing a withheld block is treated as a local PoP sub-event: the
        simulator may call act(now) multiple times at the same simulation time t.
        Each publish immediately updates the public chain and all local state,
        after which the policy is consulted again on the new state to decide
        whether to publish another withheld block.
        """
        # Decide Block vs WorkShare
        N = int(self.work_shares) if isinstance(self.work_shares, int) else 1
        is_block = (N <= 1) or (self._rng.random() <= (1.0 / float(N)))

        # Burn-out: honest-like behavior on PUBLIC head
        if self.burnout:
            parent_pub = self.public._select_head()
            if is_block:
                new_id = f"{self.miner_id}:{self._next_seq}"
                self._next_seq += 1
                uncles_pub: List[str] = []
                if parent_pub.parent_id is not None:
                    gp = parent_pub.parent_id
                    for sib_id in self.public.children.get(gp, []):
                        if sib_id == parent_pub.id:
                            continue
                        if sib_id in self.public.in_time_blocks:
                            uncles_pub.append(sib_id)
                # Build included WS ids at mine time (PUBLIC view) via helper: earliest per version, continuous from v=0
                N = int(self.work_shares) if isinstance(self.work_shares, int) else 1
                included_pub: List[str] = []
                if N > 1:
                    _cnt, included_pub = self.public._continuous_ws_for_parent(parent_pub.id, now)
                new_block_pub = Block(
                    id=new_id,
                    parent_id=parent_pub.id,
                    miner_id=self.miner_id,
                    height=parent_pub.height + 1,
                    uncles=uncles_pub,
                    created_time=now,
                    included_ws_ids=included_pub,
                )
                self._last_event = "mine"
                self.last = 's0'
                self._local_event_id += 1
                return new_block_pub
            else:
                pid = parent_pub.id
                ver = self.public.workshare_count_by_parent.get(pid, 0)
                ws_id = f"WS:{self.miner_id}:{pid}:{ver}"
                ws = WorkShare(id=ws_id, parent_id=pid, miner_id=self.miner_id, version=ver, created_time=now)
                self._last_event = "mine"
                self.last = 's0'
                self._local_event_id += 1
                return ws

        # Normal (non-burnout) mode
        if is_block:
            # Mine on PRIVATE head and withhold
            parent_prv = self.choose_parent_to_mine_upon(now)
            new_id = f"{self.miner_id}:{self._next_seq}"
            self._next_seq += 1
            # Determine uncles from PUBLIC view for reference
            uncles_pub: List[str] = []
            if parent_prv.parent_id is not None:
                gp = parent_prv.parent_id
                for sib_id in self.public.children.get(gp, []):
                    if sib_id == parent_prv.id:
                        continue
                    if sib_id in self.public.in_time_blocks:
                        uncles_pub.append(sib_id)
            # Build included WS ids from PRIVATE view via helper (PRIVATE ingests withheld + public WS for known parents)
            N = int(self.work_shares) if isinstance(self.work_shares, int) else 1
            included_prv: List[str] = []
            if N > 1:
                _cnt, included_prv = self.private._continuous_ws_for_parent(parent_prv.id, now)
            new_block = Block(
                id=new_id,
                parent_id=parent_prv.id,
                miner_id=self.miner_id,
                height=parent_prv.height + 1,
                uncles=uncles_pub,
                created_time=now,
                included_ws_ids=included_prv,
            )
            # Receive locally into PRIVATE view only; do not broadcast yet
            self.private.on_receive(new_block, received_time=now)
            self._withheld.append(new_block)
            self._last_event = "mine"
            self.last = 's0'
            self.lead = self.get_lead()
            self._local_event_id += 1
            return None
        else:
            # Withhold a WorkShare that references the PRIVATE branch (same parent as we intend to extend)
            parent_prv = self.choose_parent_to_mine_upon(now, for_workshare=True)
            pid = parent_prv.id
            base_ver = int(self.private.workshare_count_by_parent.get(pid, 0))
            local_ws = self._withheld_ws_by_parent.get(pid, [])
            ver = base_ver + len(local_ws)
            ws_id = f"WS:{self.miner_id}:{pid}:{ver}"
            ws = WorkShare(id=ws_id, parent_id=pid, miner_id=self.miner_id, version=ver, created_time=now)
            # Keep withheld and update PRIVATE view so our private weights reflect this WS immediately
            self._withheld_ws_by_parent.setdefault(pid, []).append(ws)

            self.private.on_receive_workshare(ws, received_time=now)

            self._last_event = "mine"
            self.last = 's0'
            self._local_event_id += 1
            return None


    def choose_parent_to_mine_upon(self, now: float, for_workshare: bool = False) -> Block:
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

        # WS-aware selection: compare PRIVATE head vs its parent using Miner helper to maximize immediate post-block cumulative weight.
        # If selecting for a work-share, also add marginal +base when the new WS would extend the contiguous prefix at now.
        N = int(self.work_shares) if isinstance(self.work_shares, int) else 1
        if N > 1 and parent is not None:
            candidates: List[Block] = [parent]
            if parent.parent_id is not None:
                par = self.private.blocks.get(parent.parent_id)
                if par is not None:
                    candidates.append(par)
            best_val: Optional[float] = None
            best_cands: List[Block] = []
            for cand in candidates:
                step = self.private._prospective_step_if_mined_on(cand, now)
                if for_workshare and self.work_shares > 1 and self.private._is_in_time(int(cand.height) + 1, now):
                    cnt, _ = self.private._continuous_ws_for_parent(cand.id, now)
                    next_ver = int(self.private.workshare_count_by_parent.get(cand.id, 0))
                    # If the next WS version equals the current contiguous prefix length, adding one now extends the prefix
                    if next_ver == cnt:
                        step += (1.0 / float(self.work_shares))
                prospective = float(self.private.cum_block_weight.get(cand.id, 0.0)) + float(step)
                if (best_val is None) or (prospective > best_val):
                    best_val = prospective
                    best_cands = [cand]
                elif prospective == best_val:
                    best_cands.append(cand)
            if best_cands:
                parent = self._rng.choice(best_cands) if len(best_cands) > 1 else best_cands[0]

        return parent


    def get_lead(self) -> int:
        """Return private_head_height - public_head_height."""
        pub_head = self.public.blocks[self.public.selected_head_id]
        prv_head = self.private.blocks[self.private.selected_head_id]

        self.diff_w = self.private.cum_block_weight.get(self.private.selected_head_id, 0) - self.public.cum_block_weight.get(self.public.selected_head_id, 0)
        return int(prv_head.height) - int(pub_head.height)        


    def act(self, now: float) -> Optional[Union[Block, WorkShare]]:
        """Perform one decision step and execute it.

        - If an external decider (`_decide_fn`) is configured, we invoke it once and interpret
          the returned int in {-1, 0, 1} as adopt/hide/publish-one.
        - Otherwise we fall back to the built-in heuristic that inspects `lead`, `last`,
          `_withheld`, and `burnout` state to choose among the same {-1, 0, 1} actions.
        
        Returns
        -------
        Optional[Block | WorkShare]
            When publishing, this returns either the next withheld Block or one of its
            associated WorkShares as the release queue drains. Otherwise, returns None.
        """

        # If we have a pending release batch (block and its WS), prioritize publishing.
        if self._pending_release:
            decision = 1
        # Burn-out mode: aggressively drain withheld, then align to public.
        elif self.burnout:
            if self._withheld:
                decision = 1
            else:
                decision = -1
        elif self._decide_fn:
            try:
                # Call decider with the unified interface (miner, now) -> int
                decision = int(self._decide_fn(self, now))  # type: ignore[misc]
            except Exception as e:
                logger.exception("Policy decider raised; defaulting decision=0")
                decision = 0                    
        # Default SM1 behaviour if no decider is provided
        # we found a block (s0), or get a callback when we recently broadcasted one (s1)
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
            #     # publish one, it will trigger s1 with lead = 0 (line 437)
            #     decision = 1
            # elif self.lead == 1:
            #     # publish all, it will trigger s1 with lead = 1 (line 432)
            #     decision = 1

            else: # self.lead >= 2
                # publish one, it will trigger s1 with lead >= 2 (line 437)
                decision = 1
            # Longer re-orgs: Only feasible if pure PoW, no PoP (ie, k = 1, D=0 (effectively first-seen))
            # elif self.lead < 2:
            #     decision = 1
            # else:
            #     decision = 0

        if decision < 0:
            self._adopt_public()
            return None
        if decision > 0:
            return self._publish_one(now)
        return None


    # ------------------------- actions -------------------------------------
    def _publish_one(self, now: float) -> Optional[Union[Block, WorkShare]]:
        """
        Publish the next item in the release queue, or if empty:
        - Take one withheld block and enqueue its withheld WS first (same parent.id), then the block.
        Returns one object (WS first, then Block) for the simulator to broadcast.
        """
        # If we have a pending release batch, continue publishing it
        if self._pending_release:
            item = self._pending_release.pop(0)
            if isinstance(item, Block):
                # Do not deliver to self here; simulator will broadcast (including self) with epsilon
                self.published += 1
            return item
        # Seed a new release batch with the next withheld block
        if not self._withheld:
            return None
        b = self._withheld.pop(0)
        pid = b.parent_id
        ws_list = list(self._withheld_ws_by_parent.get(pid, []))
        # Sort WS by version to keep sequence consistent
        try:
            ws_list.sort(key=lambda w: int(getattr(w, 'version', 0) or 0))
        except Exception:
            pass
        # Enqueue WS first, then the associated block
        self._pending_release = ws_list + [b]
        # Clear the withheld WS for this parent now that they will be published
        if pid is not None:
            self._withheld_ws_by_parent.pop(pid, None)
        item = self._pending_release.pop(0)
        if isinstance(item, Block):
            self.published += 1
        return item

    def _adopt_public(self) -> None:
        """Give up private chain: align private view to public and clear withheld."""
        self.private = self._clone_miner_state(self.public)
        self._withheld.clear()
        self._withheld_ws_by_parent.clear()
        self._pending_release.clear()
        # Reset published counter to reflect "since adopt" semantics
        self.published = 0
        self.lead = self.get_lead()


    # ------------------------- view forwarding (read-only) -----------------

    # Backward/alternate naming: mirror "deliver" terminology used at simulator level
    def on_deliver(self, block: Block, received_time: float) -> None:
        """Alias for on_receive for API clarity."""
        return self.on_receive(block, received_time)

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
    def cum_block_weight(self) -> Dict[str, float]:
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
            work_shares=getattr(src, 'work_shares', 1),
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
                included_ws_ids=list(getattr(b, "included_ws_ids", []) or []),
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
        # Copy cumulative and step weights
        dst.cum_block_weight = dict(src.cum_block_weight)
        try:
            dst.step_weight = dict(getattr(src, 'step_weight', {}))
        except Exception:
            pass
        # Copy WS bookkeeping if present
        try:
            dst.workshare_count_by_parent = dict(getattr(src, 'workshare_count_by_parent', {}))
            dst._seen_workshare_ids = set(getattr(src, '_seen_workshare_ids', set()))
            dst._ws_arrivals_by_parent = {k: list(v) for k, v in getattr(src, '_ws_arrivals_by_parent', {}).items()}
            dst._ws_counted_by_block = {k: set(v) for k, v in getattr(src, '_ws_counted_by_block', {}).items()}
        except Exception:
            pass
        # Preserve selected head
        dst.selected_head_id = src.selected_head_id
        # Preserve RNG state for random tie-breaking so CF matches continuation
        try:
            if hasattr(src, '_rng') and hasattr(dst, '_rng'):
                dst._rng.setstate(src._rng.getstate())
        except Exception:
            pass
        return dst


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
            included_ws_ids=list(getattr(b, "included_ws_ids", []) or []),
        )

    
