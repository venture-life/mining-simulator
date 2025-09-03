from __future__ import annotations

from typing import Dict, List, Optional, Set, Tuple, Callable
from .miner import Miner, Block

class SelfishMiner:
    """
    Wrapper around two Miner instances to support selfish-mining strategies.

    This class keeps:
    - a public view (what is broadcast/known by the network), and
    - a private view (which may include withheld blocks mined by this miner).

    It mirrors the Miner API (on_receive/on_mine) so it can be used in place of
    Miner by the simulator. For now, on_mine defaults to honest-compatible
    behavior (immediate broadcast) to remain compatible with the current
    simulator, while still maintaining a private view that can diverge if/when
    a withholding policy is enabled at the simulator level.

    """

    def __init__(self, miner_id: int, k: int, tau: float, *, genesis_id: str = "GENESIS", alpha: Optional[float] = None) -> None:
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

        # Mirror commonly read attributes for compatibility (updated from public)
        self.selected_head_id: str = self.public.selected_head_id

        # Track last event context to drive minimal publishing logic
        self._last_event: str = "init"  # one of {"init","mine","receive"}
        self._last_receive_from_competitor: bool = False

        # Metrics state (Bh, Bs, Diff_w, luck, last, published)
        # - Bh: public head height (honest chain length)
        # - Bs: private head height (selfish chain length)
        # - Diff_w: Wh - Ws (public chain weight minus private chain weight)
        # - luck: True if there exists a withheld non-late block that embeds an honest in-time uncle
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
        # Optional policy hook: Callable[[SelfishMiner, float], str]
        self.policy: Optional[Callable[["SelfishMiner", float], str]] = (lambda m, t: m._heuristic_policy(t))
        # Local-event idempotence guards
        self._local_event_id: int = 0
        self._last_acted_event_id: int = -1
        # Attacker's own hashrate share (alpha) if known; influences policy aggressiveness
        try:
            self.alpha: Optional[float] = (float(alpha) if alpha is not None else None)
        except Exception:
            self.alpha = None
        # Cache for planner results per local event (for runtime performance optimization)
        self._plan_cache_event_id: int = -1
        self._plan_cache: Optional[Dict[str, Optional[int]]] = None

    # ------------------------- public API (mirror) -------------------------
    def on_receive(self, block: Block, received_time: float) -> None:
        """
        Receive a block from the network at received_time and update both views.
        We must clone the incoming block to avoid cross-view mutation of fields
        like height during connection.
        """
        pb = self._clone_block(block)
        rb = self._clone_block(block)
        self.public.on_receive(pb, received_time)
        self.private.on_receive(rb, received_time)
        self.selected_head_id = self.public.selected_head_id
        # Remember that we just processed a receive event, and whether it was a competitor's block
        self._last_event = "receive"
        self._last_receive_from_competitor = bool(block.miner_id is not None and block.miner_id != self.miner_id)
        if self._last_receive_from_competitor:
            self.last = 'h'
        # Update metrics after processing receive
        self._recompute_state()
        # Increment local event counter (idempotence key)
        self._local_event_id += 1

    def on_mine(self, now: float) -> Optional[Block]:
        """
        Mine a block on the PRIVATE head and WITHHOLD it (return None).

        The simulator's V2 loop will call a policy via decide_action() and then act()
        after each event to determine if/what to publish.
        """
        # Select private head to extend
        parent = self.private._select_head()
        new_id = f"{self.miner_id}:{self._next_seq}"
        self._next_seq += 1

        # Determine uncles: siblings of parent that are in-time (from this miner's view)
        uncles: List[str] = []
        if parent.parent_id is not None:
            gp = parent.parent_id
            for sib_id in self.private.children.get(gp, []):
                if sib_id == parent.id:
                    continue
                if sib_id in self.private.in_time_blocks:
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
        # Update metrics after mining
        self._recompute_state()
        # Do not change self.selected_head_id: mirrors PUBLIC view
        # Increment local event counter (idempotence key)
        self._local_event_id += 1
        return None

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
        prv_head = self.private.blocks[self.private.selected_head_id]
        return int(prv_head.height) - int(pub_head.height)

    def publish_all(self, now: float) -> List[Block]:
        """
        Publish all withheld blocks to the public view in order of creation.
        Returns the list of published blocks (clones of the private ones).
        """
        published: List[Block] = []
        for b in self._withheld:
            cb = self._clone_block(b)
            self.public.on_receive(cb, received_time=now)
            published.append(cb)
        self._withheld.clear()
        self.selected_head_id = self.public.selected_head_id
        # Track count and refresh metrics
        self.published += len(published)
        self._recompute_state()
        return published

    def decide_action(self, now: float) -> str:
        """Return a PoP action to execute at time `now`.

        If a custom policy is set (self.policy), use it. Otherwise, fall back to a
        default heuristic:
        - If the last event was a competitor's receive and we have withheld blocks:
            * If we can match weights with minimal publish: return 'match'.
            * Else if we can override (>=k longer or heavier): return 'override'.
            * Else: 'hide'.
        - Otherwise: 'hide'.
        """
        # Idempotence-by-event: if we've already acted for this local event, do nothing
        if self._local_event_id == self._last_acted_event_id:
            return 'hide'
        if self.policy is not None:
            try:
                act = self.policy(self, now)
                if isinstance(act, str):
                    return act
            except Exception:
                # Fall back to default heuristic on policy errors
                pass

        return self._default_policy(now)

    def _default_policy(self, now: float) -> str:
        if not self._withheld:
            # Nothing to reveal
            self._last_receive_from_competitor = False
            return 'hide'
        if self._last_event == 'receive' and self._last_receive_from_competitor:
            # Prefer weight tie if feasible
            n_match = self._plan_publish_match(now)
            if n_match is not None:
                # Do not reset the competitor flag here; v2 will call act() next
                return 'match'
            n_override = self._plan_publish_override(now)
            if n_override is not None:
                return 'override'
            return 'hide'
        # Default: keep withholding
        return 'hide'

    def _heuristic_policy(self, now: float) -> str:
        """Simple heuristic based on the 6-tuple metrics and action feasibility.

        Uses (Bh, Bs, Diff_w, luck, last, published) to decide among
        {adopt, match, override, even, hide}. Consults the planner helpers to
        ensure the chosen action is feasible under current state.

        Preference (when last == 'h' and we have withheld):
        - Prefer the action that achieves the goal with the smallest positive n.
          Typically override > match > even, but if override is much costlier
          and a cheap match exists (especially when lucky), choose match.
        - If nothing feasible and we are not ahead: adopt; else hide.

        After self-mining (last == 's'): keep withholding (hide).

        Adaptivity by (alpha, k):
        - For alpha >= ~1/3: more aggressive overriding (allow higher publish cost).
        - For 1/4 <= alpha < 1/3: prefer match; allow only cheap overrides.
        - For alpha < 1/4 or unknown: conservative; avoid overrides unless trivially cheap.
        - Smaller k slightly increases willingness to override.
        """
        lead = int(self.Bs) - int(self.Bh)
        have_secret = len(self._withheld) > 0
        a = float(self.alpha) if self.alpha is not None else 0.0
        k = int(self.k)

        # No secrets to reveal
        if not have_secret:
            # If competitor just advanced and we're not ahead, align with public
            if self.last == 'h' and self.Bs <= self.Bh:
                return 'adopt'
            return 'hide'

        # React only when the last perceived miner was honest (competitor)
        if self.last == 'h':
            n_over = self._plan_publish_override(now)
            n_match = self._plan_publish_match(now)
            n_even = self._plan_publish_even(now)

            # Normalize to positive ints where applicable
            over_ok = (n_over is not None and n_over > 0)
            match_ok = (n_match is not None and n_match > 0)
            even_ok = (n_even is not None and n_even > 0)

            # Alpha- and k-adaptive aggressiveness thresholds
            # Allowance for overriding cost (in number of blocks to publish)
            if a >= 1.0/3.0:
                allow_over = 2
            elif a >= 1.0/4.0:
                allow_over = 1
            else:
                allow_over = 0
            # Smaller k -> easier to win by length; increase allowance slightly
            if k <= 2:
                allow_over += 1

            # Prefer override when it's cheap enough and not much worse than alternatives
            if over_ok and n_over <= allow_over and (not match_ok or n_over <= n_match) and (not even_ok or n_over <= n_even) and not self.luck:
                return 'override'
            # Prefer match when feasible, especially when lucky or override is costly
            if match_ok and (not over_ok or n_match < n_over or self.luck or n_over > allow_over):
                return 'match'
            # High-alpha fallback: still take override if feasible (even if costly)
            if over_ok and a >= 1.0/3.0:
                return 'override'
            # Even as a last resort to maintain height competitiveness
            if even_ok and (a >= 1.0/3.0 or (a >= 1.0/4.0 and k <= 2)):
                return 'even'

            # If we cannot improve and we're not ahead, give up
            if lead <= 0:
                return 'adopt'
            return 'hide'

        # After self-mining, prefer to keep building the private lead
        return 'hide'

    # ------------------------- metrics and state ---------------------------
    def _recompute_state(self) -> None:
        """Recompute the selfish-miner state tuple components.

        Bh = public head height; Bs = private head height;
        Diff_w = Wh - Ws; luck as defined in the prompt; last and published kept incrementally.
        """
        # Heads
        pub_head = self.public.blocks[self.public.selected_head_id]
        prv_head = self.private.blocks[self.private.selected_head_id]
        self.Bh = int(pub_head.height)
        self.Bs = int(prv_head.height)
        # Weights
        Wh = int(self.public.cum_block_weight.get(pub_head.id, 0))
        Ws = int(self.private.cum_block_weight.get(prv_head.id, 0))
        self.diff_w = Wh - Ws
        # Lucky block detection among withheld blocks
        self.luck = False
        if self._withheld:
            in_time_private = self.private.in_time_blocks
            for b in self._withheld:
                if b.id not in in_time_private:
                    continue  # secret but late → not lucky
                # Any honest in-time uncle?
                for u in b.uncles:
                    ub = self.private.blocks.get(u)
                    if ub is None:
                        continue
                    if u in in_time_private and (ub.miner_id is not None and int(ub.miner_id) != self.miner_id):
                        self.luck = True
                        break
                if self.luck:
                    break

    def get_state_tuple(self) -> Tuple[int, int, int, bool, str, int]:
        """Return (Bh, Bs, Diff_w, luck, last, published)."""
        return (self.Bh, self.Bs, self.diff_w, self.luck, 's' if self.last == 's' else 'h', self.published)

    # ========================= PoP action space =============================
    def act(self, action: str, now: float) -> List[Block]:
        """Execute a PoP action and return blocks to broadcast."""
        # Mark that we've acted for the current local event (at-most-once per event)
        self._last_acted_event_id = self._local_event_id
        a = (action or '').strip().lower()
        if a == 'adopt':
            self._adopt_public()
            # Clear competitor flag after acting so we don't react repeatedly to the same event
            self._last_receive_from_competitor = False
            return []
        elif a == 'override':
            # Use cached planning results (for runtime performance optimization)
            n = self._get_plan(now).get('n_override')
            res = self._publish_n(now, n) if (n is not None and n > 0) else []
            self._last_receive_from_competitor = False
            return res
        elif a == 'match':
            # Use cached planning results (for runtime performance optimization)
            n = self._get_plan(now).get('n_match')
            res = self._publish_n(now, n) if (n is not None and n > 0) else []
            self._last_receive_from_competitor = False
            return res
        elif a == 'even':
            # Use cached planning results (for runtime performance optimization)
            n = self._get_plan(now).get('n_even')
            res = self._publish_n(now, n) if (n is not None and n > 0) else []
            self._last_receive_from_competitor = False
            return res
        elif a == 'hide':
            self._last_receive_from_competitor = False
            return []
        else:
            raise ValueError(f"Unknown action: {action}")

    # ------------------------- action helpers ------------------------------
    def _publish_n(self, now: float, n: int) -> List[Block]:
        """Publish the first n withheld blocks (oldest-first)."""
        if n is None or n <= 0:
            return []
        n = min(int(n), len(self._withheld))
        published: List[Block] = []
        for _ in range(n):
            b = self._withheld.pop(0)
            cb = self._clone_block(b)
            self.public.on_receive(cb, received_time=now)
            published.append(cb)
        self.selected_head_id = self.public.selected_head_id
        self.published += len(published)
        self._recompute_state()
        return published

    def _adopt_public(self) -> None:
        """Give up private chain: align private view to public and clear withheld."""
        self.private = self._clone_miner_state(self.public)
        self._withheld.clear()
        self._recompute_state()

    def _plan_publish_override(self, now: float) -> Optional[int]:
        """Wrapper that retrieves the precomputed minimal n for override.
        Uses a shared incremental plan (for runtime performance optimization)."""
        return self._get_plan(now).get('n_override')

    def _plan_publish_match(self, now: float) -> Optional[int]:
        """Wrapper that retrieves the precomputed minimal n for match.
        Uses a shared incremental plan (for runtime performance optimization)."""
        return self._get_plan(now).get('n_match')

    def _plan_publish_even(self, now: float) -> Optional[int]:
        """Wrapper that retrieves the precomputed minimal n for even.
        Uses a shared incremental plan (for runtime performance optimization)."""
        return self._get_plan(now).get('n_even')

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

    # ------------------------- shared planning helpers ---------------------
    def _get_plan(self, now: float) -> Dict[str, Optional[int]]:
        """Return cached or freshly computed minimal n for {override, match, even}.
        Results are cached per local event id (for runtime performance optimization)."""
        if self._plan_cache_event_id == self._local_event_id and self._plan_cache is not None:
            return self._plan_cache
        plan = self._plan_all(now)
        self._plan_cache_event_id = self._local_event_id
        self._plan_cache = plan
        return plan

    def _plan_all(self, now: float) -> Dict[str, Optional[int]]:
        """Compute minimal n for override/match/even using a single clone and incremental
        application of withheld blocks (for runtime performance optimization)."""
        max_n = len(self._withheld)
        if max_n == 0:
            return {"n_override": None, "n_match": None, "n_even": None}

        m = self._clone_miner_state(self.public)
        n_override: Optional[int] = None
        n_match: Optional[int] = None
        n_even: Optional[int] = None

        def eval_conditions(n_cur: int) -> None:
            nonlocal n_override, n_match, n_even
            our, hon = self._split_heads(m)
            if our is None:
                return
            if hon is None:
                if n_override is None:
                    n_override = n_cur
                return
            hdiff = int(our.height) - int(hon.height)
            wdiff = int(m.cum_block_weight.get(our.id, 0)) - int(m.cum_block_weight.get(hon.id, 0))
            # override condition: dominance by length (>=k) or by weight; and FRP selects our head
            if n_override is None:
                if self.k > 0 and hdiff >= self.k:
                    # If k>0 and we are >=k blocks ahead by height, Rule 1 guarantees our chain wins.
                    n_override = n_cur
                else:
                    selected = m._select_head()
                    # Preserve original semantics for cases where fast path doesn't apply (incl. k==0)
                    if ((self.k > 0 and hdiff >= self.k) or (wdiff > 0)) and (selected.id == our.id):
                        n_override = n_cur
            # match condition: equal weights (best selfish vs best honest)
            if n_match is None and wdiff == 0:
                n_match = n_cur
            # even condition: height >= and weight <
            if n_even is None and (hdiff >= 0 and wdiff < 0):
                n_even = n_cur

        # Evaluate n=0 (no publications)
        eval_conditions(0)
        if n_override is not None and n_match is not None and n_even is not None:
            return {"n_override": n_override, "n_match": n_match, "n_even": n_even}

        # Incrementally publish withheld blocks on the clone and re-evaluate
        for i in range(max_n):
            cb = self._clone_block(self._withheld[i])
            m.on_receive(cb, received_time=now)
            eval_conditions(i + 1)
            if n_override is not None and n_match is not None and n_even is not None:
                break

        return {"n_override": n_override, "n_match": n_match, "n_even": n_even}

    def _split_heads(self, m: Miner) -> Tuple[Optional[Block], Optional[Block]]:
        """Return (best_selfish_head, best_honest_head) among current leaves.

        Optimized: scan leaves by descending height using m.leaves_by_height, and for
        each height select the best candidate by weight (then deterministic tie-break).
        Falls back to scanning all leaves if the index is unavailable.
        """
        lbh = getattr(m, 'leaves_by_height', None)
        if not isinstance(lbh, dict) or not lbh:
            # Fallback to the original implementation using the flat leaf set
            leaves = [m.blocks[bid] for bid in m.leaves]
            our_heads = [b for b in leaves if b.miner_id is not None and int(b.miner_id) == self.miner_id]
            hon_heads = [b for b in leaves if b.miner_id is None or int(b.miner_id) != self.miner_id]

            def _best_fallback(heads: List[Block]) -> Optional[Block]:
                if not heads:
                    return None
                max_h = max(b.height for b in heads)
                cands_h = [b for b in heads if b.height == max_h]
                max_w = max(m.cum_block_weight.get(b.id, 0) for b in cands_h)
                cands_w = [b for b in cands_h if m.cum_block_weight.get(b.id, 0) == max_w]
                if len(cands_w) == 1:
                    return cands_w[0]
                chosen_id = m._deterministic_choice([b.id for b in cands_w])
                return m.blocks[chosen_id]

            return _best_fallback(our_heads), _best_fallback(hon_heads)

        max_h = int(m.max_height)

        def _best_group(is_selfish: bool) -> Optional[Block]:
            # Scan from top height down until we find any candidate of this group
            for h in range(max_h, -1, -1):
                ids = lbh.get(h)
                if not ids:
                    continue
                cand_ids: List[str] = []
                for bid in ids:
                    b = m.blocks[bid]
                    mine = (b.miner_id is not None and int(b.miner_id) == self.miner_id)
                    if is_selfish and mine:
                        cand_ids.append(bid)
                    elif (not is_selfish) and (b.miner_id is None or int(b.miner_id) != self.miner_id):
                        cand_ids.append(bid)
                if not cand_ids:
                    continue
                # Tie-break among this height by weight then deterministic choice
                max_w = max(m.cum_block_weight.get(bid, 0) for bid in cand_ids)
                best_ids = [bid for bid in cand_ids if m.cum_block_weight.get(bid, 0) == max_w]
                if len(best_ids) == 1:
                    return m.blocks[best_ids[0]]
                chosen_id = m._deterministic_choice(best_ids)
                return m.blocks[chosen_id]
            return None

        return _best_group(True), _best_group(False)

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
