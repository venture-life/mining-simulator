from __future__ import annotations

from typing import Any, Dict, Optional, List
import logging

# NOTE: This module deliberately avoids importing SelfishMiner to prevent circular imports.
# All functions accept a `miner` object that provides the attributes and helper methods
# used by the planners and policies (e.g., _clone_miner_state, _split_heads, _clone_block,
# public/private Miner views, k, alpha, _withheld, etc.).
#
# Interface summary:
# - Base policies in this file return one of: 'adopt', 'hide', or ('publish', n)
# - StreamingStepper wraps a base policy and exposes step(miner, now) -> int in {-1,0,1}
#   where -1=adopt, 0=hide, 1=publish-one (and it tracks an internal rollout counter).


def compute_plan_all(miner: Any, now: float) -> Dict[str, Optional[int]]:
    """Compute minimal n for {'override','match','even'} by simulating publication.

    Implementation mirrors the previous SelfishMiner._plan_all but lives out-of-class to
    keep SelfishMiner lean. It uses the PUBLIC view as the baseline and incrementally
    applies withheld blocks to a cloned Miner state.
    """
    max_n = len(getattr(miner, "_withheld", []) or [])
    if max_n == 0:
        return {"n_override": None, "n_match": None, "n_even": None}

    m = miner._clone_miner_state(miner.public)
    n_override: Optional[int] = None
    n_match: Optional[int] = None
    n_even: Optional[int] = None

    def eval_conditions(n_cur: int) -> None:
        nonlocal n_override, n_match, n_even
        our, hon = miner._split_heads(m)
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
            if int(miner.k) > 0 and hdiff >= int(miner.k):
                # If k>0 and we are >=k blocks ahead by height, Rule 1 guarantees our chain wins.
                n_override = n_cur
            else:
                selected = m._select_head()
                # Preserve original semantics for cases where fast path doesn't apply (incl. k==0)
                if (((int(miner.k) > 0) and (hdiff >= int(miner.k))) or (wdiff > 0)) and (selected.id == our.id):
                    n_override = n_cur
        # match condition: equal weights (best selfish vs best honest) AND FRP would select our head
        if n_match is None and wdiff == 0:
            selected2 = m._select_head()
            if selected2.id == our.id:
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
        cb = miner._clone_block(miner._withheld[i])
        m.on_receive(cb, received_time=now)
        eval_conditions(i + 1)
        if n_override is not None and n_match is not None and n_even is not None:
            break

    return {"n_override": n_override, "n_match": n_match, "n_even": n_even}


def plan_publish_override(miner: Any, now: float) -> Optional[int]:
    return compute_plan_all(miner, now).get("n_override")


def plan_publish_match(miner: Any, now: float) -> Optional[int]:
    return compute_plan_all(miner, now).get("n_match")


def plan_publish_even(miner: Any, now: float) -> Optional[int]:
    return compute_plan_all(miner, now).get("n_even")


# ---------------------------- Policies ---------------------------------

def default_policy(miner: Any, now: float) -> Any:
    """Minimal default policy with restricted action space.

    Returns one of:
    - 'adopt' (perish)
    - 'hide'
    - ('publish', n) with n > 0
    """
    withheld = getattr(miner, "_withheld", []) or []
    if not withheld:
        return "hide"
    if getattr(miner, "_last_event", "") == "receive" and bool(getattr(miner, "_last_receive_from_competitor", False)):
        plan = compute_plan_all(miner, now)
        cands = [plan.get("n_match"), plan.get("n_override"), plan.get("n_even")]
        cands = [int(n) for n in cands if (n is not None and int(n) > 0)]
        if cands:
            return ("publish", min(cands))
        return "hide"
    return "hide"


def heuristic_policy(miner: Any, now: float) -> Any:
    """A richer heuristic that adapts by (alpha, k) and checks feasibility.

    Returns only 'adopt'/'hide'/('publish', n).
    """
    # Pull dynamic state
    Bs = int(getattr(miner, "Bs", 0))
    Bh = int(getattr(miner, "Bh", 0))
    lead = Bs - Bh
    have_secret = len(getattr(miner, "_withheld", []) or []) > 0
    a = float(getattr(miner, "alpha", 0.0) or 0.0)
    k = int(getattr(miner, "k", 0))

    # No secrets to reveal
    if not have_secret:
        if getattr(miner, "last", "h") == "h" and Bs <= Bh:
            return "adopt"
        return "hide"

    # React only when competitor advanced
    if getattr(miner, "last", "h") == "h":
        plan = compute_plan_all(miner, now)
        n_over = plan.get("n_override")
        n_match = plan.get("n_match")
        n_even = plan.get("n_even")

        over_ok = (n_over is not None and int(n_over) > 0)
        match_ok = (n_match is not None and int(n_match) > 0)
        even_ok = (n_even is not None and int(n_even) > 0)

        # Alpha- and k-adaptive aggressiveness thresholds
        if a >= (1.0 / 3.0):
            allow_over = 2
        elif a >= (1.0 / 4.0):
            allow_over = 1
        else:
            allow_over = 0
        if k <= 2:
            allow_over += 1

        # Choose n to publish without exposing match/override/even labels
        chosen_n: Optional[int] = None
        if over_ok and int(n_over) <= allow_over and (not match_ok or int(n_over) <= int(n_match)) and (not even_ok or int(n_over) <= int(n_even)) and not bool(getattr(miner, "luck", False)):
            chosen_n = int(n_over)
        elif match_ok and (not over_ok or int(n_match) < int(n_over) or bool(getattr(miner, "luck", False)) or (over_ok and int(n_over) > allow_over)):
            chosen_n = int(n_match)
        elif lead >= 2 and not over_ok and k <= 2:
            chosen_n = 1
        elif over_ok and a >= (1.0 / 3.0):
            chosen_n = int(n_over)
        elif even_ok and (a >= (1.0 / 3.0) or (a >= (1.0 / 4.0) and k <= 2)):
            chosen_n = int(n_even)

        if chosen_n is not None and chosen_n > 0:
            return ("publish", int(chosen_n))
        if lead <= 0:
            return "adopt"
        return "hide"

    return "hide"


def simple_policy_factory(reveal_on_lead: int = 2, adopt_on_competitor_when_not_ahead: bool = True):
    """Return a lightweight policy callback that avoids internal planning.

    Returns a callable (miner, now) -> action among {'adopt','hide',('publish', n)}.
    """
    def _policy(m: Any, now: float) -> Any:
        try:
            # Lead computed from miner API/state
            try:
                lead = int(getattr(m, "lead", lambda: 0)())
            except Exception as e:
                logging.exception("simple_policy_factory: lead() call failed; falling back to Bs-Bh")
                lead = int(getattr(m, "Bs", 0)) - int(getattr(m, "Bh", 0))
            have_secret = len(getattr(m, "_withheld", []) or []) > 0
            if not have_secret:
                if adopt_on_competitor_when_not_ahead and getattr(m, "last", "h") == 'h' and lead <= 0:
                    return 'adopt'
                return 'hide'
            if getattr(m, "last", "h") == 'h':
                if lead >= int(reveal_on_lead):
                    return ('publish', 1)
                if adopt_on_competitor_when_not_ahead and lead <= 0:
                    return 'adopt'
                return 'hide'
            return 'hide'
        except Exception as e:
            logging.exception("simple_policy_factory: policy evaluation failed; returning 'hide'")
            return 'hide'
    return _policy


# ----------------------- Streaming stepper (decision+action) -----------------------
class StreamingStepper:
    """Stateful streaming decider that returns -1/0/1 per step with rollout state.

    Returns:
    -1 => adopt
     0 => hide/no-op
     1 => publish one block this step

    The stepper does not perform actions; it only decides and tracks remaining
    publications for multi-block plans via 'active_rollout'.
    """

    def __init__(self, base_policy: Any) -> None:
        self.base_policy = base_policy
        self.active_rollout: Optional[int] = None

    def _coerce_plan(self, plan: Any) -> Dict[str, Optional[int]]:
        # Normalize policy output to a dict {'action': str, 'n': Optional[int]}
        if isinstance(plan, str):
            a = plan.strip().lower()
            if a == 'publish':
                return {'action': 'publish', 'n': 1}
            if a in ('adopt', 'hide'):
                return {'action': a, 'n': None}
            return {'action': 'hide', 'n': None}
        if isinstance(plan, (tuple, list)) and len(plan) >= 1:
            a = str(plan[0]).strip().lower()
            n = None
            if len(plan) >= 2:
                try:
                    n = int(plan[1])
                except Exception:
                    n = None
            if a == 'publish':
                return {'action': 'publish', 'n': (n if n is not None and n > 0 else 1)}
            if a in ('adopt', 'hide'):
                return {'action': a, 'n': None}
            return {'action': 'hide', 'n': None}
        if isinstance(plan, dict) and 'action' in plan:
            a = str(plan.get('action', '')).strip().lower()
            try:
                n = int(plan.get('n')) if plan.get('n') is not None else None
            except Exception:
                n = None
            if a == 'publish':
                return {'action': 'publish', 'n': (n if n is not None and n > 0 else 1)}
            if a in ('adopt', 'hide'):
                return {'action': a, 'n': None}
            return {'action': 'hide', 'n': None}
        # Fallback safe no-op
        return {'action': 'hide', 'n': None}

    def step(self, miner: Any, now: float) -> int:
        """Decide a single-step action and return -1/0/1 without performing it.

        - If an active rollout exists and there are withheld blocks: return 1 and
          decrement the internal counter. If no withheld remain, end rollout and
          return 0.
        - Otherwise, query base policy, coerce to {'adopt','hide',('publish',n)};
          return -1 for adopt, 0 for hide, and 1 for publish while starting a
          rollout with n.
        """
        # Continue an active rollout if any
        if self.active_rollout is not None and self.active_rollout > 0:
            try:
                have_secret = len(getattr(miner, "_withheld", []) or []) > 0
            except Exception:
                have_secret = False
            if not have_secret:
                self.active_rollout = 0
                return 0
            # consume one publication from the rollout
            self.active_rollout -= 1
            if self.active_rollout < 0:
                self.active_rollout = 0
            return 1

        # No active rollout: ask the base policy
        try:
            plan_raw = self.base_policy(miner, now)
        except Exception:
            plan_raw = 'hide'
        plan = self._coerce_plan(plan_raw)
        a = plan.get('action')
        n = plan.get('n')
        if a == 'adopt':
            return -1
        if a == 'hide':
            return 0
        if a == 'publish':
            n_eff = int(n) if (n is not None and int(n) > 0) else 1
            self.active_rollout = n_eff - 1  # one will be executed now
            try:
                have_secret = len(getattr(miner, "_withheld", []) or []) > 0
            except Exception:
                have_secret = False
            if not have_secret:
                # cannot publish; cancel rollout
                self.active_rollout = 0
                return 0
            return 1
        return 0


def make_streaming_stepper(base_policy: Any) -> StreamingStepper:
    return StreamingStepper(base_policy)
