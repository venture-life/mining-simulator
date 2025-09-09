"""
SARSA(λ) skeleton for SelfishMiner policy learning.

This module defines:
- Discretizer: maps SelfishMiner + scenario context to a compact tabular state.
- Masked action space with two primitives: Perish (adopt) and Publish(n).
- SARSA(λ) learner with eligibility traces.
- A policy callback factory that returns actions in the format expected by SelfishMiner
  ("adopt" or ("reveal", n)) so we bypass the internal planner.

Integration notes (no code changes performed here):
- To actually learn inside the existing simulator, `simulate_mining_eventqV2` must
  instantiate SelfishMiner with `policy=policy_callback` (or expose a handle so you
  can assign `miner.policy = policy_callback` after creation).
- Until then, this module provides a runnable skeleton (unit-testable on a toy MDP)
  and the exact callback signature required by SelfishMiner.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Hashable, List, Optional, Sequence, Tuple
import math
import random
import logging

# Public action encoding for tabular SARSA
#   PERISH_ID  = 0        -> Perish ('adopt')
#   PUBLISH_ID = 1        -> Publish(0) ('hide')
#   PUBLISH_ID + n (n>=1) -> Publish(n)
PERISH_ID = 0
PUBLISH_ID = 1  # Publish(0)


def publish_id(n: int) -> int:
    """Map Publish(n) to an integer action id.
    n==0 -> PUBLISH_ID. n>=1 -> PUBLISH_ID + n
    """
    if n < 0:
        n = 0
    return PUBLISH_ID + n


def publish_n_from_id(action_id: int) -> int:
    """Inverse of publish_id for Publish(n) ids (including PUBLISH_ID as n=0)."""
    if action_id < PUBLISH_ID:
        raise ValueError("Action id does not encode Publish(n)")
    return action_id - PUBLISH_ID


@dataclass
class ScenarioCtx:
    """Static/global scenario context used for state and action masks.

    Attributes
    ----------
    alpha: attacker's hashrate share in (0,1)
    k: dominance threshold for Rule 1
    mu: concurrency parameter (Lambda * D)
    mode_flag: 0=fixed_total, 1=additive_attacker
    """
    alpha: float
    k: int
    mu: float
    mode_flag: int  # 0 or 1


class Discretizer:
    """Discretize SelfishMiner state and scenario context to a tabular key."""

    def __init__(self,
                 lead_clip: int = 5,
                 diffw_bins: Sequence[int] = (-5,-4,-3,-2, -1, 0, 1, 2,3,4,5),
                 mu_edges: Sequence[float] = (0.06, 0.10),
                 pub_clip: int = 5) -> None:
        self.lead_clip = int(lead_clip)
        self.diffw_bins = tuple(diffw_bins)
        self.mu_edges = tuple(mu_edges)
        self.pub_clip = int(pub_clip)

    def _bin_lead(self, lead: int) -> int:
        # L = max(-self.lead_clip, min(self.lead_clip, int(lead)))
        # return L
        return lead

    def _bin_diffw(self, dw: int) -> int:
        if dw <= -5:
            return -5
        if dw >= 5:
            return 5
        return dw

    def _bucket_alpha(self, a: float) -> str:
        a = float(a)
        if a < 0.25:
            return "a0"
        if a < (1.0/3.0):
            return "a1"
        if a <= 0.42:
            return "a2"
        if a <= 0.5:
            return "a3"
        return "a4"  # strictly > 0.5

    def _bucket_k(self, k: int) -> str:
        k = int(k)
        if k <= 2:
            return "k2"
        if k == 3:
            return "k3"
        if k == 4:
            return "k4"
        return "k5p"

    def _bucket_mu(self, mu: float) -> str:
        mu = float(mu)
        if mu < self.mu_edges[0]:
            return "mulo"         # 120s block frequency with 5s window (mu=0.0416666)
        if mu < self.mu_edges[1]:
            return "mumi"         # 60s block frequency with 5s window (mu=0.0833333)
        return "muhi"             # 30s block frequency with 5s window (mu=0.1666666)

    def _bucket_mode(self, flag: int) -> str:
        return "m1" if int(flag) else "m0"

    def discretize(self, miner: Any, ctx: ScenarioCtx) -> Tuple[Hashable, int]:
        """Return (state_key, k_cap) for tabular Q lookup.

        state_key contains:
        - lead_bin ∈ [-clip..clip]
        - diffw_bin bounded by diffw_bins (default [-5..+5])
        - luck_bool ∈ {0,1}
        - luck2_bool ∈ {0,1} (weight tie and deterministic FRP selects attacker)
        - last ∈ {h,s}
        - withheld_count_clip ∈ [0..min(5,k)]
        - published_count_clip ∈ [0..pub_clip]
        - alpha_bucket ∈ {a0,a1,a2,a3,a4}
        - k_bucket ∈ {k2,k3,k4,k5p}
        - mu_bucket ∈ {mulo,mumi,muhi}
        - mode_flag_bucket ∈ {m0,m1}
        """
        # Pull dynamic fields from SelfishMiner
        try:
            lead = int(miner.Bs) - int(miner.Bh) if hasattr(miner, "Bs") and hasattr(miner, "Bh") else int(miner.lead())
        except Exception:
            lead = int(getattr(miner, "lead", lambda: 0)())
        diffw = int(getattr(miner, "diff_w", 0))
        luck = 1 if bool(getattr(miner, "luck", False)) else 0
        luck2 = 1 if bool(getattr(miner, "luck2", False)) else 0
        last = 's' if getattr(miner, "last", 'h') == 's' else 'h'
        withheld_count = len(getattr(miner, "_withheld", []) or [])
        published = int(getattr(miner, "published", 0))

        lead_b = self._bin_lead(lead)
        diffw_b = self._bin_diffw(diffw)
        wc_cap = min(5, int(ctx.k))
        # wc_b = min(int(withheld_count), wc_cap)
        wc_b = int(withheld_count)
        # pub_b = min(max(0, int(published)), int(self.pub_clip))
        pub_b = int(published)
        a_b = self._bucket_alpha(ctx.alpha)
        k_b = self._bucket_k(ctx.k)
        mu_b = self._bucket_mu(ctx.mu)
        mode_b = self._bucket_mode(ctx.mode_flag)

        state_key = (lead_b, diffw_b, luck, luck2, last, wc_b, pub_b, a_b, k_b, mu_b, mode_b)
        return state_key, int(ctx.k)


def allowed_actions(miner: Any, k: int, now: Optional[float] = None) -> List[int]:
    """Return a list of action ids feasible in the current state at time 'now'.

    Always include PERISH_ID ('adopt') and PUBLISH_ID ('hide').
    Publish(n) actions are included for n in [1..min(k, withheld_count)],
    BUT we "clip on the lower side": only include Publish(n) if n >= n_eff,
    where n_eff is the minimal positive n that achieves either 'match' or 'override'
    according to the miner's planner at time 'now'. If neither is achievable (no
    positive n), no Publish(n) actions are offered.
    """
    acts = [PERISH_ID, PUBLISH_ID]
    wcnt = len(getattr(miner, "_withheld", []) or [])
    ncap = min(int(k), int(wcnt))
    ncap = int(wcnt) # to be checked later if the above makes more sense. currenlty, publishing more than k for deeper splits / branches makes sense
    # Determine threshold n_eff via planner if possible
    n_eff: Optional[int] = None
    try:
        if now is not None:
            # Prefer wrappers if available
            n_match = None
            n_over = None
            if hasattr(miner, "_plan_publish_match"):
                try:
                    n_match = miner._plan_publish_match(float(now))  # type: ignore[attr-defined]
                except Exception:
                    n_match = None
            if hasattr(miner, "_plan_publish_override"):
                try:
                    n_over = miner._plan_publish_override(float(now))  # type: ignore[attr-defined]
                except Exception:
                    n_over = None
            cands = [int(n) for n in (n_match, n_over) if (n is not None and int(n) > 0)]
            if cands:
                n_eff = min(cands)
    except Exception:
        n_eff = None
    # Add publish actions subject to threshold
    if ncap > 0:
        for n in range(1, ncap + 1):
            if n_eff is None or n >= int(n_eff):
                acts.append(publish_id(n))
    return acts


class SarsaLambda:
    """Tabular SARSA(λ) with replacing traces and masked ε-greedy policy."""

    def __init__(self,
                 gamma: float = 1.0,
                 lam: float = 0.9,
                 alpha: float = 0.05,
                 epsilon: float = 0.1) -> None:
        self.gamma = float(gamma)
        self.lam = float(lam)
        self.alpha = float(alpha)
        self.epsilon = float(epsilon)
        self.Q: Dict[Tuple[Hashable, int], float] = {}
        self.E: Dict[Tuple[Hashable, int], float] = {}

    def _q(self, s: Hashable, a: int) -> float:
        return self.Q.get((s, a), 0.0)

    def _set_q(self, s: Hashable, a: int, v: float) -> None:
        self.Q[(s, a)] = float(v)

    def _e(self, s: Hashable, a: int) -> float:
        return self.E.get((s, a), 0.0)

    def _set_e(self, s: Hashable, a: int, v: float) -> None:
        self.E[(s, a)] = float(v)

    def choose_action(self, s: Hashable, mask: Sequence[int], preferred: Optional[int] = None) -> int:
        """ε-greedy choice among masked actions with optional tie-break preference.

        If preferred is provided and is among the greedy-best actions, it will be chosen.
        """
        if not mask:
            return PUBLISH_ID
        if random.random() < self.epsilon:
            return random.choice(list(mask))
        # Greedy among mask, tie-break uniformly
        vals = [(a, self._q(s, a)) for a in mask]
        max_v = max(v for _, v in vals)
        best = [a for a, v in vals if v == max_v]
        if preferred is not None and preferred in best:
            return preferred
        return random.choice(best)

    def reset_traces(self) -> None:
        self.E.clear()

    def update(self,
               s: Hashable,
               a: int,
               r: float,
               s_next: Optional[Hashable],
               a_next: Optional[int],
               gamma_override: Optional[float] = None) -> None:
        """One SARSA(λ) update step with replacing traces.

        If gamma_override is provided, it is used for this step (e.g., time-aware γ_eff).
        """
        g = float(self.gamma if gamma_override is None else gamma_override)
        q_sa = self._q(s, a)
        if s_next is None or a_next is None:
            target = r
        else:
            target = r + g * self._q(s_next, a_next)
        delta = target - q_sa
        # Replacing traces
        # decay all traces
        for key in list(self.E.keys()):
            self.E[key] *= g * self.lam
            if self.E[key] < 1e-12:
                self.E.pop(key, None)
        # set current trace to 1
        self._set_e(s, a, 1.0)
        # update all Q with their eligibility
        for (ss, aa), e_val in list(self.E.items()):
            self._set_q(ss, aa, self._q(ss, aa) + self.alpha * delta * e_val)


# Stateful SARSA(λ) policy with per-decision bootstrapping
class BootstrappedPolicy:
    """Callable policy object implementing true SARSA(λ) online updates.

    Usage:
    - Instantiate with (learner, discretizer, context).
    - Call start_episode() before running the simulator episode.
    - Pass the instance itself as `selfish_policy` to simulate_mining_eventqV2.
    - After the episode, call end_episode(final_return) to apply the terminal update and reset traces.

    The object maintains (prev_state, prev_action) across policy calls and updates the learner
    at every decision with r=0 and bootstrapped Q(s',a') until the terminal step, where it uses
    the provided final_return as the terminal reward.
    """
    def __init__(self, learner: SarsaLambda, disc: Discretizer, ctx: ScenarioCtx, *, gamma_per_second: float = 1.0) -> None:
        self.learner = learner
        self.disc = disc
        self.ctx = ctx
        self.gamma_per_second = float(gamma_per_second)
        self._prev_s: Optional[Hashable] = None
        self._prev_a: Optional[int] = None
        self._prev_t: Optional[float] = None
        self._started: bool = False

    def start_episode(self) -> None:
        self.learner.reset_traces()
        self._prev_s = None
        self._prev_a = None
        self._prev_t = None
        self._started = True

    def __call__(self, miner: Any, now: float) -> Any:
        s_key, k_cap = self.disc.discretize(miner, self.ctx)
        mask = allowed_actions(miner, k_cap, now=now)
        # Baseline preference: if last=='h' → Perish; if last=='s' → minimal allowed Publish(n) else Hide
        last = 's' if getattr(miner, "last", 'h') == 's' else 'h'
        wcnt = len(getattr(miner, "_withheld", []) or [])
        preferred: Optional[int] = None
        if last == 'h':
            preferred = PERISH_ID
        else:
            # Find minimal allowed Publish(n) from mask
            publish_ids = [aid for aid in mask if aid >= PUBLISH_ID + 1]
            preferred = (min(publish_ids) if publish_ids else PUBLISH_ID)
        a_id = self.learner.choose_action(s_key, mask, preferred=preferred)
        # Perform one-step SARSA(λ) update for the previous decision (r=0 during episode)
        if self._started and self._prev_s is not None and self._prev_a is not None:
            gamma_eff: Optional[float] = None
            if self.gamma_per_second != 1.0 and self._prev_t is not None:
                try:
                    dt = max(0.0, float(now) - float(self._prev_t))
                except Exception:
                    dt = 0.0
                gamma_eff = float(self.gamma_per_second) ** float(dt)
            self.learner.update(self._prev_s, self._prev_a, r=0.0, s_next=s_key, a_next=a_id, gamma_override=gamma_eff)
        # Store current for next step
        self._prev_s = s_key
        self._prev_a = a_id
        self._prev_t = float(now)
        # Return action in simulator's format
        if a_id == PERISH_ID:
            return 'adopt'
        if a_id == PUBLISH_ID:
            return 'hide'
        n = publish_n_from_id(a_id)
        return ('reveal', int(n))

    def end_episode(self, final_return: float) -> None:
        """Apply terminal update with final_return and reset traces for next episode."""
        if self._started and self._prev_s is not None and self._prev_a is not None:
            self.learner.update(self._prev_s, self._prev_a, r=float(final_return), s_next=None, a_next=None, gamma_override=None)
        # Prepare for the next episode
        self.learner.reset_traces()
        self._prev_s = None
        self._prev_a = None
        self._prev_t = None
        self._started = False


def make_bootstrapped_policy(learner: SarsaLambda, disc: Discretizer, ctx: ScenarioCtx, *, gamma_per_second: float = 1.0) -> BootstrappedPolicy:
    """Factory for a stateful SARSA(λ) bootstrapping policy object."""
    return BootstrappedPolicy(learner, disc, ctx, gamma_per_second=gamma_per_second)



def policy_callback_factory(learner: SarsaLambda,
                            disc: Discretizer,
                            ctx: ScenarioCtx) -> Callable[[Any, float], Any]:
    """Return a SelfishMiner-compatible policy callback.

    The callback inspects the current SelfishMiner state, discretizes it, selects a masked
    action with ε-greedy SARSA policy, and returns either 'adopt' (Perish) or ('reveal', n)
    for Publish(n). It does not block, and it does not access any global planner.
    """
    def _policy(miner: Any, now: float) -> Any:
        try:
            s_key, k_cap = disc.discretize(miner, ctx)
            mask = allowed_actions(miner, k_cap, now=now)
            # Baseline preference: if last=='h' → Perish; if last=='s' → minimal allowed Publish(n) else Hide
            last = 's' if getattr(miner, "last", 'h') == 's' else 'h'
            wcnt = len(getattr(miner, "_withheld", []) or [])
            preferred: Optional[int] = None
            if last == 'h':
                preferred = PERISH_ID
            else:
                publish_ids = [aid for aid in mask if aid >= PUBLISH_ID + 1]
                preferred = (min(publish_ids) if publish_ids else PUBLISH_ID)
            a_id = learner.choose_action(s_key, mask, preferred=preferred)
            if a_id == PERISH_ID:
                return 'adopt'
            if a_id == PUBLISH_ID:
                return 'hide'
            # Publish(n) for n>=1
            n = publish_n_from_id(a_id)
            return ('reveal', int(n))
        except Exception:
            # Fallback to safe no-op
            return 'hide'
    return _policy


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    # Smoke test: initialize learner and policy without touching the simulator
    ctx = ScenarioCtx(alpha=0.3, k=3, mu=0.0833, mode_flag=0)
    disc = Discretizer()
    agent = SarsaLambda(gamma=1.0, lam=0.9, alpha=0.05, epsilon=0.1)
    policy_cb = policy_callback_factory(agent, disc, ctx)
    # Fake miner with minimal attributes for a quick policy call
    class _Fake:
        Bh=10; Bs=11; diff_w=0; luck=False; luck2=False; last='h'; _withheld=[object()]
        def lead(self):
            return self.Bs - self.Bh
    fake = _Fake()
    act = policy_cb(fake, now=0.0)
    print("Policy callback smoke test:", act)
    print("RL SARSA(λ) skeleton ready. Integrate with simulator by passing policy=policy_cb to SelfishMiner.")
