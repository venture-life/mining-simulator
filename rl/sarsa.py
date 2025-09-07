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
#   0 -> Perish (adopt)
#   1..(k_cap+1) -> Publish(n) where n = action_id - 1, thus 1->n=1, etc.; n=0 is represented by action_id=1? No.
# We reserve action_id=1.. for n>=1, and represent Publish(0)=hide by a dedicated id = HIDE_ID.
PERISH_ID = 0
HIDE_ID = 1  # Publish(0)


def publish_id(n: int) -> int:
    """Map Publish(n) to an integer action id.
    n==0 -> HIDE_ID. n>=1 -> HIDE_ID + n
    """
    if n < 0:
        n = 0
    return HIDE_ID + n


def publish_n_from_id(action_id: int) -> int:
    """Inverse of publish_id for Publish(n) ids (including HIDE_ID as n=0)."""
    if action_id < HIDE_ID:
        raise ValueError("Action id does not encode Publish(n)")
    return action_id - HIDE_ID


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
                 lead_clip: int = 3,
                 diffw_bins: Sequence[int] = (-2, -1, 0, 1, 2),
                 mu_edges: Sequence[float] = (0.06, 0.10)) -> None:
        self.lead_clip = int(lead_clip)
        self.diffw_bins = tuple(diffw_bins)
        self.mu_edges = tuple(mu_edges)

    def _bin_lead(self, lead: int) -> int:
        L = max(-self.lead_clip, min(self.lead_clip, int(lead)))
        return L

    def _bin_diffw(self, dw: int) -> int:
        if dw <= -2:
            return -2
        if dw == -1:
            return -1
        if dw == 0:
            return 0
        if dw == 1:
            return 1
        return 2

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
            return "mulo"
        if mu < self.mu_edges[1]:
            return "mumi"
        return "muhi"

    def _bucket_mode(self, flag: int) -> str:
        return "m1" if int(flag) else "m0"

    def discretize(self, miner: Any, ctx: ScenarioCtx) -> Tuple[Hashable, int]:
        """Return (state_key, k_cap) for tabular Q lookup.

        state_key contains:
        - lead_bin ∈ [-clip..clip]
        - diffw_bin ∈ {-2,-1,0,1,2}
        - luck_bool ∈ {0,1}
        - last ∈ {h,s}
        - withheld_count_clip ∈ [0..min(3,k)]
        - alpha_bucket ∈ {a0,a1,a2,a3}
        - k_bucket ∈ {k2,k3,k4p}
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
        last = 's' if getattr(miner, "last", 'h') == 's' else 'h'
        withheld_count = len(getattr(miner, "_withheld", []) or [])

        lead_b = self._bin_lead(lead)
        diffw_b = self._bin_diffw(diffw)
        wc_cap = min(3, int(ctx.k))
        wc_b = min(int(withheld_count), wc_cap)
        a_b = self._bucket_alpha(ctx.alpha)
        k_b = self._bucket_k(ctx.k)
        mu_b = self._bucket_mu(ctx.mu)
        mode_b = self._bucket_mode(ctx.mode_flag)

        state_key = (lead_b, diffw_b, luck, last, wc_b, a_b, k_b, mu_b, mode_b)
        return state_key, int(ctx.k)


def allowed_actions(miner: Any, k: int) -> List[int]:
    """Return a list of action ids feasible in the current state.

    Always include PERISH_ID and HIDE_ID. Include Publish(n) for n=1..min(k, withheld_count).
    Optionally, callers may further prune using 'last' or 'lead' if desired.
    """
    acts = [PERISH_ID, HIDE_ID]
    wcnt = len(getattr(miner, "_withheld", []) or [])
    ncap = min(int(k), int(wcnt))
    for n in range(1, ncap + 1):
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
            return HIDE_ID
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
        mask = allowed_actions(miner, k_cap)
        # Baseline preference: if last=='h' → Perish; if last=='s' → Publish(1) if feasible else Hide
        last = 's' if getattr(miner, "last", 'h') == 's' else 'h'
        wcnt = len(getattr(miner, "_withheld", []) or [])
        preferred: Optional[int] = None
        if last == 'h':
            preferred = PERISH_ID
        else:
            preferred = publish_id(1) if wcnt >= 1 else HIDE_ID
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
            mask = allowed_actions(miner, k_cap)
            # Baseline preference: if last=='h' → Perish; if last=='s' → Publish(1) if feasible else Hide
            last = 's' if getattr(miner, "last", 'h') == 's' else 'h'
            wcnt = len(getattr(miner, "_withheld", []) or [])
            preferred: Optional[int] = None
            if last == 'h':
                preferred = PERISH_ID
            else:
                preferred = publish_id(1) if wcnt >= 1 else HIDE_ID
            a_id = learner.choose_action(s_key, mask, preferred=preferred)
            if a_id == PERISH_ID:
                return 'adopt'
            # Publish(n) including hide via n=0
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
        Bh=10; Bs=11; diff_w=0; luck=False; last='h'; _withheld=[object()]
        def lead(self):
            return self.Bs - self.Bh
    fake = _Fake()
    act = policy_cb(fake, now=0.0)
    print("Policy callback smoke test:", act)
    print("RL SARSA(λ) skeleton ready. Integrate with simulator by passing policy=policy_cb to SelfishMiner.")
