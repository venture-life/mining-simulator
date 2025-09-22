"""
SARSA(λ) skeleton for SelfishMiner policy learning.

This module defines:
- Discretizer: maps SelfishMiner + scenario context to a compact tabular state.
- Masked action space with three primitives encoded as integers:
  -1 => adopt (perish), 0 => hide (no-op), 1 => publish exactly one block.
- SARSA(λ) learner with eligibility traces.
- Policy factories that return the integer actions expected by SelfishMiner's
  unified decider interface (miner, now) -> int in {-1, 0, 1}.

Integration notes:
- To learn inside the simulator, pass the returned policy as `selfish_policy`
  to `simulate_mining_eventqV2` so the SelfishMiner uses it directly.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Hashable, List, Optional, Sequence, Tuple
import math
import random
import logging

# New action space: integers {-1, 0, 1}
# -1 => adopt (perish), 0 => hide, 1 => publish one withheld block


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
                 withheld_cap: int = 8) -> None:
        self.lead_clip = int(lead_clip)
        self.diffw_bins = tuple(diffw_bins)
        self.mu_edges = tuple(mu_edges)
        self.wc_cap = int(withheld_cap)

    def _bin_lead(self, lead: int) -> int:
        L = max(-self.lead_clip, min(self.lead_clip, int(lead)))
        return L

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
        - diffw_bin (private-public) bounded by diffw_bins (default [-5..+5])
        - luck_bool ∈ {0,1} (when withholding and diff_w==0, FRP on cf selects attacker)
        - last ∈ {s0,s1,h0,h1}
        - withheld_count_clip ∈ [0..withheld_cap] (cap independent of k)
        - alpha_bucket ∈ {a0,a1,a2,a3,a4}
        - k_bucket ∈ {k2,k3,k4,k5p}
        - mu_bucket ∈ {mulo,mumi,muhi}
        - mode_flag_bucket ∈ {m0,m1}
        """
        # Pull dynamic fields from SelfishMiner
        lead = int(getattr(miner, "lead",  0))
        diffw = int(getattr(miner, "diff_w", 0))
        luck = 1 if bool(getattr(miner, "luck", False)) else 0
        last = str(getattr(miner, "last", 'h0'))
        withheld_count = len(getattr(miner, "_withheld", []) or [])

        lead_b = self._bin_lead(lead)
        diffw_b = self._bin_diffw(diffw)
        wc_b = min(int(withheld_count), int(self.wc_cap))
        a_b = self._bucket_alpha(ctx.alpha)
        k_b = self._bucket_k(ctx.k)
        mu_b = self._bucket_mu(ctx.mu)
        mode_b = self._bucket_mode(ctx.mode_flag)

        state_key = (lead_b, diffw_b, luck, last, wc_b, a_b, k_b, mu_b, mode_b)
        return state_key, int(ctx.k)


def allowed_actions(miner: Any, k: int, now: Optional[float] = None) -> List[int]:
    """Return masked actions in {-1, 0, 1} based on current miner state.

    Rules
    -----
    - If no withheld blocks: only -1 (adopt). There is nothing to hide or publish.
    - If withholding (withheld_count > 0): always allow 0 (hide) and 1 (publish-one).
      Allow -1 (adopt) unless the last miner was selfish AND the last event was 'mine'
      (never adopt immediately after our own mine).

    Parameter k is unused but kept for signature stability.
    """

    last_sym = str(getattr(miner, "last", 'h0'))
    withheld_count = len(getattr(miner, "_withheld", []) or [])
    # lead = int(getattr(miner, "lead", lambda: 0)())
    acts: List[int] = [0]
    if withheld_count > 0: # and not (withheld_count == 1 and last_sym == 's0'):
        acts.append(1)
    if not (last_sym == 's0' or last_sym == 's1'):
        acts.append(-1)

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
            return 0  # default to hide if no actions given
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
        # Tracing/debug
        self.debug: bool = False
        self.traces: List[Dict[str, Any]] = []
        self._trace_cap: int = 100000
        # Optional last-based Q bootstrap (honest-baseline): if all Q(s,·)==0, seed
        self.bootstrap_last: bool = False
        self.last_prior: float = 0.05

    def enable_trace(self, debug: bool = True, *, max_len: Optional[int] = None) -> None:
        """Enable per-decision tracing for debugging. Set max_len to cap buffer size."""
        self.debug = bool(debug)
        if max_len is not None:
            try:
                self._trace_cap = max(1, int(max_len))
            except Exception as e:
                logging.exception("BootstrappedPolicy.enable_trace: invalid max_len=%r", max_len)

    def enable_last_bootstrap(self, prior: float = 0.05) -> None:
        """Enable lazy Q seeding based solely on 'last'.

        If all Q(s,·) for the current masked actions are exactly zero, we set
        Q(s, a_pref) += prior where a_pref = (-1 if last=='h' else 1 if allowed else 0).
        This biases the initial policy to honest-miner behavior without precomputing
        the entire state-space. Learning updates proceed normally thereafter.
        """
        self.bootstrap_last = True
        try:
            self.last_prior = float(prior)
        except Exception as e:
            logging.exception("BootstrappedPolicy.enable_last_bootstrap: invalid prior=%r; defaulting to 0.05", prior)
            self.last_prior = 0.05

    def _append_trace(self, entry: Dict[str, Any]) -> None:
        try:
            self.traces.append(entry)
            if len(self.traces) > self._trace_cap:
                # drop from the front (keep recent)
                self.traces = self.traces[-self._trace_cap:]
        except Exception as e:
            logging.exception("BootstrappedPolicy._append_trace failed; entry truncated")

    def start_episode(self) -> None:
        self.learner.reset_traces()
        self._prev_s = None
        self._prev_a = None
        self._prev_t = None
        self._started = True

    def __call__(self, miner: Any, now: float) -> Any:
        s_key, k_cap = self.disc.discretize(miner, self.ctx)
        mask = allowed_actions(miner, k_cap, now=now)
        # Baseline preference: if last is honest (h0/h1) → adopt (-1); if last=='s' → publish (1) if allowed else hide (0)
        last_raw = str(getattr(miner, "last", "h0"))
        last_is_honest = last_raw.startswith('h')
        preferred: Optional[int] = None
        if last_is_honest:
            preferred = -1
        else:
            preferred = 1 if (1 in mask) else 0

        # Snapshot Q-values for masked actions prior to selection (for debugging)
        try:
            q_vals = {int(a): float(self.learner._q(s_key, int(a))) for a in mask}
            if q_vals:
                max_q = max(q_vals.values())
                best_actions = [a for a, v in q_vals.items() if v == max_q]
            else:
                best_actions = []
        except Exception as e:
            logging.exception("BootstrappedPolicy.__call__: snapshot Q-values failed; proceeding with empty q_vals"); q_vals = {}; best_actions = []

        # Optional: last-based Q bootstrap when all Q(s,·) are zero/equal
        if self.bootstrap_last:
            try:
                all_zero = (len(q_vals) > 0) and all(abs(v) == 0.0 for v in q_vals.values())
            except Exception as e:
                logging.exception("BootstrappedPolicy.__call__: all_zero check failed; skipping bootstrap")
                all_zero = False
            if all_zero:
                seed_action = preferred if preferred in mask else (0 if 0 in mask else (mask[0] if mask else 0))
                try:
                    self.learner._set_q(s_key, int(seed_action), self.learner._q(s_key, int(seed_action)) + float(self.last_prior))
                    # refresh local snapshot
                    q_vals[int(seed_action)] = self.learner._q(s_key, int(seed_action))
                    max_q = max(q_vals.values())
                    best_actions = [a for a, v in q_vals.items() if v == max_q]
                except Exception as e:
                    logging.exception("BootstrappedPolicy.__call__: failed to bootstrap Q at preferred action=%s", seed_action)

        a_id = self.learner.choose_action(s_key, mask, preferred=preferred)
        is_greedy = bool(a_id in best_actions) if best_actions else True

        # Optional trace entry
        if self.debug:
            try:
                # Pull a few raw features for readability
                lead = int(getattr(miner, "lead", 0)())
                diff_w = int(getattr(miner, "diff_w", 0))
                luck = int(bool(getattr(miner, "luck", False)))
                last_sym = str(getattr(miner, "last", "h0"))
                wcnt = int(len(getattr(miner, "_withheld", []) or []))
                entry = {
                    "t": float(now),
                    "state_key": s_key,
                    "lead": lead,
                    "diffw": diff_w,
                    "luck": luck,
                    "last": last_sym,
                    "withheld": wcnt,
                    "mask": list(mask),
                    "preferred": int(preferred) if preferred is not None else None,
                    "q": q_vals,
                    "action": int(a_id),
                    "greedy": bool(is_greedy),
                }
                self._append_trace(entry)
                logging.debug("SARSA decision t=%.6f a=%s greedy=%s mask=%s q=%s state=%s", now, a_id, is_greedy, mask, q_vals, s_key)
            except Exception as e:
                logging.exception("BootstrappedPolicy.__call__: failed to append debug trace entry")
        # Perform one-step SARSA(λ) update for the previous decision (r=0 during episode)
        if self._started and self._prev_s is not None and self._prev_a is not None:
            gamma_eff: Optional[float] = None
            if self.gamma_per_second != 1.0 and self._prev_t is not None:
                try:
                    dt = max(0.0, float(now) - float(self._prev_t))
                except Exception as e:
                    logging.exception("BootstrappedPolicy.__call__: failed to compute dt; using 0.0")
                    dt = 0.0
                gamma_eff = float(self.gamma_per_second) ** float(dt)
            self.learner.update(self._prev_s, self._prev_a, r=0.0, s_next=s_key, a_next=a_id, gamma_override=gamma_eff)
        # Store current for next step
        self._prev_s = s_key
        self._prev_a = a_id
        self._prev_t = float(now)
        # Return integer action for SelfishMiner: -1 adopt, 0 hide, 1 publish-one
        if a_id not in (-1, 0, 1):
            # Safety clamp in case of unexpected mask
            a_id = 0
        return a_id

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
                            ctx: ScenarioCtx) -> Callable[[Any, float], int]:
    """Return a SelfishMiner-compatible policy callback returning -1/0/1.

    The callback inspects the current SelfishMiner state, discretizes it, selects a masked
    action with ε-greedy SARSA policy, and returns -1 (adopt), 0 (hide), or 1 (publish-one).
    """
    def _policy(miner: Any, now: float) -> Any:
        try:
            s_key, k_cap = disc.discretize(miner, ctx)
            mask = allowed_actions(miner, k_cap, now=now)
            # Baseline preference: if last is honest (h0/h1) → adopt (-1); if last=='s' → publish (1) if allowed else hide (0)
            last_raw = str(getattr(miner, "last", "h0"))
            last_is_honest = last_raw.startswith('h')
            preferred: Optional[int] = None
            if last_is_honest:
                preferred = -1
            else:
                preferred = 1 if (1 in mask) else 0
            a_id = learner.choose_action(s_key, mask, preferred=preferred)
            if a_id not in (-1, 0, 1):
                a_id = 0
            return a_id
        except Exception as e:
            logging.exception("policy_callback_factory: policy call failed; returning hide(0)")
            # Fallback to safe no-op
            return 0
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
        Bh=10; Bs=11; luck=False; last='h'; _withheld=[object()]
        def lead(self):
            return self.Bs - self.Bh
    fake = _Fake()
    act = policy_cb(fake, now=0.0)
    print("Policy callback smoke test (-1/0/1):", act)
    print("RL SARSA(λ) skeleton ready. Integrate with simulator by passing policy=policy_cb (returns -1/0/1) to SelfishMiner.")
