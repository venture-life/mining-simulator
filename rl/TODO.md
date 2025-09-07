# TODO: RL Integration for Selfish Miner

This document tracks the plan to integrate Reinforcement Learning for the selfish miner policy using the existing policy injection hook. No code changes yet; this is a planning artifact.

## Overview
- Objective: maximize episodic return at the end of a long simulation run.
- Policy injection: use `SelfishMiner(policy=...)` to decide actions without invoking the internal planner.
- Start with tabular SARSA(λ), move to actor–critic if needed.

## Tasks
- [ ] Define episodic RL objective and metrics
  - Return R = attacker_revenue_fraction (or attacker_revenue) at episode end.
  - Optional advantage baseline: R − α.
  - Optional time-aware discounting: γ_per_second^Δt.
- [ ] Specify state representation and discretization (include α, k, μ)
  - lead = Bs − Bh (clip to [-3..+3])
  - diff_w bins ∈ {−2, −1, 0, +1, +2}
  - luck ∈ {0,1}; last ∈ {h,s}
  - withheld_count clip to [0..3]
  - α bucketization: [0, 0.25), [0.25, 1/3), [1/3, 0.5], (0.5, 1)  — note the high bucket is strictly > 0.5
  - k as small integer (e.g., {2,3,4+} or exact when in range)
  - μ = Λ·D bucketization (e.g., low/med/high); optionally include mode flag {fixed_total, additive_attacker}
- [ ] Define action primitives (planner-free)
  - Perish ≡ adopt (discard private and follow public)
  - Publish(n): publish exactly n withheld blocks (FIFO); n=0 ≡ hide
  - Set Ncap = k and cap n ∈ [0 .. min(k, withheld_count)]
  - Dynamic masks: always allow n=0 and Perish; allow n>0 only if withheld_count≥n; preferably after last=='h' (or allow n=1 after last=='s' when lead≥2)
  - Legacy mapping: hide→Publish(0); reveal_one→Publish(1); match/override/even emerge from FRP once Publish(n) is executed
- [ ] SARSA(λ) algorithm
  - λ ≈ 0.9 eligibility traces; ε-greedy; γ ≈ 1.0 (or time-aware)
  - Learning rate schedule α_lr: 0.05 → 0.005
- [ ] Episode driver
  - Run `simulate_mining_eventqV2` for ~100k steps per episode, multiple seeds
  - Policy collects decision steps; terminal reward applied with eligibility traces
- [ ] Variance reduction / shaping
  - Advantage baseline (R − α)
  - Optional potential-based shaping Φ(s)=c1·lead−c2·max(0,−diff_w)+c3·1{luck}
- [ ] Evaluation protocol
  - Curriculum: α ∈ {0.2,0.3,0.4}, k ∈ {2,3,4}, modes ∈ {fixed_total, additive_attacker}
  - Report attacker_revenue_fraction, canonical_rate, time-to-convergence
- [ ] Environment interface
  - Gym-like wrapper around simulator; vectorized seeds for throughput
- [ ] Scalability path
  - Actor–critic (PPO/A2C) with two heads (action type, n) if tabular plateaus
- [ ] Hyperparameters
  - λ=0.9, ε 0.1→0.02, α_lr 0.05→0.005, optional γ_per_second

- [ ] Empirical MDP estimator (optional)
  - Log (state_key, action_id, next_state_key, Δt, terminal) from policy episodes
  - Estimate P̂(z'|z,a), E[Δt|z,a], and (optionally) successor representation under the learned policy
  - Validate by Bellman evaluation vs learned Q on-coverage

## Notes
- Terminal-only rewards avoid biasing toward premature reveal; eligibility traces handle long temporal dependencies (e.g., hide…hide…override).
- Keep actions planner-free to minimize runtime during learning; your structured plan path in `act()` supports (action, n) directly.
