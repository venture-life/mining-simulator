# RL Module (SARSA)

This directory contains a lightweight, on-policy RL setup for learning a selfish-mining policy using tabular SARSA(λ) with eligibility traces. It integrates with the simulator via the `SelfishMiner(policy=...)` injection hook.

## Files

- `sarsa.py`
  - `ScenarioCtx`: static context for a scenario (α, k, μ, mode).
  - `Discretizer`: maps `SelfishMiner` + `ScenarioCtx` to a compact tabular state key.
  - `SarsaLambda`: tabular SARSA(λ) learner with replacing traces and masked ε‑greedy.
  - `BootstrappedPolicy`: stateful policy object that performs true SARSA bootstrapping (per‑decision) and applies a terminal update at episode end. Supports optional time‑aware discounting via `gamma_per_second`.
  - `make_bootstrapped_policy(...)`: factory for `BootstrappedPolicy`.
  - `policy_callback_factory(...)`: optional stateless callback factory (convenient for quick tests or MC ablations).

- `train_sarsa.py`
  - CLI trainer that runs episodes, logs returns, supports save/load of Q‑tables, eval‑only mode, and time‑aware discounting.

## State and Action Design

- State features (from `SelfishMiner` and scenario context):
  - `lead = Bs − Bh` clipped to `[-3..+3]`.
  - `diff_w = Wh − Ws` binned into `{−2, −1, 0, +1, +2}`.
  - `luck ∈ {0,1}` (exists withheld in‑time block that embeds ≥1 honest in‑time uncle, in PRIVATE view).
  - `last ∈ {h, s}` (producer of last perceived block).
  - `withheld_count` clipped to `[0..min(3, k)]`.
  - α bucket: `[0, 0.25)`, `[0.25, 1/3)`, `[1/3, 0.5]`, `(0.5, 1)`.
  - k bucket: `{k2, k3, k4+}`.
  - μ bucket: `{mulo, mumi, muhi}` (based on Λ·D thresholds).
  - mode flag: `{m0, m1}` for `{fixed_total, additive_attacker}`.

- Action space (planner‑free):
  - `Perish` ≡ `'adopt'` (discard private, follow public).
  - `Publish(n)` via `('reveal', n)` (FIFO); `n=0` ≡ hide.
  - Masks: always allow Perish and `Publish(0)`; allow `Publish(n>0)` only if `withheld_count ≥ n` and `n ≤ k` (i.e., `Ncap = k`).

- Baseline tie‑break preference (built into policies):
  - If `last == 'h'` → prefer Perish.
  - If `last == 's'` → prefer `Publish(1)` if feasible, else hide.
  - This only affects greedy tie‑breaks; learning can override when Q indicates a better choice.

## Time‑Aware Discounting (Optional)

- Default: `γ_per_second = 1.0` (no time discount across wall‑clock).
- When enabled, each per‑decision update uses `γ_eff = (γ_per_second)^Δt`, where `Δt` is the elapsed simulation time between the previous and current policy calls. Useful to softly penalize delays.

Example choices:

```text
half_life_seconds = T
γ_per_second = exp(-ln(2)/T)
```

## Using the Policy Programmatically

```python
from honest_mining import simulate_mining_eventqV2
from rl.sarsa import ScenarioCtx, Discretizer, SarsaLambda, make_bootstrapped_policy

Lambda = 1.0/60.0
D = 5.0
ctx = ScenarioCtx(alpha=0.30, k=3, mu=Lambda*D, mode_flag=0)

disc = Discretizer()
agent = SarsaLambda(gamma=1.0, lam=0.9, alpha=0.05, epsilon=0.1)
policy = make_bootstrapped_policy(agent, disc, ctx, gamma_per_second=1.0)

policy.start_episode()
res = simulate_mining_eventqV2(
    steps=10000,
    groups=3,
    Lambda=Lambda,
    D=D,
    k=ctx.k,
    attacker_share=ctx.alpha,
    selfish_policy=policy,
)
R = res.to_dict()["attacker_revenue_fraction"]
policy.end_episode(R)
```

## Trainer CLI

Train and save Q:

```bash
python -m rl.train_sarsa \
  --episodes 50 --steps 10000 \
  --groups 3 --attacker-share 0.3 --k 3 \
  --rate 0.0166667 --window 5.0 \
  --save-q out/q_sarsa.json
```

Evaluate (greedy, no learning):

```bash
python -m rl.train_sarsa \
  --episodes 10 --steps 10000 \
  --groups 3 --attacker-share 0.3 --k 3 \
  --rate 0.0166667 --window 5.0 \
  --eval-only --load-q out/q_sarsa.json
```

Time‑aware discounting example (optional):

```bash
# About 50% discount every 30 seconds
python -m rl.train_sarsa \
  --episodes 50 --steps 10000 \
  --groups 3 --attacker-share 0.3 --k 3 \
  --rate 0.0166667 --window 5.0 \
  --gamma-per-second 0.9772
```

## Q‑Table Format

- Stored as a JSON dict: keys are stringified Python tuples `(state_key, action_id) → float`.
- `train_sarsa.py` loads with `ast.literal_eval` and ignores malformed entries.
- `--save-q` writes the current Q after the run; `--load-q` preloads Q before training/eval.

## Notes & Tips

- The simulator accepts the policy via `simulate_mining_eventqV2(..., selfish_policy=callback)`.
- The policy returns `'adopt'` for Perish and `('reveal', n)` for Publish(n); this bypasses the internal planner.
- Reward is terminal only by default (`attacker_revenue_fraction`), with per‑decision SARSA updates bootstrapping on Q.
- Keep γ=1 initially if you care only about terminal revenue fraction; enable `--gamma-per-second` if you want to bias against long delays.

## Roadmap (Optional)

- Empirical MDP estimator from rollouts (estimate P̂(z'|z,a), E[Δt|z,a), SR).
- Linear/tile‑coded approximator for better generalization across α, k, μ.
- Actor–critic (PPO/A2C) with two heads (action type and n) using the same masks.
