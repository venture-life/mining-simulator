"""
Episodic SARSA(λ) trainer for SelfishMiner using the policy injection hook.

Usage (example):
  python -m rl.train_sarsa --episodes 10 --steps 10000 \
      --groups 3 --attacker-share 0.3 --k 3 --rate 0.0166667 --window 5.0 \
      --mode fixed_total --seed 123

Notes:
- The trainer runs true SARSA(λ) bootstrapping with per-decision updates via BootstrappedPolicy.
- Terminal reward per episode is attacker_revenue_fraction (optionally minus alpha when --baseline-advantage).
- The action space is {Perish, Publish(n)} with n∈[0..min(k, withheld_count)], where Publish(0) ≡ hide.
- This is a simple reference trainer; extend as needed (checkpointing Q, richer schedules, vectorized seeds).
"""
from __future__ import annotations

import argparse
import json
import ast
import math
import random
from typing import Any, Dict, Optional

from honest_mining import simulate_mining_eventqV2
from rl.sarsa import (
    ScenarioCtx,
    Discretizer,
    SarsaLambda,
    make_bootstrapped_policy,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train SARSA(λ) for SelfishMiner via policy injection")
    p.add_argument("--episodes", type=int, default=5000, help="Number of episodes (long runs)")
    p.add_argument("--steps", type=int, default=800, help="Steps per episode (simulate_mining_eventqV2)")
    p.add_argument("--groups", type=int, default=2, help="Number of honest groups (excluding attacker)")
    p.add_argument("--attacker-share", type=float, default=0.4199, dest="attacker_share", help="Attacker hashrate α in (0,1)")
    p.add_argument("--k", type=int, default=3, help="Dominance threshold k")
    p.add_argument("--rate", type=float, default=1.0/120.0, help="Global mining rate Λ (blocks/sec)")
    p.add_argument("--window", type=float, default=5.0, dest="D", help="Rival window D (seconds)")
    p.add_argument("--mode", type=str, choices=["fixed_total", "additive_attacker"], default="fixed_total",
                   help="Context flag only; does not alter the simulator here.")
    p.add_argument("--seed", type=int, default=None, help="Base RNG seed; episodes will increment it")
    p.add_argument("--gamma", type=float, default=1.0, help="Discount factor γ for SARSA")
    p.add_argument("--lambda", type=float, default=0.98, dest="lam", help="Eligibility trace decay λ")
    p.add_argument("--alpha", type=float, default=0.018, help="Learning rate α_lr (initial)")
    p.add_argument("--epsilon", type=float, default=0.06, help="Exploration ε (initial)")
    p.add_argument("--alpha-final", type=float, default=0.01, dest="alpha_final", help="Final learning rate")
    p.add_argument("--epsilon-final", type=float, default=0.005, dest="epsilon_final", help="Final exploration")
    p.add_argument("--baseline-advantage", action="store_true", help="Use (R - α) as terminal return")
    p.add_argument("--gamma-per-second", type=float, default=1.0,
                   help="Time-aware discount base γ_per_second; effective γ at each step is γ_per_second^Δt (default 1.0)")
    p.add_argument("--save-q", type=str, default=None, help="Path to save Q-table as JSON after training/evaluation")
    p.add_argument("--load-q", type=str, default=None, help="Path to load an existing Q-table JSON before running")
    p.add_argument("--eval-only", action="store_true", help="Run evaluation only (no learning): epsilon=0, alpha=0; prints returns")
    return p.parse_args()


def lin_schedule(start: float, end: float, t: int, T: int) -> float:
    if T <= 1:
        return end
    frac = max(0.0, min(1.0, t / float(T - 1)))
    return (1.0 - frac) * start + frac * end


def main() -> None:
    args = parse_args()

    mode_flag = 0 if args.mode == "fixed_total" else 1
    mu = float(args.rate) * float(args.D)

    ctx = ScenarioCtx(alpha=float(args.attacker_share), k=int(args.k), mu=mu, mode_flag=mode_flag)
    disc = Discretizer()
    agent = SarsaLambda(gamma=float(args.gamma), lam=float(args.lam), alpha=float(args.alpha), epsilon=float(args.epsilon))
    # Load Q-table if provided
    if args.load_q:
        try:
            with open(args.load_q, "r", encoding="utf-8") as f:
                data = json.load(f)
            loaded = 0
            for k_str, v in data.items():
                try:
                    key = ast.literal_eval(k_str)
                    if isinstance(key, tuple) and len(key) == 2:
                        agent.Q[key] = float(v)
                        loaded += 1
                except Exception:
                    continue
            print(f"Loaded {loaded} Q-entries from {args.load_q}")
        except FileNotFoundError:
            print(f"Warning: could not find Q-table at {args.load_q}; starting fresh.")
        except Exception as e:
            print(f"Warning: failed to load Q-table from {args.load_q}: {e}")
    policy = make_bootstrapped_policy(agent, disc, ctx, gamma_per_second=float(args.gamma_per_second))

    returns: list[float] = []

    base_seed: Optional[int] = args.seed

    for ep in range(int(args.episodes)):
        # Schedules (disabled in eval-only)
        if args.eval_only:
            agent.alpha = 0.0
            agent.epsilon = 0.0
        else:
            agent.alpha = lin_schedule(args.alpha, args.alpha_final, ep, args.episodes)
            agent.epsilon = lin_schedule(args.epsilon, args.epsilon_final, ep, args.episodes)
        # Seed per episode for stochasticity
        ep_seed = None if base_seed is None else int(base_seed + ep)

        policy.start_episode()
        R = 0.0
        sample_runs = 2
        for i in range(sample_runs):
            res = simulate_mining_eventqV2(
                steps=int(args.steps),
                groups=int(args.groups),
                Lambda=float(args.rate),
                D=float(args.D),
                k=int(args.k),
                seed=ep_seed+i+1,
                attacker_share=float(args.attacker_share),
                selfish_policy=policy,
            )
            d: Dict[str, Any] = res.to_dict()
            if args.baseline_advantage:
                R = R + float(d.get("attacker_revenue_fraction", 0.0)) - float(args.attacker_share)
            else:
                R = R + float(d.get("attacker_revenue_fraction", 0.0))
        R = R/sample_runs
        returns.append(R)
        # Terminal update (suppressed by alpha=0 in eval-only)
        policy.end_episode(R)

        print(f"Episode {ep+1}/{args.episodes} | R={R*100:.2f} | ε={agent.epsilon:.3f} α={agent.alpha:.4f}")

    avg_R = sum(returns) / max(1, len(returns))
    print(f"Average return over {len(returns)} episodes: {avg_R*100:.2f}")

    if args.save_q:
        # Serialize Q to a simple JSON dict with stringified keys
        q_out = {str(k): v for k, v in agent.Q.items()}
        with open(args.save_q, "w", encoding="utf-8") as f:
            json.dump(q_out, f)
        print(f"Saved Q-table to {args.save_q}")


if __name__ == "__main__":
    main()
