"""Event-queue mining simulator utilities.

Exports:
- simulate_honest_eventq: full-window + thinning model with Bernoulli presence per group
- simulate_mining_eventqV2: global Poisson look-ahead with thinning + dedup per group (optionally with attacker)
- HonestEventqResult: dataclass with summary statistics
"""
from .simulator import HonestEventqResult
from .v2 import simulate_mining_eventqV2, simulate_honest_eventqV2
