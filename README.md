# Mining Simulator (Event-Queue V2 with Optional Attacker)

A lightweight Python project to simulate Bitcoin mining dynamics using a continuous-time event-queue model (global Poisson arrivals with thinning and de-duplication). Supports honest-only runs or scenarios with a selfish attacker, and includes optional plotting.

## Features
- Event-queue V2 simulator (subcommand: `mining-eventq-v2`):
  - Global Poisson arrival process with per-miner thinning and parent repair
  - In-time rival window `D` (seconds) controlling uncle eligibility
  - Fork-resolution dominance `k` exposed via CLI and API (`--k`), applied to the longest-chain rule
  - Optional selfish attacker via `--attacker-share α` (adds one attacker miner)
- Local-chains plotting of each miner's knowledge over time (sub-tracks per timestamp; lanes auto-scale by concurrency)
- Deterministic tie-breaking consistent across miners

## Quickstart
1. Create/activate a virtual environment (recommended) and install dependencies:
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```
2. Run Event-Queue V2 (honest-only) with a custom longest-chain dominance `k` and save a local-chains plot:
   ```bash
   python run_simulation.py mining-eventq-v2 \
     --groups 3 --steps 100000 --rate 0.0166667 --window 5.0 --k 3 \
     --local-chains --save-local-chains out/local_chains_demo.svg
   ```
3. Add a selfish attacker with share α=0.30 and include `k` in the plot title automatically:
   ```bash
   python run_simulation.py mining-eventq-v2 \
     --groups 3 --steps 80000 --k 4 --attacker-share 0.30 \
     --local-chains --save-local-chains out/local_chains_attacker_k4.svg
   ```
4. CLI help for all options:
   ```bash
   python run_simulation.py mining-eventq-v2 -h
   ```

## Notes
- Event-Queue V2 uses a global Poisson process with per-delivery delays and parent repair. The in-time rival window is `D` seconds; `k` is the dominance threshold for longest-chain resolution.
- In `mining-eventq-v2`, you can include a selfish attacker by setting `--attacker-share α ∈ (0,1)`; this adds one attacker miner as the last lane.
- Local-chains plotting scales each miner lane by its concurrency (number of sub-tracks) by default: `scale_lanes_by_sublanes=True`, `lane_growth_per_track=0.35`. Titles include `k` automatically from the CLI.
- Plotting is optional and only imported when requested.

## Project Structure
```
honest_mining/
  __init__.py
  miner.py
  simulator.py
  v2.py
visualization/
  plotter.py
run_simulation.py
requirements.txt
README.md
```

## References
- Eyal, I., & Sirer, E. G. (2014). Majority is not Enough: Bitcoin Mining is Vulnerable. Financial Cryptography and Data Security.
- Heilman, E., Narula, N., Tanzer, G., & Virza, M. (2020). Publish or Perish: A Backward-Compatible Defense Against Selfish Mining in Bitcoin.
