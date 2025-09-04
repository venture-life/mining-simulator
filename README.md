# Mining Simulator (Event-Queue V2 with Optional Attacker)

Project to simulate mining dynamics using a continuous-time event-queue model (global Poisson arrivals with thinning by share). Supports honest-only runs or scenarios with a selfish attacker, and includes optional plotting.

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
     --groups 3 --steps 1000 --rate 0.0166667 --window 5.0 --k 3 \
     --local-chains --save-local-chains out/local_chains_demo.svg
   ```
3. Add a selfish attacker with share α=0.40. The last lane in the plot will be the attacker:
   Note, the attacker currently is very cpu-intensive only steps <=2000 are recommended. (and not very effective yet. proper policy still needs to be implemented)
   ```bash
   python run_simulation.py mining-eventq-v2 \
     --groups 2 --steps 1000 --k 3 --attacker-share 0.40 \
     --rate 0.0166667 --window 5.0 \
     --local-chains --save-local-chains out/local_chains_attacker.svg
   ```
4. CLI help for all options:
   ```bash
   python run_simulation.py mining-eventq-v2 -h
   ```
## Example runs
honest-run:
   ```bash
   python3 run_simulation.py mining-eventq-v2 --groups 3 --steps 3000 --rate 0.0083333333 --window 5.0 --k 3 --seed 42`
   ```
> Event-queue V2 run:
> {'groups': 3, 'shares': \[0.3333333333333333, 0.3333333333333333, 0.3333333333333333\], 'Lambda': 0.0083333333, 'D': 5.0, 'steps': 3000, 
>  'fork\_fraction': 0.007333333333333333, 'total\_uncles': 22, 'canonical\_counts': \[998, 1012, 990\], 'uncle\_counts': \[7, 10, 5\], 
>  'revenue\_counts': \[998, 1012, 990\], 'revenue\_fraction': \[0.33266666666666667, 0.3373333333333333, 0.33\], 'attacker\_index': None, 
>  'S\_size\_hist\_1\_to\_G': \[2978, 22, 0\], 'max\_prop\_delay': 2.5, 'elapsed\_time': 362364.75972952275, 'mine\_events': 3022, 
>  'canonical\_rate': 0.008278950751831575, 'mine\_rate': 0.008339663057345006}

Selfish-Miner and 2 honest groups
   ```bash
   python3 run_simulation.py mining-eventq-v2 --groups 2 --attacker-share 0.4 --steps 3000 --rate 0.0083333333 --window 5.0 --k 3 --seed 42`
   ```

> Event-queue V2 run (attacker α=0.400)
> {'groups': 3, 'shares': \[0.3, 0.3, 0.4\], 'Lambda': 0.0083333333, 'D': 5.0, 'steps': 3000, 
>  'fork\_fraction': 0.4593333333333333, 'total\_uncles': 483, 'canonical\_counts': \[898, 837, 1265\], 'uncle\_counts': \[89, 105, 289\], 
>   'revenue\_counts': \[898, 837, 1265\], 'revenue\_fraction': \[0.29933333333333334, 0.279, 0.4216666666666667\], 
>   'attacker\_index': 2, 'attacker\_revenue': 1265, 'attacker\_revenue\_fraction': 0.4216666666666667, 
>   'S\_size\_hist\_1\_to\_G': \[1622, 1372, 6\], 'max\_prop\_delay': 2.5, 'elapsed\_time': 536651.254887309, 'mine\_events': 4556, 
>   'canonical\_rate': 0.005590222649586402, 'mine\_rate': 0.008489684797171881}

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
- Heilman, E., Narula, N., Tanzer, G., & Virza, M. (2020). Publish or Perish: A Backward-Compatible Defense Against Selfish Mining in Bitcoin.
