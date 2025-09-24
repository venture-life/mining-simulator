# Simulation results

The selfish-mining strategy used is trailing-stubborn [1] as it was shown that the original Selfish-Mine strategy by Eyal and Sirer (2014) is sub-optimal. Given the nature of Publish-Perish trailing-stubborn has achieved the best fit against the countermeasure. Further selfish mining strategies may be explored with RL/sarsa, this however is not working yet.

The theoretical upper bound for any selfish-mining strategy is given by it's hashrate share ```α```, namely: ```α/(1-α)```, so, for 40% hashrate it equals 66.6%, which is only achievable with a connectiviy edge (```ɣ=1```) ***and*** having the honest hashrate divided by many honest sub-groups. 

The following parameters were set for the simulation results:
1. ```seed=22800-22851``` (52 runs for mean, min, max, median and stdev)
2. ```steps=3000``` for each run (#steps denote the canonical chain height assessed by honest miner[0])
3. ```burnout-selfish= 3```: The cutoff max_steps - burnout-selfish denote the point in time in after which the selfish miner aggressively publishes all withheld blocks and behaves honestly afterwards. This is done in order to avoid skewing the performance by having a long withheld private chain at the end of the simulation
4. ```ƛ = 1/120```: Mining rate. 1 block every 2 minutes.
5. ```max_propagation_delay=2.5s```. the propagation is modeled by a lognormal distribution with a mean of ```max_prop_delay/2``` and capped at ```max_prop_delay```. Connectivity boost by the selfish miner of factor 2 would have the max_prop_delay (receive and send) for the selfish-miner halved.
6. ```hashrate_mode=fixed_total``` (alternative is additive_attacker). this simulates whether the attacker joins prior difficulty adjustment
7. ```tie-break= random-tie-break```. This is not relevant for plain vanilla PoW since ```D=0.0```. For ```D>0``` this value can be set to deterministic selection
8. ```ɑ=0.4``` Selfish-miner's hashrate share
9. ```groups``` see the table below. It denotes the number of honest miners. for ```alpha=0.4 and groups = 6```, for example, there would be 6 honest miners with 10% hashrate each
10. ```connectivity-edge``` selfish-miner's edge on network connectivity (see max_propagation_delay). If set to 1, the selfish-miner has no network edge

The results are in percentages. Note, for alpha=0.4, 40 is the honest baseline performance.
A high connectivity-edge (eg, factor 250) models roughly a gamma of 1. 

## Plain vanilla PoW 

This mining-simulator implements Publish-or-Perish modifications to honest miner's head selection. However, plain vanilla PoW can be assessed with the following parameters

1. k = 1 (always switch immediately to the heaviest work branch)
2. D = 0.0 (if there are is a tie / more than one branch with the same work, choose the one you have seen first)

**alpha=0.4**
| conn_edge | 1   | 1   | 1   | 250   | 250   | 250   |
|-----------|-------|-------|-------|-------|-------|-------|
| **groups**    | **2**     | **3**     | **6**     | **2**     | **3**     | **6**     |
| **mean**      | 47.34 | 47.36 | 48.28 |53.86 | 55.61 |57.43 |
| **min**       | 42.73   | 41.90  | 43.83   |49.97 | 50.77 | 53.53 |
| **max**       | 53.07   | 51.80  | 54.03   |58.80 | 59.83 | 62.03 |
| **median**    | 47.27   | 47.40  | 48.30  |53.89 | 55.64 | 57.20 |
| **stdev**     | 2.35   | 2.55  | 2.29   |2.24 | 2.16 | 1.95 |



## Publish-or-Perish with k=3 and D=5.0


**alpha=0.4**
| conn_edge | 1   | 1   | 250   | 250   |
|-----------|-----|-----|-------|-------|
| **groups**    | **2**     | **6**     |  **2**    | **6**     |
| **mean**      | 47.63 | 47.80 | 48.26 |48.07 |
| **min**       | 41.7 | 42.53 | 43.27 |43.20 |
| **max**       | 52.43	 | 53.50 | 56.2	 |54.97	 |
| **median**    | 48.03 | 47.95	 | 47.92	 |48.22 |
| **stdev**     | 2.22 | 2.68 | 2.54 |2.18 |



## Publish-or-Perish with k=3 and D=5.0 and deterministic tie-break

**alpha=0.4**
| conn_edge | 1   | 1   | 250   | 250   |
|-----------|-----|-----|-------|-------|
| **groups**    | **2**     | **6**     |  **2**    | **6**     |
| **mean**      | 47.70 | 47.95 | 48.01 |47.70 |
| **min**       | 42.53 | 43.23 | 40.77 |43 |
| **max**       | 53.4 | 53.37	 | 53.23 |52.87	 |
| **median**    | 47.97 | 47.95	 | 48.05 |47.59 |
| **stdev**     | 2.18 | 2.39 | 2.46 |2.15 |



### References
[1] Nayak, Kartik, et al. "Stubborn mining: Generalizing selfish mining and combining with an eclipse attack." 2016 ieee european symposium on security and privacy (euros&P). IEEE, 2016.
