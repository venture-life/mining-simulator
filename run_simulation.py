#!/usr/bin/env python3
from __future__ import annotations

import argparse
from honest_mining import simulate_mining_eventqV2

def main() -> None:
    p = argparse.ArgumentParser(description="Event-Queue V2 Mining Simulator")
    sub = p.add_subparsers(dest="cmd")

    # Event-queue V2 (global Poisson look-ahead + thinning, optional attacker)
    p_mine_v2 = sub.add_parser("mining-eventq-v2", help="Run event-queue V2 (global Poisson look-ahead + thinning); optionally include an attacker miner")
    p_mine_v2.add_argument("--groups", type=int, default=3, help="Number of honest groups (pools)")
    p_mine_v2.add_argument("--shares", type=str, default=None, help="Comma-separated shares summing to 1 (default: equal)")
    p_mine_v2.add_argument("--steps", type=int, default=1000, help="Number of heights to simulate")
    p_mine_v2.add_argument("--rate", type=float, default=1.0/120.0, help="Global block rate Λ (blocks/sec)")
    p_mine_v2.add_argument("--window", type=float, default=5.0, help="Δ window seconds for in-window rivals")
    p_mine_v2.add_argument("--k", type=int, default=3, help="Fork-resolution dominance k for the longest-chain rule")
    p_mine_v2.add_argument("--seed", type=int, default=None, help="RNG seed")
    p_mine_v2.add_argument("--track-times", action="store_true", help="Also simulate and report first rival timing stats")
    p_mine_v2.add_argument("--time-bins", type=int, default=50, help="Histogram bins for timing when --track-times is set")
    p_mine_v2.add_argument("--timing-verbose", action="store_true", help="Print full timing/streaks diagnostics; default prints compact sanity summary only")
    p_mine_v2.add_argument("--attacker-share", type=float, default=None, help="Attacker (selfish miner) share α in (0,1); if set, adds one attacker miner. Honest shares are rescaled to sum to (1-α): use provided --shares as relative weights, else uniform (1-α)/groups.")
    # Total hashrate interpretation and concurrency control
    p_mine_v2.add_argument(
        "--total-hashrate-mode",
        choices=["fixed_total", "additive_attacker"],
        default="fixed_total",
        help=(
            "How to interpret total hashrate when an attacker is present. "
            "'fixed_total' (default): α displaces honest hashrate, Σ shares = 1, global Λ is unchanged. "
            "This models a constant-difficulty, constant-total-hashrate scenario. "
            "'additive_attacker': attacker joins ON TOP of a fixed honest baseline BEFORE difficulty adjustment; "
            "total hashrate increases by f = 1/(1-α) and Λ is scaled by f to reflect faster blocks pre-retarget."
        ),
    )
    # Local chains (per-miner knowledge) visualization
    p_mine_v2.add_argument("--local-chains", action="store_true", help="Plot per-miner local chains with branching on each lane")
    p_mine_v2.add_argument("--save-local-chains", type=str, default=None, help="Path to save local chains plot (PNG/SVG)")
    p_mine_v2.add_argument("--max-edges", type=int, default=None, help="Limit number of delivery arrows to draw")
    p_mine_v2.add_argument("--annotate-ihw", action="store_true", help="Annotate id, height, weight on three lines inside nodes (i\\nh\\nw)")
    # Layout controls
    p_mine_v2.add_argument("--time-layout", action="store_true", help="Use real time on x-axis (may overlap boxes); default is discrete slots per timestamp")
    p_mine_v2.add_argument("--slot-width", type=float, default=1.0, help="Slot width in data units when using discrete layout")
    p_mine_v2.add_argument("--trace-limit", type=int, default=None, help="Limit number of events recorded in trace")

    # Require a subcommand
    args = p.parse_args()
    if args.cmd is None:
        p.print_help()
        return

    if args.cmd == "mining-eventq-v2":
        # Parse shares if provided
        if args.shares is not None:
            try:
                shares = [float(x) for x in args.shares.split(",") if x.strip() != ""]
            except ValueError:
                raise SystemExit("--shares must be a comma-separated list of floats, e.g., '0.33,0.33,0.34'")
        else:
            shares = None

        want_trace = bool(args.local_chains or args.save_local_chains)
        # Compute effective Λ and D based on hashrate mode and attacker α
        lambda0 = float(args.rate)
        D0 = float(args.window)
        mode = getattr(args, "total_hashrate_mode", "fixed_total")
        a = args.attacker_share
        f = 1.0
        if mode == "additive_attacker" and a is not None:
            # Attacker joins on top of a fixed honest baseline (pre-difficulty adjustment)
            # Honest shares are still rescaled to (1-α) for thinning, but we scale Λ by f = 1/(1-α)
            # so that honest miners' absolute rates remain unchanged while total frequency increases.
            if not (0.0 < float(a) < 1.0):
                raise SystemExit("--attacker-share must be in (0,1) when using additive_attacker mode")
            f = 1.0 / (1.0 - float(a))
        Lambda_eff = lambda0 * f
        D_eff = D0  # Physical mode only: keep D fixed; concurrency μ varies with Λ

        res = simulate_mining_eventqV2(
            steps=args.steps,
            groups=args.groups,
            shares=shares,
            Lambda=Lambda_eff,
            D=D_eff,
            k=args.k,
            seed=args.seed,
            track_times=args.track_times,
            time_bins=args.time_bins,
            trace=want_trace,
            trace_limit=args.trace_limit,
            attacker_share=args.attacker_share,
        )
        d = res.to_dict()
        keys = (
            "groups",
            "shares",
            "Lambda",
            "D",
            "steps",
            "fork_fraction",
            "total_uncles",
            "canonical_counts",
            "uncle_counts",
            # revenue (parents-only, same as canonical_counts)
            "revenue_counts",
            "revenue_fraction",
            "attacker_index",
            "attacker_revenue",
            "attacker_revenue_fraction",
            "S_size_hist_1_to_G",
            "max_prop_delay",
            # Sanity-check metrics from V2
            "elapsed_time",
            "mine_events",
            "canonical_rate",
            "mine_rate",
        )
        out = {k: d[k] for k in keys if k in d}
        # Add scenario metadata
        out["total_hashrate_mode"] = mode
        out["hashrate_factor"] = f
        out["mu"] = float(out.get("Lambda", Lambda_eff)) * float(out.get("D", D_eff))
        if getattr(args, "track_times", False) and d.get("timing") is not None:
            if getattr(args, "timing_verbose", False):
                out["timing"] = d["timing"]
            else:
                t = dict(d["timing"])  # shallow copy
                summary = {
                    "enabled": bool(t.get("enabled", False)),
                    "first_rival_fraction": t.get("first_rival_fraction"),
                    "mean_first_rival_time": t.get("mean_first_rival_time"),
                }
                s = t.get("streaks") or None
                if isinstance(s, dict):
                    summary["streaks"] = {"sanity": s.get("sanity")}
                out["timing"] = summary
        header = "Event-queue V2 run (attacker α={:.3f})".format(args.attacker_share) if args.attacker_share is not None else "Event-queue V2 run:"
        print(header)
        print(out)

        # Optional visualizations using the event trace
        if want_trace:
            from visualization.plotter import plot_local_chains_eventq_v2
            total_G = int(d.get("groups", args.groups))
            Lambda_print = float(d.get("Lambda", Lambda_eff))
            D_print = float(d.get("D", D_eff))
            mu_print = Lambda_print * D_print
            if args.attacker_share is not None:
                title = (
                    f"V2 (G={total_G}, Λ={Lambda_print:.5f} 1/s, D={D_print}s, k={args.k}, "
                    f"α={args.attacker_share:.3f}, mode={mode}, μ={mu_print:.3f})"
                )
            else:
                title = f"V2 (G={total_G}, Λ={Lambda_print:.5f} 1/s, D={D_print}s, k={args.k}, μ={mu_print:.3f})"
            # Internal visualization defaults (CLI flags removed)
            box_pad = 0.06
            inch_per_step = 1.5 if args.time_layout else 0.3
            # Per-miner local chains (knowledge-centric)
            if args.local_chains or args.save_local_chains:
                plot_local_chains_eventq_v2(
                    res, title=f"Local chains – {title}",
                    save_path=args.save_local_chains,
                    show=bool(args.local_chains),
                    annotate_ihw=bool(args.annotate_ihw),
                    max_edges=args.max_edges,
                    discrete_layout=(not args.time_layout),
                    slot_width=float(args.slot_width),
                    box_pad=box_pad,
                    inch_per_step=inch_per_step,
                )


if __name__ == "__main__":
    main()
