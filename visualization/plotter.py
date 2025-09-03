from __future__ import annotations

from typing import List, Dict, Optional
import math

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.ticker import MultipleLocator, FuncFormatter, Locator



def plot_local_chains_eventq_v2(
    result: Dict,
    title: str = "",
    save_path: Optional[str] = None,
    show: bool = False,
    annotate_ids: bool = False,
    annotate_hw: bool = False,
    annotate_ihw: bool = False,
    discrete_layout: bool = True,
    slot_width: float = 1.0,
    box_pad: float = 0.1,
    box_vpad: float = 0.05,
    rect_height_frac: Optional[float] = None,
    rect_height_max_frac: Optional[float] = 0.20,
    inch_per_step: float = 0.18,
    max_edges: Optional[int] = None,
    scale_lanes_by_sublanes: bool = True,
    lane_growth_per_track: float = 0.65,
    track_persistence_slots: int = 8,
):
    """
    Plot per-miner local chains (knowledge) over time. Each lane is a miner; whenever a miner
    first learns about a block (by mining it or receiving it), we place a node on that miner's
    lane at the corresponding time slot. Parent->child edges are drawn within the same lane,
    capturing branching/forks in that miner's local view.

    To avoid overlaps when multiple blocks are first seen at the same time by a miner, each
    miner lane is subdivided into vertical sub-tracks. Blocks in the same slot occupy separate
    sub-tracks while keeping full slot width, improving aspect ratio and readability. When
    annotate_hw is True, the label is rendered on two lines: "height" on the first line and
    "weight" on the second line.

    Parameters mirror `plot_swimlanes_eventq_v2` for layout control and consistency.
    Colors indicate the producing miner of each block (origin), regardless of who knows it.

    Lane sizing: by default (scale_lanes_by_sublanes=True), each miner lane's height grows with the
    number of sub-tracks S it needs to display concurrent items. Lane height is computed as
    H_lane = 1.0 + lane_growth_per_track * (S - 1). Set scale_lanes_by_sublanes=False to keep
    equal-height lanes.

    Padding and size controls:
    - box_pad controls horizontal padding within a slot: rectangle width = slot_width * (1 - 2*box_pad).
    - box_vpad controls vertical padding within a lane track: auto rectangle height uses
      step * (1 - 2*box_vpad), where step is spacing between sub-tracks (or the full available
      lane height when S==1). Typical range: 0..0.5.
    - rect_height_frac optionally forces a fixed rectangle height = rect_height_frac * lane_height
      (bypassing track-based auto sizing).
    - rect_height_max_frac caps auto rectangle height to a fraction of lane height (default 0.20).
    - track_persistence_slots keeps sublanes "active" for this many discrete timestamps after
      last use. This bounds growth by recent concurrency rather than historical branches, and
      lets new branches reuse lower sublanes once old ones expire.
    """
    # Accept both dataclass and dict
    groups = result.get("groups") if isinstance(result, dict) else result.groups
    trace = result.get("trace") if isinstance(result, dict) else getattr(result, "trace", None)
    elapsed = result.get("elapsed_time") if isinstance(result, dict) else getattr(result, "elapsed_time", None)

    if not trace:
        raise ValueError("Trace is empty. Run simulate_mining_eventqV2(..., trace=True) to enable tracing.")

    # Collect block metadata: origin miner and parent id
    origin: Dict[str, int] = {}
    parent: Dict[str, Optional[str]] = {}
    height_map: Dict[str, Optional[int]] = {}
    weight_map: Dict[str, Optional[float]] = {}
    for ev in trace:
        if ev["type"] == "MINE":
            bid = str(ev.get("block_id"))
            origin[bid] = int(ev.get("miner"))
            parent[bid] = ev.get("parent_id")
            height_map[bid] = ev.get("height")
            weight_map[bid] = ev.get("weight")
        elif ev["type"] == "DELIVER":
            bid = str(ev.get("block_id"))
            if bid not in parent:
                parent[bid] = ev.get("parent_id")
        elif ev["type"] == "PUBLISH":
            # Selfish miner publication (no direct MINE trace); treat like MINE for origin metadata
            bid = str(ev.get("block_id"))
            origin[bid] = int(ev.get("miner"))
            parent[bid] = ev.get("parent_id")
            height_map[bid] = ev.get("height")
            weight_map[bid] = ev.get("weight")

    # Build first-seen times per miner for each block
    seen_time: Dict[int, Dict[str, float]] = {m: {} for m in range(groups)}
    # MINE: miner sees immediately at t
    for ev in trace:
        if ev["type"] == "MINE":
            m = int(ev["miner"]) ; bid = str(ev["block_id"]) ; t = float(ev["t"])
            seen_time[m][bid] = min(t, seen_time[m].get(bid, t))
        elif ev["type"] == "PUBLISH":
            # Publication by selfish miner: mark first-seen in attacker's lane at publish time
            m = int(ev["miner"]) ; bid = str(ev["block_id"]) ; t = float(ev["t"])
            seen_time[m][bid] = min(t, seen_time[m].get(bid, t))
    # DELIVER: 'to' miner sees at t_deliver
    for ev in trace:
        if ev["type"] == "DELIVER":
            m = int(ev["to"]) ; bid = str(ev["block_id"]) ; t = float(ev.get("t_deliver", 0.0))
            prev = seen_time[m].get(bid)
            seen_time[m][bid] = t if prev is None else min(prev, t)

    # Gather timeline values for layout mapping: use the same global discrete slot mapping
    all_times = []
    for ev in trace:
        if ev["type"] == "MINE":
            all_times.append(float(ev["t"]))
        elif ev["type"] == "DELIVER":
            all_times.append(float(ev.get("t_deliver", 0.0)))
            all_times.append(float(ev.get("t_mine", 0.0)))
        elif ev["type"] == "PUBLISH":
            all_times.append(float(ev.get("t", 0.0)))

    if not all_times:
        raise ValueError("Trace has no timestamps.")

    unique_ts = sorted(set(all_times))
    n_slots = len(unique_ts)

    if discrete_layout:
        t_to_x = {t: i * slot_width for i, t in enumerate(unique_ts)}
        x_min = 0.0
        x_max = (n_slots - 1) * slot_width + slot_width
        x_label = "event steps"
    else:
        t_min = 0.0
        t_max = max(all_times) if all_times else (elapsed or 1.0)
        if elapsed is not None:
            t_max = max(t_max, float(elapsed))
        # Use real time; compute rectangle widths per-unique timestamp based on local spacing
        D_val = float(result.get("D", 5.0)) if isinstance(result, dict) else float(getattr(result, "D", 5.0))
        Lam = float(result.get("Lambda", 1.0 / 60.0)) if isinstance(result, dict) else float(getattr(result, "Lambda", 1.0 / 60.0))
        mean_dt = (1.0 / Lam) if Lam > 0 else 60.0
        # Sensible fallback width in seconds
        fallback_w = max(0.8, min(0.8 * D_val, 0.35 * mean_dt))
        # Dynamic width per timestamp: 0.6 * min(gap_left, gap_right), clamped to [0.6*fallback, 2.0*fallback]
        width_of_t: Dict[float, float] = {}
        for i, t in enumerate(unique_ts):
            left_gap = (t - unique_ts[i - 1]) if i > 0 else fallback_w
            right_gap = (unique_ts[i + 1] - t) if i + 1 < len(unique_ts) else fallback_w
            dyn = 0.6 * max(1e-9, min(left_gap, right_gap))
            width_of_t[t] = max(0.6 * fallback_w, min(2.0 * fallback_w, dyn))
        pad = 0.5 * max(width_of_t.get(unique_ts[0], fallback_w), width_of_t.get(unique_ts[-1], fallback_w))
        t_to_x = None
        x_min = t_min - pad
        x_max = t_max + pad
        x_label = "time (min)"

    # Colors by producing miner (origin)
    cmap = plt.get_cmap("tab10")
    colors = [cmap(i % 10) for i in range(groups)]

    # First pass: per-timestamp sublane assignment with reuse of tracks
    # Build per-miner grouping of blocks by first-seen time
    pos: Dict[int, Dict[str, tuple]] = {m: {} for m in range(groups)}
    by_t_per_miner: Dict[int, Dict[float, List[str]]] = {m: {} for m in range(groups)}
    for m in range(groups):
        m_by_t: Dict[float, List[str]] = {}
        for bid, t in seen_time[m].items():
            m_by_t.setdefault(t, []).append(bid)
        by_t_per_miner[m] = m_by_t

    # Assign track indices per miner per timestamp with a persistence window.
    # Tracks remain active for 'track_persistence_slots' timestamps after last use, allowing
    # growth to reflect recent concurrency and then compact again as branches die out.
    assigned_index: Dict[int, Dict[float, Dict[str, int]]] = {m: {} for m in range(groups)}
    parent_track: Dict[int, Dict[str, int]] = {m: {}}  # block id -> track index when assigned
    track_tip_block: Dict[int, Dict[int, str]] = {i: {} for i in range(groups)}  # track idx -> last block id placed    
    lane_max_idx: Dict[int, int] = {m: 0 for m in range(groups)}
    active_tracks: Dict[int, Dict[int, int]] = {i: {} for i in range(groups)}  # track_idx -> last_used_slot (index in ordered_ts)
    for m in range(groups):
        ordered_ts = sorted(by_t_per_miner[m].keys())
        # Defensive: ensure per-miner map exists
        if m not in active_tracks:
            active_tracks[m] = {}
        if m not in parent_track:
            parent_track[m] = {}
        if m not in track_tip_block:
            track_tip_block[m] = {}
        for slot_idx, t in enumerate(ordered_ts):
            bids = list(by_t_per_miner[m][t])
            # Stable order for drawing: by height then origin then id
            bids.sort(key=lambda b: ((height_map.get(b) if height_map.get(b) is not None else -1), origin.get(b, 0), b))

            # Expire stale active tracks
            if track_persistence_slots is not None and track_persistence_slots >= 0:
                expire_before = slot_idx - int(track_persistence_slots)
                to_del = [idx for idx, last in active_tracks[m].items() if last < expire_before]
                for idx in to_del:
                    del active_tracks[m][idx]

            occupied = set(active_tracks[m].keys())  # reserved indices from recent past
            used: set[int] = set()
            assigned_index[m][t] = {}

            for b in bids:
                pid = parent.get(b)
                pref = None
                if pid is not None and pid in parent_track[m]:
                    p_idx = parent_track[m][pid]
                    # Prefer parent's track only if it's still active AND the parent is the current tip on that track
                    if p_idx in occupied and track_tip_block[m].get(p_idx) == pid:
                        pref = p_idx

                # Choose track: reuse parent's active track if possible; otherwise, pick nearest free
                def take_nearest_free(start_pref: Optional[int]) -> int:
                    # Allow using the preferred track even if it's in 'occupied' (reserved) as long as
                    # it's not already used in this timestamp.
                    if start_pref is not None and start_pref not in used:
                        return start_pref
                    # Otherwise, find the smallest index not conflicting with current-slot assignments
                    # or other reserved tracks.
                    blocked = set(occupied)
                    j = 0
                    while j in used or j in blocked:
                        j += 1
                    return j

                j = take_nearest_free(pref)
                assigned_index[m][t][b] = j
                used.add(j)
                occupied.add(j)
                parent_track[m][b] = j
                active_tracks[m][j] = slot_idx
                # Update the tip on the chosen track
                track_tip_block[m][j] = b
                
            if occupied:
                lane_max_idx[m] = max(lane_max_idx[m], max(occupied))

    # Compute per-miner number of tracks and lane heights (bounded by max concurrent tracks per timestamp)
    lane_tracks: Dict[int, int] = {}
    lane_heights: Dict[int, float] = {}
    for m in range(groups):
        S = max(1, lane_max_idx[m] + 1)
        lane_tracks[m] = S
        if scale_lanes_by_sublanes:
            lane_heights[m] = 1.0 + float(lane_growth_per_track) * float(S - 1)
        else:
            lane_heights[m] = 1.0

    # Y-position mapping: place miner 0 at the top, miners increasing downward
    heights_list = [lane_heights[m] for m in range(groups)]
    cum_heights_from_bottom = [0.0]
    for m in range(1, groups):
        cum_heights_from_bottom.append(cum_heights_from_bottom[-1] + heights_list[m - 1])
    total_height = sum(heights_list) if heights_list else 0.0
    def y_of(m: int) -> float:
        # Center of lane m measured from bottom, then flipped so miner 0 is at the top
        center_from_bottom = cum_heights_from_bottom[m] + 0.5 * heights_list[m]
        return total_height - center_from_bottom

    # Figure sizing takes total lane height into account
    fig_w = max(12.0, inch_per_step * n_slots)
    fig_h = max(3.0, 1.1 * total_height + 2)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    # Draw miner labels (no center raster lines)
    for m in range(groups):
        y = y_of(m)
        ax.text(x_min - 0.02 * (x_max - x_min), y, f"Miner {m}", va="center", ha="right", fontsize=9)

    # Lane background fills per miner (brightened miner color), replacing thick separators
    for m in range(groups):
        y = y_of(m)
        lane_h = lane_heights[m]
        yb = y - 0.5 * lane_h
        r, g, b = colors[m][:3]
        t = 0.80  # lighten toward white
        bg = (r + (1 - r) * t, g + (1 - g) * t, b + (1 - b) * t, 1.0)
        ax.add_patch(Rectangle((x_min, yb), x_max - x_min, lane_h, facecolor=bg, edgecolor="none", zorder=0))

    # Choose rectangle height and compute track centers per miner
    for m in range(groups):
        y_lane = y_of(m)
        m_by_t = by_t_per_miner[m]
        # Normalize track indices to 0..S-1
        S = lane_tracks[m]
        lane_h = lane_heights[m]
        lane_margin = 0.12 * lane_h
        available_h = max(0.2, lane_h - 2 * lane_margin)
        # Vertical spacing per track; when S==1, treat the entire available lane as one track
        if S == 1:
            step = available_h
            track_centers = [y_lane]
        else:
            step = available_h / (S - 1)
            start = y_lane - 0.5 * available_h
            track_centers = [start + i * step for i in range(S)]
        # Compute rectangle height with configurable vertical padding and optional overrides
        if rect_height_frac is not None:
            # Fixed fraction of lane height
            rect_h = max(0.0, float(rect_height_frac)) * lane_h
        else:
            # Auto: based on inter-track spacing and vertical padding
            pad_y_factor = max(0.0, 1.0 - 2.0 * float(box_vpad))
            rect_h = max(0.0, step * pad_y_factor)
            # Optional cap to keep boxes from dominating the lane
            if rect_height_max_frac is not None:
                rect_h = min(rect_h, max(0.0, float(rect_height_max_frac)) * lane_h)
        # Keep within the lane's available vertical space and ensure a minimal visible height
        rect_h = min(rect_h, available_h)
        rect_h = max(rect_h, 0.02 * lane_h)

        # Place rectangles time by time; use assigned persistent tracks
        for t in sorted(m_by_t.keys()):
            bids = m_by_t[t]
            # Stable order for drawing: by height then origin then id
            bids.sort(key=lambda b: ((height_map.get(b) if height_map.get(b) is not None else -1), origin.get(b, 0), b))
            # Compute x and width for this timestamp
            x = t_to_x[t] if t_to_x is not None else t
            if discrete_layout:
                width = slot_width * max(0.0, 1.0 - 2.0 * box_pad)
                slot_left = x + slot_width * box_pad
            else:
                w = width_of_t.get(t, fallback_w)
                width = w
                slot_left = x - 0.5 * width
            for bid in bids:
                idx = assigned_index[m].get(t, {}).get(bid, 0)
                idx = max(0, min(S - 1, idx))
                cy = track_centers[idx]
                bottom = cy - 0.5 * rect_h
                prod = origin.get(bid, 0)
                rect = Rectangle((slot_left, bottom), width, rect_h, facecolor=colors[prod], edgecolor="k", linewidth=0.5, zorder=10)
                ax.add_patch(rect)
                cx = slot_left + 0.5 * width
                if annotate_ihw:
                    h = height_map.get(bid)
                    wv = weight_map.get(bid)
                    wtxt = "?" if wv is None else (f"{float(wv):.1f}" if isinstance(wv, (int, float)) else str(wv))
                    txt = f"{bid}\n{h}\n{wtxt}"
                    ax.text(cx, cy, txt, ha="center", va="center", fontsize=5.8, color="#000000", zorder=11)
                elif annotate_hw:
                    h = height_map.get(bid)
                    wv = weight_map.get(bid)
                    wtxt = "?" if wv is None else (f"{float(wv):.1f}" if isinstance(wv, (int, float)) else str(wv))
                    txt = f"{h}\n{wtxt}"
                    ax.text(cx, cy, txt, ha="center", va="center", fontsize=6.0, color="#000000", zorder=11)
                elif annotate_ids:
                    ax.text(cx, cy, str(bid), ha="center", va="center", fontsize=6.0, color="#000000", zorder=11)
                pos[m][bid] = (cx, cy)

    # Second pass: draw parent->child edges within each lane to show branching
    for m in range(groups):
        for bid, (cx, cy) in pos[m].items():
            pid = parent.get(bid)
            if pid is None:
                continue
            ppos = pos[m].get(pid)
            if ppos is None:
                continue  # parent not in view (should be rare due to repair); skip
            px, py = ppos
            # Draw above rectangles so branches remain visible
            if abs(px - cx) < 1e-9:
                # Same-slot linkage: route via a small left dogleg to avoid center text
                dx = (0.15 * slot_width) if discrete_layout else (0.02 * (x_max - x_min))
                xL = cx - dx
                style = dict(color="#444444", linewidth=1.2, alpha=0.95, zorder=4)
                ax.plot([cx, xL], [cy, cy], **style)
                ax.plot([xL, xL], [cy, py], **style)
                ax.plot([xL, cx], [py, py], **style)
            else:
                ax.plot([px, cx], [py, cy], color="#444444", linewidth=1.2, alpha=0.95, zorder=4)

    # Dotted back-links to uncles (from child to referenced uncles in the same miner's local view)
    for ev in trace:
        if ev["type"] != "MINE":
            continue
        m = int(ev["miner"])
        child_id = str(ev.get("block_id"))
        u_list = ev.get("uncles") or []
        if not u_list:
            continue
        cpos = pos[m].get(child_id)
        if cpos is None:
            continue
        cx, cy = cpos
        for uid in u_list:
            upos = pos[m].get(str(uid))
            if upos is None:
                continue
            ux, uy = upos
            ax.plot([cx, ux], [cy, uy], linestyle=":", color="#888888", linewidth=1.0, alpha=0.9, zorder=3)

    # Third pass: overlay delivery arrows (network propagation), similar to swimlanes
    edges_drawn = 0
    for ev in trace:
        if ev["type"] != "DELIVER":
            continue
        if max_edges is not None and edges_drawn >= max_edges:
            continue
        t0 = float(ev.get("t_mine", ev.get("t_deliver", x_min)))
        t1 = float(ev.get("t_deliver", t0))
        f = ev.get("from")
        to = ev.get("to")
        if f is None or to is None:
            continue
        f = int(f); to = int(to)
        # Route arrows to actual sub-lane positions if known
        bid = str(ev.get("block_id"))
        y0 = y_of(f)
        p_from = pos[f].get(bid)
        if p_from is not None:
            y0 = p_from[1]
        y1 = y_of(to)
        p_to = pos[to].get(bid)
        if p_to is not None:
            y1 = p_to[1]
        # Direct arrow from source center to target center (previous style)
        repair = bool(ev.get("repair", False))
        # Prefer exact block centers if positions are known; fall back to time-mapped x
        x0 = (p_from[0] if p_from is not None else (t_to_x[t0] if t_to_x is not None else t0))
        x1 = (p_to[0] if p_to is not None else (t_to_x[t1] if t_to_x is not None else t1))
        style = dict(
            arrowstyle="->",
            color=colors[f],
            lw=1.0,
            alpha=0.7,
            linestyle="--" if repair else "-",
            shrinkA=2,
            shrinkB=2,
            zorder=4,
        )
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0), arrowprops=style)
        edges_drawn += 1

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(-0.5, total_height + 0.5)
    ax.set_xlabel(x_label)
    # In time layout, show sparse major labels (adaptive) and minute gridlines via minor ticks
    if not discrete_layout:
        span = max(x_max - x_min, 1e-9)
        target_max_ticks = 80.0
        # Candidate step sizes in seconds (1m, 2m, 5m, 10m, 15m, 30m, 60m, 2h, 4h)
        step_candidates = [60.0, 120.0, 300.0, 600.0, 900.0, 1800.0, 3600.0, 7200.0, 14400.0]
        tick_step = step_candidates[-1]
        for s in step_candidates:
            if (span / s) <= target_max_ticks:
                tick_step = s
                break
        ax.xaxis.set_major_locator(MultipleLocator(tick_step))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{int(round(x / 60.0))}"))
        # Minor gridlines: choose the finest spacing that keeps tick count <= MAXTICKS
        ax.set_axisbelow(True)
        minor_candidates = [60.0, 120.0, 300.0, 600.0, 900.0, 1800.0, 3600.0, 7200.0, 14400.0]
        minor_step = None
        for s in minor_candidates:
            if (span / s) <= Locator.MAXTICKS:
                minor_step = s
                break
        if minor_step is not None:
            ax.xaxis.set_minor_locator(MultipleLocator(minor_step))
            ax.grid(True, which="minor", axis="x", alpha=0.35, linewidth=0.9, color="#555555", zorder=2)
    ax.set_yticks([])
    # Keep y-grid off; x-grid handled above
    ax.grid(False, axis="y")
    if not title:
        title = "Per-miner local chains over time"
    ax.set_title(title)

    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=160)
        print(f"Saved local chains -> {save_path}")

    if show:
        plt.show()

    return fig


