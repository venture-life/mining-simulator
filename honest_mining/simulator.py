from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import List, Dict, Optional

try:
    import numpy as np
except ImportError as e:
    raise SystemExit(
        "numpy is required. Please install dependencies: `pip install -r requirements.txt`"
    ) from e


@dataclass
class HonestEventqResult:
    groups: int
    shares: List[float]
    Lambda: float  # global block rate (blocks/second)
    D: float       # window seconds
    steps: int

    canonical_counts: List[int]
    uncle_counts: List[int]
    fork_heights: int
    S_size_hist: List[int]  # index s counts |S| == s, length = groups+1, we will use indices 0..G
    max_prop_delay: float
    timing: Optional[Dict] = None  # optional timing metrics when enabled
    effective_rival_window: Optional[float] = None
    # Optional runtime stats (only populated by event-queue V2 which tracks wall-clock time)
    elapsed_time: Optional[float] = None
    mine_events: Optional[int] = None
    # Optional event trace for visualization (only when enabled by simulator)
    trace: Optional[List[Dict]] = None
    # Optional: attacker index (when a selfish miner is appended as the last miner)
    attacker_index: Optional[int] = None

    @property
    def total_uncles(self) -> int:
        return int(sum(self.uncle_counts))

    @property
    def fork_fraction(self) -> float:
        return float(self.fork_heights) / float(self.steps) if self.steps > 0 else 0.0

    @property
    def mean_S_size(self) -> float:
        # Exclude index 0, valid sizes 1..G
        total = 0
        denom = 0
        for s, c in enumerate(self.S_size_hist):
            if s == 0:
                continue
            total += s * c
            denom += c
        return float(total) / float(denom) if denom > 0 else 1.0

    @property
    def size_hist(self) -> List[int]:
        """Histogram over sizes 1..G (trimmed), excluding the always-zero 0-bin."""
        return self.S_size_hist[1:]

    def to_dict(self) -> Dict:
        d = asdict(self)
        d["total_uncles"] = self.total_uncles
        d["fork_fraction"] = self.fork_fraction
        d["mean_S_size"] = self.mean_S_size
        # Convenience: include trimmed size histogram for printing
        d["S_size_hist_1_to_G"] = self.size_hist
        d["max_prop_delay"] = 0.5 * self.D
        # Include effective rival window if present
        if self.effective_rival_window is not None:
            d["effective_rival_window"] = self.effective_rival_window
        # Revenue (parents-only): identical to canonical_counts; also provide fractions
        rev_counts = list(self.canonical_counts)
        d["revenue_counts"] = rev_counts
        denom = float(sum(rev_counts)) if rev_counts else 0.0
        if denom > 0.0:
            d["revenue_fraction"] = [float(c) / denom for c in rev_counts]
        else:
            d["revenue_fraction"] = [0.0 for _ in rev_counts]
        # If attacker index is known, add attacker-specific revenue fields
        if self.attacker_index is not None and 0 <= int(self.attacker_index) < len(rev_counts):
            ai = int(self.attacker_index)
            d["attacker_index"] = ai
            d["attacker_revenue"] = int(rev_counts[ai])
            d["attacker_revenue_fraction"] = (float(rev_counts[ai]) / denom) if denom > 0.0 else 0.0
        # Derived rate sanity checks when elapsed_time is available
        et = d.get("elapsed_time")
        if isinstance(et, (int, float)) and et and et > 0:
            d["canonical_rate"] = float(self.steps) / float(et)
            me = d.get("mine_events")
            if isinstance(me, (int, float)) and me and me > 0:
                d["mine_rate"] = float(me) / float(et)
        return d

