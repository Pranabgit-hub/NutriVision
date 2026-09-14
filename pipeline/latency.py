"""
Tiny in-process latency tracker.

Every external model call in this pipeline (the Hugging Face VLM call in
`vlm_classifier.py`, the Hugging Face chat call in `chatbot.py`) is the
kind of network-bound, variable-latency hop where an average is close to
useless -- a handful of slow calls (cold model load on HF's side, a retry,
a large image) can dominate real user-perceived latency without moving the
mean much. So this module tracks **p95/p99**, not just averages, per named
operation.

This is intentionally simple (in-memory, per-process, no persistence) --
enough to `print(LATENCY.summary("vlm_classify"))` after a demo run or to
back a `/metrics` endpoint on a real server. For a multi-process/multi-node
deployment you'd export these to something like Prometheus histograms
instead; the percentile math here (nearest-rank on a sorted sample) is a
fine estimate for a single-process demo but isn't a substitute for that.

See `service/gateway.go` for the analogous (Go-side) latency tracker for
the network-facing gateway hop, kept as a *separate* measurement from these
Python-side model-call latencies on purpose -- conflating "time in the Go
edge" with "time inside the Python model call" hides which one you'd
actually need to optimize during an incident.
"""
from __future__ import annotations

import time
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass


@dataclass
class LatencySummary:
    count: int
    p50_ms: float | None
    p95_ms: float | None
    p99_ms: float | None
    max_ms: float | None

    def as_dict(self) -> dict:
        return {
            "count": self.count,
            "p50_ms": self.p50_ms,
            "p95_ms": self.p95_ms,
            "p99_ms": self.p99_ms,
            "max_ms": self.max_ms,
        }


class LatencyTracker:
    """Records wall-clock durations per named operation and reports percentiles."""

    def __init__(self) -> None:
        self._samples: dict[str, list[float]] = defaultdict(list)

    @contextmanager
    def track(self, name: str):
        """Usage: `with LATENCY.track("vlm_classify"): ...call the model...`"""
        start = time.perf_counter()
        try:
            yield
        finally:
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            self._samples[name].append(elapsed_ms)

    def _percentile(self, name: str, p: float) -> float | None:
        data = sorted(self._samples.get(name, []))
        if not data:
            return None
        # Nearest-rank method -- simple and fine for demo-sized sample counts.
        idx = min(int(round(p / 100 * (len(data) - 1))), len(data) - 1)
        return data[idx]

    def summary(self, name: str) -> LatencySummary:
        data = self._samples.get(name, [])
        return LatencySummary(
            count=len(data),
            p50_ms=self._percentile(name, 50),
            p95_ms=self._percentile(name, 95),
            p99_ms=self._percentile(name, 99),
            max_ms=max(data) if data else None,
        )

    def reset(self, name: str | None = None) -> None:
        if name is None:
            self._samples.clear()
        else:
            self._samples.pop(name, None)


# Module-level singleton so every caller in the pipeline shares one tracker
# without threading an instance through every function signature.
LATENCY = LatencyTracker()
