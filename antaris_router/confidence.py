"""
Confidence-gated routing, provider health tracking, and A/B testing
for Antaris Router — Sprint 7.

Zero external dependencies. All logic is deterministic and file-free
(health/AB state is in-process; persist externally if needed).
"""

from __future__ import annotations

import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

# ── Confidence basis constants ────────────────────────────────────────────────
CONFIDENCE_BASIS_SEMANTIC  = "semantic_classifier"
CONFIDENCE_BASIS_QUALITY   = "quality_tracker"
CONFIDENCE_BASIS_RULE      = "rule_based"
CONFIDENCE_BASIS_KEYWORD   = "keyword_density"
CONFIDENCE_BASIS_COMPOSITE = "composite"

VALID_CONFIDENCE_BASES = {
    CONFIDENCE_BASIS_SEMANTIC,
    CONFIDENCE_BASIS_QUALITY,
    CONFIDENCE_BASIS_RULE,
    CONFIDENCE_BASIS_KEYWORD,
    CONFIDENCE_BASIS_COMPOSITE,
}

# ── Escalation strategy constants ─────────────────────────────────────────────
ESCALATION_ALWAYS   = "always"
ESCALATION_LOG_ONLY = "log_only"
ESCALATION_ASK      = "ask"

VALID_ESCALATION_STRATEGIES = {ESCALATION_ALWAYS, ESCALATION_LOG_ONLY, ESCALATION_ASK}


# ── Provider Health Tracking ──────────────────────────────────────────────────

@dataclass
class ProviderEvent:
    """A single recorded event for a provider/model."""
    model: str
    event: str           # "success" | "error" | "timeout"
    timestamp: float
    latency_ms: Optional[float] = None
    details: Optional[str] = None


class ProviderHealthTracker:
    """Track real-time health and performance of models/providers.

    Maintains a sliding window of events (default 1 hour) and computes
    health statistics on demand.  All state is in-process; serialize
    ``_events`` externally if persistence across restarts is required.

    Attributes:
        WINDOW_SECONDS: Size of the rolling health window in seconds.
    """

    WINDOW_SECONDS: int = 3_600  # 1-hour window

    def __init__(self) -> None:
        # model_name → bounded deque of ProviderEvent
        self._events: Dict[str, deque] = defaultdict(lambda: deque(maxlen=10_000))
        # model_name → unix timestamp when rate limit was last hit
        self._rate_limited: Dict[str, float] = {}

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------

    def record_event(
        self,
        model: str,
        event: str,
        latency_ms: Optional[float] = None,
        details: Optional[str] = None,
    ) -> None:
        """Record a provider event.

        Args:
            model: Model name string.
            event: One of ``"success"``, ``"error"``, ``"timeout"``.
            latency_ms: Response latency in milliseconds (optional).
            details: Optional detail string, e.g. ``"rate_limited"``.
        """
        self._events[model].append(
            ProviderEvent(
                model=model,
                event=event,
                timestamp=time.time(),
                latency_ms=latency_ms,
                details=details,
            )
        )
        # Track rate limit events
        if event == "error" and details and (
            "rate_limit" in details or "429" in details
        ):
            self._rate_limited[model] = time.time()

    # ------------------------------------------------------------------
    # Querying
    # ------------------------------------------------------------------

    def _recent_events(self, model: str) -> List[ProviderEvent]:
        """Return events within the rolling window for *model*."""
        cutoff = time.time() - self.WINDOW_SECONDS
        return [e for e in self._events.get(model, []) if e.timestamp >= cutoff]

    def get_health(self, model: str) -> Dict[str, Any]:
        """Compute current health status for *model*.

        Returns a dict with keys:
            model, status, success_rate_1h, avg_latency_ms,
            recent_errors, last_seen
        """
        recent = self._recent_events(model)

        if not recent:
            return {
                "model": model,
                "status": "unknown",
                "success_rate_1h": None,
                "avg_latency_ms": None,
                "recent_errors": 0,
                "last_seen": None,
            }

        total = len(recent)
        successes = sum(1 for e in recent if e.event == "success")
        errors = sum(1 for e in recent if e.event in ("error", "timeout"))
        success_rate = successes / total

        latencies = [e.latency_ms for e in recent if e.latency_ms is not None]
        avg_latency: Optional[float] = (
            round(sum(latencies) / len(latencies), 1) if latencies else None
        )

        if success_rate >= 0.95:
            status = "healthy"
        elif success_rate >= 0.70:
            status = "degraded"
        else:
            status = "down"

        last_event = max(recent, key=lambda e: e.timestamp)
        last_seen = datetime.fromtimestamp(
            last_event.timestamp, tz=timezone.utc
        ).strftime("%Y-%m-%dT%H:%M:%SZ")

        return {
            "model": model,
            "status": status,
            "success_rate_1h": round(success_rate, 4),
            "avg_latency_ms": avg_latency,
            "recent_errors": errors,
            "last_seen": last_seen,
        }

    def get_health_score(self, model: str) -> float:
        """Return a 0.0–1.0 health score for routing preference.

        Unknown models receive a neutral score of 0.5.
        """
        health = self.get_health(model)
        if health["status"] == "unknown":
            return 0.5
        rate = health["success_rate_1h"] or 0.5
        avg_lat = health["avg_latency_ms"] or 1_000
        # Latency factor: 0 ms → 1.0, 10 000 ms → ~0.5
        lat_factor = 1.0 / (1.0 + avg_lat / 10_000)
        return round(0.7 * rate + 0.3 * lat_factor, 4)

    def is_available(self, model: str) -> bool:
        """Return *True* unless the model's status is ``"down"``."""
        return self.get_health(model)["status"] != "down"

    def is_rate_limited(self, model: str, backoff_seconds: float = 60) -> bool:
        """Return *True* if *model* was rate-limited within the last *backoff_seconds*.

        Args:
            model: Model name.
            backoff_seconds: How long (in seconds) to consider a rate-limit sticky.

        Returns:
            True when the model is in backoff due to a rate-limit error.
        """
        ts = self._rate_limited.get(model)
        if ts is None:
            return False
        return (time.time() - ts) < backoff_seconds

    def known_models(self) -> List[str]:
        """Return list of all models with at least one recorded event."""
        return list(self._events.keys())


# ── A/B Testing ───────────────────────────────────────────────────────────────

@dataclass
class _ABVariantStats:
    """Internal per-variant accumulator."""
    requests: int = 0
    quality_scores: List[float] = field(default_factory=list)

    @property
    def avg_quality(self) -> Optional[float]:
        if not self.quality_scores:
            return None
        return round(sum(self.quality_scores) / len(self.quality_scores), 4)


class ABTest:
    """A simple A/B test between two routing strategies.

    Uses deterministic round-based assignment (not random) so that with
    ``split=0.5`` requests alternate a–b–a–b, and with ``split=0.75``
    every 4th request goes to B.  This avoids drift at small sample sizes.

    Example::

        ab = ABTest("test-name", "cost_optimized", "quality_first", split=0.5)
        variant = ab.assign_variant()  # "a" or "b"
        ab.record_result(variant, quality_score=0.9)
        stats = ab.get_stats()
    """

    def __init__(
        self,
        name: str,
        strategy_a: str,
        strategy_b: str,
        split: float = 0.5,
    ) -> None:
        """
        Args:
            name: Human-readable identifier for the test.
            strategy_a: Strategy label for variant A.
            strategy_b: Strategy label for variant B.
            split: Fraction of traffic routed to variant A (0.0–1.0).
        """
        self.name = name
        self.strategy_a = strategy_a
        self.strategy_b = strategy_b
        self.split = max(0.0, min(1.0, split))
        self._stats: Dict[str, _ABVariantStats] = {
            "a": _ABVariantStats(),
            "b": _ABVariantStats(),
        }
        self._counter: int = 0

    # ------------------------------------------------------------------
    # Routing
    # ------------------------------------------------------------------

    def assign_variant(self) -> str:
        """Assign the next request to a variant.

        Returns:
            ``"a"`` or ``"b"``.
        """
        self._counter += 1
        if self.split <= 0.0:
            return "b"
        if self.split >= 1.0:
            return "a"
        # Determine cycle length and which slot(s) get B
        if abs(self.split - 0.5) < 1e-9:
            # Exactly 50/50 — simple alternation
            return "a" if self._counter % 2 == 1 else "b"
        elif self.split > 0.5:
            # More A: every N-th request is B
            cycle = max(2, round(1.0 / (1.0 - self.split)))
            return "b" if (self._counter % cycle == 0) else "a"
        else:
            # More B: every N-th request is A
            cycle = max(2, round(1.0 / self.split))
            return "a" if (self._counter % cycle == 0) else "b"

    # ------------------------------------------------------------------
    # Outcome tracking
    # ------------------------------------------------------------------

    def record_result(
        self, variant: str, quality_score: Optional[float] = None
    ) -> None:
        """Record an outcome for a variant.

        Args:
            variant: ``"a"`` or ``"b"``.
            quality_score: Optional 0.0–1.0 quality rating.
        """
        if variant not in self._stats:
            return
        self._stats[variant].requests += 1
        if quality_score is not None:
            self._stats[variant].quality_scores.append(float(quality_score))

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    def get_stats(self) -> Dict[str, Any]:
        """Return current test statistics.

        Returns a dict with keys:
            name, strategy_a, strategy_b, split, variant_a, variant_b
        """
        return {
            "name": self.name,
            "strategy_a": self.strategy_a,
            "strategy_b": self.strategy_b,
            "split": self.split,
            "variant_a": {
                "requests": self._stats["a"].requests,
                "avg_quality": self._stats["a"].avg_quality,
            },
            "variant_b": {
                "requests": self._stats["b"].requests,
                "avg_quality": self._stats["b"].avg_quality,
            },
        }

    def winner(self) -> Optional[str]:
        """Return ``"a"``, ``"b"``, or *None* (too early / too close).

        Requires at least 10 samples per variant and a quality gap > 0.05.
        """
        sa = self._stats["a"]
        sb = self._stats["b"]
        if sa.requests < 10 or sb.requests < 10:
            return None
        qa = sa.avg_quality or 0.5
        qb = sb.avg_quality or 0.5
        if abs(qa - qb) < 0.05:
            return None
        return "a" if qa > qb else "b"

    def __repr__(self) -> str:  # pragma: no cover
        stats = self.get_stats()
        return (
            f"ABTest({self.name!r}, "
            f"a={stats['variant_a']['requests']}, "
            f"b={stats['variant_b']['requests']})"
        )
