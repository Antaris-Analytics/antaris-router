"""
Cost-Performance SLA guarantees for Antaris Router — Sprint 5.

Provides SLAConfig, SLAMonitor, and supporting data-classes for
tracking, enforcing, and reporting Service Level Agreements on
latency, cost, quality, and budget.

Zero external dependencies. All logic is deterministic and in-process.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


# ── SLAConfig ─────────────────────────────────────────────────────────────────

@dataclass
class SLAConfig:
    """Configuration object for Cost-Performance SLAs.

    All fields are optional — only configured constraints are enforced.

    Attributes:
        max_latency_ms: Hard latency cap in milliseconds.  Routing will
            prefer faster models when estimated latency exceeds this value.
        cost_reduction_target: Fraction (0.0–1.0) of cost savings sought
            versus always routing to the most expensive model.
        min_quality_score: Minimum acceptable quality score (0.0–1.0).
            When a model's tracked quality falls below this, the router
            escalates to a better model.
        budget_per_hour_usd: Hard hourly spend cap in USD.  When exceeded,
            the router switches to cheaper alternatives.
        auto_escalate_on_breach: When *True*, automatically switch models
            on SLA breach.  When *False*, log breaches but keep the model.
    """

    max_latency_ms: Optional[float] = None
    cost_reduction_target: Optional[float] = None
    min_quality_score: Optional[float] = None
    budget_per_hour_usd: Optional[float] = None
    auto_escalate_on_breach: bool = True

    def __post_init__(self) -> None:
        """Validate all fields on construction."""
        if self.max_latency_ms is not None and self.max_latency_ms <= 0:
            raise ValueError(
                f"max_latency_ms must be > 0, got {self.max_latency_ms}"
            )
        if self.cost_reduction_target is not None:
            if not (0.0 <= self.cost_reduction_target <= 1.0):
                raise ValueError(
                    f"cost_reduction_target must be 0.0–1.0, "
                    f"got {self.cost_reduction_target}"
                )
        if self.min_quality_score is not None:
            if not (0.0 <= self.min_quality_score <= 1.0):
                raise ValueError(
                    f"min_quality_score must be 0.0–1.0, "
                    f"got {self.min_quality_score}"
                )
        if self.budget_per_hour_usd is not None and self.budget_per_hour_usd <= 0:
            raise ValueError(
                f"budget_per_hour_usd must be > 0, got {self.budget_per_hour_usd}"
            )


# ── SLARecord ─────────────────────────────────────────────────────────────────

@dataclass
class SLARecord:
    """Record of SLA compliance for a single routing decision."""

    timestamp: float
    model: str
    estimated_cost_usd: float
    estimated_latency_ms: float
    quality_score: Optional[float]  # None if unknown
    sla_compliant: bool
    breaches: List[str]      # e.g. ["latency_exceeded", "quality_below_threshold"]
    adjustments: List[str]   # e.g. ["escalated_model_due_to_quality_sla"]
    # Baseline cost for cost-savings calculation (most expensive available)
    baseline_cost_usd: float = 0.0


# ── SLAMonitor ────────────────────────────────────────────────────────────────

class SLAMonitor:
    """Track and enforce SLA compliance across routing decisions.

    Call :meth:`evaluate` for each routing decision to obtain breaches and
    adjustments.  Use :meth:`get_report` to generate period summaries and
    :meth:`check_budget_alert` to query spend status.
    """

    # Tier-based latency heuristics (ms) when no per-model data available.
    _TIER_LATENCY_MS: Dict[str, float] = {
        "trivial": 100.0,
        "simple": 300.0,
        "moderate": 700.0,
        "complex": 1_500.0,
        "expert": 3_000.0,
    }

    # Cost-band latency heuristics (avg $/1K tokens → estimated ms)
    _COST_LATENCY_BANDS: List[tuple] = [
        (0.0001, 80.0),   # near-zero (local / free models)
        (0.001,  200.0),
        (0.005,  500.0),
        (0.015, 1_000.0),
        (0.05,  2_000.0),
        (float("inf"), 4_000.0),
    ]

    def __init__(self, sla: SLAConfig) -> None:
        self.sla = sla
        self._records: List[SLARecord] = []
        # model → quality scores reported externally
        self._quality_scores: Dict[str, List[float]] = {}

    # ── Quality score ingestion ───────────────────────────────────────────

    def record_quality(self, model: str, score: float) -> None:
        """Record a quality score for a model (called externally after use).

        Args:
            model: Model name.
            score: Quality score 0.0–1.0.
        """
        if model not in self._quality_scores:
            self._quality_scores[model] = []
        self._quality_scores[model].append(max(0.0, min(1.0, float(score))))

    def get_avg_quality(self, model: str) -> Optional[float]:
        """Return the rolling average quality score for *model*, or *None*."""
        scores = self._quality_scores.get(model, [])
        if not scores:
            return None
        return sum(scores) / len(scores)

    # ── Latency estimation ────────────────────────────────────────────────

    def estimate_latency(
        self,
        model_name: str,
        tier: str,
        avg_cost_per_1k: float,
    ) -> float:
        """Estimate request latency in ms for a model.

        Uses cost-band heuristics when cost data is available, falling
        back to tier-based defaults.

        Args:
            model_name: Model name (not used directly, kept for future override).
            tier: Complexity tier of the routed task.
            avg_cost_per_1k: Average cost per 1K tokens (input+output)/2.

        Returns:
            Estimated latency in milliseconds.
        """
        if avg_cost_per_1k > 0:
            for threshold, latency in self._COST_LATENCY_BANDS:
                if avg_cost_per_1k <= threshold:
                    return latency
        return self._TIER_LATENCY_MS.get(tier, 1_000.0)

    # ── Core evaluation ───────────────────────────────────────────────────

    def evaluate(
        self,
        model: str,
        tier: str,
        estimated_cost_usd: float,
        avg_cost_per_1k: float,
        *,
        baseline_cost_usd: float = 0.0,
        quality_score: Optional[float] = None,
    ) -> SLARecord:
        """Evaluate SLA compliance for one routing decision.

        Call this *before* returning the decision from the router.  The
        resulting :class:`SLARecord` carries ``breaches`` and ``adjustments``
        that are attached to :class:`RoutingDecision`.

        Args:
            model: Selected model name.
            tier: Complexity tier.
            estimated_cost_usd: Estimated cost for this request.
            avg_cost_per_1k: Average cost per 1K tokens for latency estimation.
            baseline_cost_usd: Cost if always using most expensive model (for savings).
            quality_score: Optional externally provided quality score override.

        Returns:
            :class:`SLARecord` with compliance details.
        """
        sla = self.sla
        now = time.time()
        breaches: List[str] = []
        adjustments: List[str] = []

        estimated_latency = self.estimate_latency(model, tier, avg_cost_per_1k)

        # ── Latency SLA ───────────────────────────────────────────────────
        if sla.max_latency_ms is not None:
            if estimated_latency > sla.max_latency_ms:
                breaches.append("latency_exceeded")

        # ── Cost / budget SLA ─────────────────────────────────────────────
        if sla.budget_per_hour_usd is not None:
            hourly_spend = self._hourly_spend()
            if hourly_spend > sla.budget_per_hour_usd:
                breaches.append("budget_exceeded")

        # ── Quality SLA ───────────────────────────────────────────────────
        if sla.min_quality_score is not None:
            # Use provided score, then tracked average, then assume compliant
            eff_quality = quality_score
            if eff_quality is None:
                eff_quality = self.get_avg_quality(model)
            if eff_quality is not None and eff_quality < sla.min_quality_score:
                breaches.append("quality_below_threshold")

        # ── Adjustments ───────────────────────────────────────────────────
        if breaches and sla.auto_escalate_on_breach:
            if "latency_exceeded" in breaches:
                adjustments.append("preferred_faster_model_due_to_latency_sla")
            if "budget_exceeded" in breaches:
                adjustments.append("routed_to_cheaper_model_due_to_budget_sla")
            if "quality_below_threshold" in breaches:
                adjustments.append("escalated_model_due_to_quality_sla")

        record = SLARecord(
            timestamp=now,
            model=model,
            estimated_cost_usd=estimated_cost_usd,
            estimated_latency_ms=estimated_latency,
            quality_score=quality_score if quality_score is not None else self.get_avg_quality(model),
            sla_compliant=len(breaches) == 0,
            breaches=breaches,
            adjustments=adjustments,
            baseline_cost_usd=baseline_cost_usd,
        )
        self._records.append(record)
        return record

    # ── Spend helpers ─────────────────────────────────────────────────────

    def _hourly_spend(self, window_seconds: float = 3_600.0) -> float:
        """Compute total estimated spend in the rolling window."""
        cutoff = time.time() - window_seconds
        return sum(
            r.estimated_cost_usd
            for r in self._records
            if r.timestamp >= cutoff
        )

    def _cost_savings(self, window_seconds: float = 3_600.0) -> float:
        """Compute cost savings vs always using the baseline (expensive) model."""
        cutoff = time.time() - window_seconds
        return sum(
            r.baseline_cost_usd - r.estimated_cost_usd
            for r in self._records
            if r.timestamp >= cutoff and r.baseline_cost_usd > 0
        )

    # ── Reporting ─────────────────────────────────────────────────────────

    def get_report(self, since_hours: float = 1.0) -> Dict[str, Any]:
        """Generate an SLA compliance report for the specified period.

        Args:
            since_hours: How many hours back to analyse (default: 1).

        Returns:
            Dict with compliance_rate, breaches, adjustments_made,
            cost_savings_usd, avg_latency_ms, and budget_utilization.
        """
        window = since_hours * 3_600.0
        cutoff = time.time() - window
        records = [r for r in self._records if r.timestamp >= cutoff]

        if not records:
            return {
                "period_hours": since_hours,
                "compliance_rate": 1.0,
                "breaches": {"latency": 0, "cost": 0, "quality": 0},
                "adjustments_made": 0,
                "cost_savings_usd": 0.0,
                "avg_latency_ms": 0.0,
                "budget_utilization": 0.0,
                "total_requests": 0,
            }

        total = len(records)
        compliant = sum(1 for r in records if r.sla_compliant)
        compliance_rate = round(compliant / total, 4)

        latency_breaches = sum(
            1 for r in records if "latency_exceeded" in r.breaches
        )
        budget_breaches = sum(
            1 for r in records if "budget_exceeded" in r.breaches
        )
        quality_breaches = sum(
            1 for r in records if "quality_below_threshold" in r.breaches
        )

        adjustments_made = sum(len(r.adjustments) for r in records)

        total_spend = sum(r.estimated_cost_usd for r in records)
        cost_savings = sum(
            r.baseline_cost_usd - r.estimated_cost_usd
            for r in records
            if r.baseline_cost_usd > 0
        )
        cost_savings = max(0.0, round(cost_savings, 6))

        avg_latency = (
            sum(r.estimated_latency_ms for r in records) / total
        )

        budget_util = 0.0
        if self.sla.budget_per_hour_usd:
            budget_util = round(
                total_spend / self.sla.budget_per_hour_usd, 4
            )

        return {
            "period_hours": since_hours,
            "compliance_rate": compliance_rate,
            "breaches": {
                "latency": latency_breaches,
                "cost": budget_breaches,
                "quality": quality_breaches,
            },
            "adjustments_made": adjustments_made,
            "cost_savings_usd": cost_savings,
            "avg_latency_ms": round(avg_latency, 2),
            "budget_utilization": budget_util,
            "total_requests": total,
        }

    def check_budget_alert(self) -> Dict[str, Any]:
        """Return current budget alert status.

        Returns:
            Dict with status (ok/warning/critical), hourly_spend_usd,
            budget_usd, utilization, projected_hourly_usd, and
            recommendation.
        """
        budget = self.sla.budget_per_hour_usd

        if budget is None:
            return {
                "status": "ok",
                "hourly_spend_usd": None,
                "budget_usd": None,
                "utilization": None,
                "projected_hourly_usd": None,
                "recommendation": "No budget cap configured.",
            }

        now = time.time()
        one_hour_ago = now - 3_600.0
        fifteen_min_ago = now - 900.0

        hourly_spend = self._hourly_spend()
        utilization = round(hourly_spend / budget, 4)

        # Project hourly rate from the last 15 min
        recent_spend = sum(
            r.estimated_cost_usd
            for r in self._records
            if r.timestamp >= fifteen_min_ago
        )
        projected = round(recent_spend * 4.0, 6)  # 15 min × 4 = 1 hour

        if utilization >= 1.0:
            status = "critical"
            recommendation = (
                "Budget exhausted — route all remaining requests to "
                "the cheapest available model."
            )
        elif utilization >= 0.80:
            status = "warning"
            recommendation = (
                "Approaching budget limit — consider routing to cheaper "
                "models for the remainder of this hour."
            )
        else:
            status = "ok"
            recommendation = "Budget utilization is within acceptable limits."

        return {
            "status": status,
            "hourly_spend_usd": round(hourly_spend, 6),
            "budget_usd": budget,
            "utilization": utilization,
            "projected_hourly_usd": projected,
            "recommendation": recommendation,
        }


# ── Tier-based latency helper (standalone, used by Router) ────────────────────

TIER_LATENCY_MS: Dict[str, float] = {
    "trivial": 100.0,
    "simple": 300.0,
    "moderate": 700.0,
    "complex": 1_500.0,
    "expert": 3_000.0,
}
