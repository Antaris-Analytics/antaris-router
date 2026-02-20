"""
Main routing interface for Antaris Router.

Combines classification, model registry, and cost tracking to provide
intelligent model selection based on task complexity and cost optimization.

Sprint 7: Adds confidence-gated routing, escalation, explainability,
provider health tracking, cost forecasting, and A/B testing.

Sprint 5: Adds SLA configuration, SLA monitoring, smart auto-scaling
(fallback chains), budget alerts, and cost optimization suggestions.
"""

from __future__ import annotations

import hashlib
import logging
from typing import Any, Dict, List, Optional

from dataclasses import dataclass, field

from .config import Config
from .classifier import TaskClassifier, ClassificationResult
from .registry import ModelRegistry, ModelInfo
from .costs import CostTracker
from .confidence import (
    ProviderHealthTracker,
    ABTest,
    CONFIDENCE_BASIS_KEYWORD,
    CONFIDENCE_BASIS_COMPOSITE,
    CONFIDENCE_BASIS_RULE,
    ESCALATION_ALWAYS,
    ESCALATION_LOG_ONLY,
    ESCALATION_ASK,
    VALID_ESCALATION_STRATEGIES,
)
from .sla import SLAConfig, SLAMonitor

_log = logging.getLogger(__name__)


# ── RoutingDecision ───────────────────────────────────────────────────────────

@dataclass
class RoutingDecision:
    """Result of routing decision with model selection and metadata.

    Sprint 7 additions (all have defaults for backward compatibility):
        confidence_basis, evidence, escalated, original_confidence,
        escalation_reason, ab_variant, explanation
    """

    # ── Original fields (Sprint 1–6) ──────────────────────────────────────
    model: str
    provider: str
    tier: str
    confidence: float
    reasoning: List[str]
    estimated_cost: float
    fallback_models: List[str]
    classification: ClassificationResult

    # ── Sprint 7 additions ────────────────────────────────────────────────
    confidence_basis: str = CONFIDENCE_BASIS_RULE
    """Which mechanism produced the confidence score."""

    evidence: List[str] = field(default_factory=list)
    """Human-readable list of signals that drove this decision."""

    escalated: bool = False
    """True when low-confidence escalation changed the selected model."""

    original_confidence: Optional[float] = None
    """Pre-escalation confidence (set only when ``escalated=True``)."""

    escalation_reason: Optional[str] = None
    """Why escalation was triggered (set only when ``escalated=True``)."""

    ab_variant: Optional[str] = None
    """A/B variant label (``"a"`` or ``"b"``) when an A/B test is active."""

    explanation: str = ""
    """Human-readable explanation, populated by Router.explain()."""

    # ── Feature 2: streaming support ─────────────────────────────────────
    supports_streaming: bool = True
    """Whether the selected model supports streaming responses."""

    # ── Sprint 5: SLA fields ──────────────────────────────────────────────
    sla_compliant: bool = True
    """True when this routing decision met all configured SLAs."""

    sla_breaches: List[str] = field(default_factory=list)
    """List of SLA breach identifiers, e.g. ``["latency_exceeded"]``."""

    sla_adjustments: List[str] = field(default_factory=list)
    """Actions taken to correct SLA breaches, e.g. ``["escalated_model_due_to_quality_sla"]``."""

    # ── Convenience property ──────────────────────────────────────────────

    @property
    def selected_model(self) -> str:
        """Alias for ``model``, matches the Antaris event-schema naming."""
        return self.model

    # ── Serialisation ─────────────────────────────────────────────────────

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation (includes Sprint 7 fields)."""
        d: Dict[str, Any] = {
            "model": self.model,
            "provider": self.provider,
            "tier": self.tier,
            "confidence": self.confidence,
            "reasoning": self.reasoning,
            "estimated_cost": self.estimated_cost,
            "fallback_models": self.fallback_models,
            "classification": {
                "tier": self.classification.tier,
                "confidence": self.classification.confidence,
                "reasoning": self.classification.reasoning,
                "signals": self.classification.signals,
            },
            # Sprint 7
            "confidence_basis": self.confidence_basis,
            "evidence": self.evidence,
            "escalated": self.escalated,
            "ab_variant": self.ab_variant,
            # Feature 2
            "supports_streaming": self.supports_streaming,
        }
        if self.escalated:
            d["original_confidence"] = self.original_confidence
            d["escalation_reason"] = self.escalation_reason
        # Sprint 5
        d["sla_compliant"] = self.sla_compliant
        d["sla_breaches"] = self.sla_breaches
        d["sla_adjustments"] = self.sla_adjustments
        return d


# ── Router ────────────────────────────────────────────────────────────────────

class Router:
    """Main router interface for intelligent model selection.

    Sprint 7 constructor additions:
        low_confidence_threshold: float — escalate when confidence < this.
        escalation_model: str | None — model to escalate to.
        escalation_strategy: str — ``"always"`` | ``"log_only"`` | ``"ask"``.

    Sprint 5 constructor additions:
        sla: SLAConfig | None — cost-performance SLA configuration.
        fallback_chain: list[str] — ordered list of model names to try when
            the primary model is degraded or over-budget.
    """

    def __init__(
        self,
        config_path: str = None,
        enable_cost_tracking: bool = True,
        # ── Sprint 7 ──────────────────────────────────────────────────────
        low_confidence_threshold: float = 0.0,
        escalation_model: Optional[str] = None,
        escalation_strategy: str = ESCALATION_ALWAYS,
        # ── Sprint 5 ──────────────────────────────────────────────────────
        sla: Optional[SLAConfig] = None,
        fallback_chain: Optional[List[str]] = None,
        # ── Feature 4: unified classifier interface ────────────────────────
        classifier: Optional[Any] = None,
    ):
        """Initialize the router.

        Args:
            config_path: Path to configuration directory.
            enable_cost_tracking: Whether to enable cost tracking.
            low_confidence_threshold: Confidence below this triggers escalation
                (0.0 means never escalate, which is the backward-compatible default).
            escalation_model: Name of the model to escalate to when confidence
                is below the threshold.
            escalation_strategy: One of ``"always"``, ``"log_only"``, ``"ask"``.
            sla: Optional :class:`SLAConfig` with cost-performance SLAs.
            fallback_chain: Ordered list of model names to try when the primary
                model is degraded or over-budget during auto-scaling.
            classifier: Optional classifier instance to use instead of the
                default :class:`TaskClassifier`.  The object must implement
                ``.classify(prompt) -> result`` where ``result`` has ``.tier``
                and ``.confidence`` attributes.

                Example — using the semantic v2 classifier::

                    from antaris_router.semantic import SemanticClassifier
                    sem = SemanticClassifier(workspace="/tmp/myrouter")
                    router = Router(classifier=sem)
                    decision = router.route("Design a microservices platform")
        """
        if escalation_strategy not in VALID_ESCALATION_STRATEGIES:
            raise ValueError(
                f"escalation_strategy must be one of "
                f"{sorted(VALID_ESCALATION_STRATEGIES)}, "
                f"got {escalation_strategy!r}"
            )

        self.config = Config(config_path)
        # Use provided classifier or fall back to default keyword-based one
        self.classifier = classifier if classifier is not None else TaskClassifier(self.config)
        self.registry = ModelRegistry(self.config)
        self.cost_tracker = CostTracker() if enable_cost_tracking else None
        self.routing_history: List[RoutingDecision] = []

        # Sprint 7
        self.low_confidence_threshold = low_confidence_threshold
        self.escalation_model = escalation_model
        self.escalation_strategy = escalation_strategy
        self._health_tracker = ProviderHealthTracker()

        # Sprint 5
        self.sla = sla
        self.fallback_chain: List[str] = fallback_chain or []
        self._sla_monitor: Optional[SLAMonitor] = (
            SLAMonitor(sla) if sla is not None else None
        )

    # ── Public routing interface ──────────────────────────────────────────

    def route(
        self,
        prompt: str,
        context: Dict = None,
        prefer: str = None,
        min_tier: str = None,
        capability: str = None,
        estimate_tokens: tuple = (100, 50),
        # Sprint 7 additions
        ab_test: Optional[ABTest] = None,
        prefer_healthy: bool = False,
        # Sprint 5 additions
        auto_scale: bool = False,
    ) -> RoutingDecision:
        """Route a prompt to the most appropriate model.

        Args:
            prompt: The text prompt to route.
            context: Optional context dictionary for classification.
            prefer: Preferred model provider (e.g. ``"claude"``).
            min_tier: Minimum complexity tier to consider.
            capability: Required capability (e.g. ``"vision"``).
            estimate_tokens: ``(input_tokens, output_tokens)`` for cost estimation.
            ab_test: Active :class:`ABTest` instance, or *None*.
            prefer_healthy: When *True*, deprioritise models with degraded health.

        Returns:
            :class:`RoutingDecision` with selected model and metadata.
        """
        # ── Classify ──────────────────────────────────────────────────────
        classification = self._classify(prompt, context)

        # ── Tier selection ────────────────────────────────────────────────
        effective_tier = classification.tier
        if min_tier and self._tier_level(min_tier) > self._tier_level(classification.tier):
            effective_tier = min_tier

        # ── A/B variant assignment ────────────────────────────────────────
        ab_variant: Optional[str] = None
        if ab_test is not None:
            ab_variant = ab_test.assign_variant()
            # Strategy A → cost-optimised (default); strategy B → override prefer
            if ab_variant == "b" and ab_test.strategy_b == "quality_first":
                # Bump tier one level for quality-first variant
                effective_tier = self._bump_tier(effective_tier)

        # ── Model candidates ──────────────────────────────────────────────
        suitable_models = self.registry.models_for_tier(effective_tier)

        if capability:
            suitable_models = [m for m in suitable_models if m.has_capability(capability)]

        if prefer:
            preferred = [m for m in suitable_models if prefer.lower() in m.provider.lower()]
            if preferred:
                suitable_models = preferred

        # ── Health-aware filtering ────────────────────────────────────────
        if prefer_healthy and suitable_models:
            available = [
                m for m in suitable_models
                if self._health_tracker.is_available(m.name)
            ]
            if available:
                suitable_models = sorted(
                    available,
                    key=lambda m: (
                        -self._health_tracker.get_health_score(m.name),
                        m.calculate_cost(*estimate_tokens),
                    ),
                )
            # If all models are "down", fall through to normal ordering

        # ── Rate-limit filtering ──────────────────────────────────────────
        non_rate_limited = [
            m for m in suitable_models
            if not self._health_tracker.is_rate_limited(m.name)
        ]
        if non_rate_limited:
            suitable_models = non_rate_limited
        # else: all models are rate-limited; fall through to use full list

        if not suitable_models:
            fallback_decision = self._fallback_routing(
                prompt, classification, context, prefer, capability, estimate_tokens
            )
            if fallback_decision:
                return fallback_decision
            raise ValueError(
                f"No suitable models found for tier {effective_tier!r}"
                + (f" with capability {capability!r}" if capability else "")
                + (f" from provider {prefer!r}" if prefer else "")
            )

        selected_model = suitable_models[0]

        # ── Cost estimation ───────────────────────────────────────────────
        input_tokens, output_tokens = estimate_tokens
        estimated_cost = selected_model.calculate_cost(input_tokens, output_tokens)

        fallback_models = [m.name for m in suitable_models[1:4]]

        # ── Reasoning & evidence ──────────────────────────────────────────
        reasoning = self._generate_routing_reasoning(
            classification, selected_model, effective_tier, prefer, capability
        )
        confidence_basis = self._determine_confidence_basis(classification)
        evidence = self._generate_evidence(
            classification, selected_model, effective_tier, prompt
        )

        # ── Build decision ────────────────────────────────────────────────
        decision = RoutingDecision(
            model=selected_model.name,
            provider=selected_model.provider,
            tier=effective_tier,
            confidence=classification.confidence,
            reasoning=reasoning,
            estimated_cost=estimated_cost,
            fallback_models=fallback_models,
            classification=classification,
            confidence_basis=confidence_basis,
            evidence=evidence,
            ab_variant=ab_variant,
            supports_streaming=selected_model.supports_streaming,
        )

        # ── Confidence-gated escalation ───────────────────────────────────
        decision = self._maybe_escalate(decision, selected_model, estimate_tokens)

        # ── Sprint 5: SLA enforcement & auto-scaling ──────────────────────
        decision = self._apply_sla_and_autoscale(
            decision=decision,
            classification=classification,
            effective_tier=effective_tier,
            capability=capability,
            estimate_tokens=estimate_tokens,
            auto_scale=auto_scale,
        )

        # ── Explanation ───────────────────────────────────────────────────
        decision.explanation = self.explain(decision)

        # ── Track & return ────────────────────────────────────────────────
        self.routing_history.append(decision)
        return decision

    # ── Explainability ────────────────────────────────────────────────────

    def explain(self, result: RoutingDecision) -> str:
        """Generate a human-readable explanation of a routing decision.

        Args:
            result: The :class:`RoutingDecision` to explain.

        Returns:
            Multi-line explanation string.
        """
        conf_pct = int(round(result.confidence * 100))
        basis_label = result.confidence_basis.replace("_", " ")

        # ── Model line ────────────────────────────────────────────────────
        lines: List[str] = [
            f"Model selected: {result.model} (confidence: {conf_pct}%)"
        ]

        # ── Escalation note ───────────────────────────────────────────────
        if result.escalated:
            orig_pct = int(round((result.original_confidence or 0.0) * 100))
            lines.append(
                f"  [Escalated from original confidence {orig_pct}%: "
                f"{result.escalation_reason}]"
            )

        # ── Basis ─────────────────────────────────────────────────────────
        lines.append(f"Basis: {basis_label}")

        # ── Reasoning ─────────────────────────────────────────────────────
        lines.append("Reasoning:")
        tier = result.classification.tier
        cls_conf = int(round(result.classification.confidence * 100))
        lines.append(
            f"  Input classified as '{tier}' task ({cls_conf}% confidence)."
        )
        for r in result.classification.reasoning:
            lines.append(f"  {r}")

        # ── Cost ──────────────────────────────────────────────────────────
        model_info = self.registry.get_model(result.model)
        if model_info:
            avg_per_1k = (
                model_info.cost_per_1k_input + model_info.cost_per_1k_output
            ) / 2
            lines.append(
                f"Estimated cost: ${avg_per_1k:.4f} per 1K tokens "
                f"(this request: ${result.estimated_cost:.6f})."
            )

        # ── Evidence ─────────────────────────────────────────────────────
        if result.evidence:
            lines.append("Evidence:")
            for ev in result.evidence:
                lines.append(f"  • {ev}")

        # ── Alternatives ─────────────────────────────────────────────────
        if result.fallback_models:
            alt_parts: List[str] = []
            for fb_name in result.fallback_models[:3]:
                fb_info = self.registry.get_model(fb_name)
                if fb_info:
                    fb_avg = (
                        fb_info.cost_per_1k_input + fb_info.cost_per_1k_output
                    ) / 2
                    if model_info:
                        this_avg = (
                            model_info.cost_per_1k_input
                            + model_info.cost_per_1k_output
                        ) / 2
                        if fb_avg > this_avg:
                            factor = fb_avg / max(this_avg, 1e-9)
                            alt_parts.append(
                                f"{fb_name} (more capable, {factor:.1f}x cost)"
                            )
                        else:
                            alt_parts.append(f"{fb_name} (cheaper)")
                    else:
                        alt_parts.append(fb_name)
                else:
                    alt_parts.append(fb_name)
            if alt_parts:
                lines.append("Alternatives considered: " + ", ".join(alt_parts))

        return "\n".join(lines)

    # ── Provider health ───────────────────────────────────────────────────

    def record_provider_event(
        self,
        model: str,
        event: str,
        details: Optional[str] = None,
        latency_ms: Optional[float] = None,
    ) -> None:
        """Record a provider event for health tracking.

        Args:
            model: Model name.
            event: ``"success"``, ``"error"``, or ``"timeout"``.
            details: Optional detail string (e.g. ``"rate_limited"``).
            latency_ms: Response latency in milliseconds.
        """
        self._health_tracker.record_event(
            model=model,
            event=event,
            latency_ms=latency_ms,
            details=details,
        )

    def get_provider_health(self, model: str) -> Dict[str, Any]:
        """Return real-time health statistics for *model*.

        Returns a dict with keys:
            model, status, success_rate_1h, avg_latency_ms,
            recent_errors, last_seen
        """
        return self._health_tracker.get_health(model)

    # ── Cost forecasting ──────────────────────────────────────────────────

    def forecast_cost(
        self,
        requests_per_hour: float,
        avg_input_tokens: int,
        avg_output_tokens: int,
    ) -> Dict[str, Any]:
        """Forecast costs based on expected usage patterns.

        Uses the current routing-history distribution (if available) to
        estimate which models will be called and at what rates.

        Args:
            requests_per_hour: Expected request volume per hour.
            avg_input_tokens: Average input tokens per request.
            avg_output_tokens: Average output tokens per request.

        Returns:
            Dict with hourly/daily/monthly costs and a per-model breakdown.
        """
        # Build model usage distribution from routing history
        model_counts: Dict[str, int] = {}
        for d in self.routing_history:
            model_counts[d.model] = model_counts.get(d.model, 0) + 1

        total_hist = sum(model_counts.values())
        all_models = list(self.registry.list_models())

        if total_hist == 0 or not model_counts:
            # No history: distribute evenly across all known models
            if all_models:
                frac = 1.0 / len(all_models)
                model_fracs = {m.name: frac for m in all_models}
            else:
                return {
                    "hourly_cost_usd": 0.0,
                    "daily_cost_usd": 0.0,
                    "monthly_cost_usd": 0.0,
                    "breakdown_by_model": {},
                    "optimization_tip": "No routing history available; register models first.",
                }
        else:
            model_fracs = {
                name: count / total_hist
                for name, count in model_counts.items()
            }

        # Cost per request per model
        hourly_cost = 0.0
        breakdown: Dict[str, Dict[str, Any]] = {}

        for m_name, frac in model_fracs.items():
            m_info = self.registry.get_model(m_name)
            if not m_info:
                continue
            cost_per_req = m_info.calculate_cost(avg_input_tokens, avg_output_tokens)
            model_req_per_hour = requests_per_hour * frac
            model_hourly = cost_per_req * model_req_per_hour
            hourly_cost += model_hourly
            breakdown[m_name] = {
                "requests_pct": round(frac, 4),
                "cost_per_request_usd": round(cost_per_req, 6),
                "hourly_cost_usd": round(model_hourly, 4),
            }

        daily_cost = hourly_cost * 24
        monthly_cost = daily_cost * 30

        # Optimisation tip
        tip = self._generate_cost_tip(breakdown, daily_cost, avg_input_tokens, avg_output_tokens)

        return {
            "hourly_cost_usd": round(hourly_cost, 4),
            "daily_cost_usd": round(daily_cost, 4),
            "monthly_cost_usd": round(monthly_cost, 4),
            "breakdown_by_model": breakdown,
            "optimization_tip": tip,
        }

    def _generate_cost_tip(
        self,
        breakdown: Dict[str, Dict[str, Any]],
        daily_cost: float,
        avg_input_tokens: int,
        avg_output_tokens: int,
    ) -> str:
        """Generate a cost-optimisation tip from the breakdown."""
        if not breakdown:
            return "No data available for optimisation."

        # Find the most expensive model and the cheapest model
        sorted_by_cost = sorted(
            breakdown.items(), key=lambda kv: kv[1]["cost_per_request_usd"], reverse=True
        )
        most_expensive = sorted_by_cost[0][0]
        expensive_pct = breakdown[most_expensive]["requests_pct"]

        if len(sorted_by_cost) > 1:
            cheapest = sorted_by_cost[-1][0]
            cheap_info = self.registry.get_model(cheapest)
            exp_info = self.registry.get_model(most_expensive)
            if cheap_info and exp_info:
                cheap_per_req = cheap_info.calculate_cost(avg_input_tokens, avg_output_tokens)
                exp_per_req = exp_info.calculate_cost(avg_input_tokens, avg_output_tokens)
                # Estimate savings of routing 20% more to cheapest model
                shift_pct = min(0.20, expensive_pct)
                daily_req = breakdown[most_expensive]["hourly_cost_usd"] / max(exp_per_req, 1e-12) * 24
                savings = shift_pct * daily_req * (exp_per_req - cheap_per_req)
                if savings > 0.01:
                    return (
                        f"Routing {int(shift_pct * 100)}% more requests to {cheapest} "
                        f"instead of {most_expensive} would save ~${savings:.2f}/day."
                    )

        return f"Current mix looks reasonable; {most_expensive} handles {int(expensive_pct * 100)}% of traffic."

    # ── A/B testing ───────────────────────────────────────────────────────

    def create_ab_test(
        self,
        name: str,
        strategy_a: str,
        strategy_b: str,
        split: float = 0.5,
    ) -> ABTest:
        """Create and return a new :class:`ABTest`.

        Pass the returned object to :meth:`route` via ``ab_test=`` to activate it.

        Args:
            name: Human-readable identifier for the test.
            strategy_a: Strategy label for variant A.
            strategy_b: Strategy label for variant B.
            split: Fraction of traffic for variant A (0.0–1.0).

        Returns:
            A new :class:`ABTest` instance.
        """
        return ABTest(name=name, strategy_a=strategy_a, strategy_b=strategy_b, split=split)

    # ── Existing public methods (Sprint 1–6, unchanged) ───────────────────

    def log_usage(
        self, decision: RoutingDecision, input_tokens: int, output_tokens: int
    ) -> float:
        """Log actual usage for cost tracking."""
        if not self.cost_tracker:
            return 0.0
        model = self.registry.get_model(decision.model)
        if not model:
            return 0.0
        prompt_hash = hashlib.md5(
            f"{decision.tier}:{decision.confidence}".encode()
        ).hexdigest()[:12]
        return self.cost_tracker.log_usage(
            model=model,
            tier=decision.tier,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            prompt_hash=prompt_hash,
            confidence=decision.confidence,
        )

    def cost_report(self, period: str = "week") -> Dict[str, Any]:
        """Generate cost report for the specified period."""
        if not self.cost_tracker:
            return {"error": "Cost tracking is disabled"}
        return self.cost_tracker.report(period, self.registry)

    def savings_estimate(self, comparison_model: str = "gpt-4o") -> Dict[str, Any]:
        """Estimate cost savings compared to always using an expensive model."""
        if not self.cost_tracker:
            return {"error": "Cost tracking is disabled"}
        return self.cost_tracker.savings_estimate(comparison_model, self.registry)

    def routing_analytics(self) -> Dict[str, Any]:
        """Get analytics on routing decisions."""
        if not self.routing_history:
            return {"error": "No routing history available"}

        total_decisions = len(self.routing_history)
        tier_counts: Dict[str, int] = {}
        model_counts: Dict[str, int] = {}
        provider_counts: Dict[str, int] = {}
        avg_confidence = 0.0

        for decision in self.routing_history:
            tier_counts[decision.tier] = tier_counts.get(decision.tier, 0) + 1
            model_counts[decision.model] = model_counts.get(decision.model, 0) + 1
            provider_counts[decision.provider] = provider_counts.get(decision.provider, 0) + 1
            avg_confidence += decision.confidence

        avg_confidence /= total_decisions
        tier_percentages = {
            tier: count / total_decisions * 100
            for tier, count in tier_counts.items()
        }

        return {
            "total_decisions": total_decisions,
            "avg_confidence": round(avg_confidence, 3),
            "tier_distribution": tier_counts,
            "tier_percentages": {k: round(v, 1) for k, v in tier_percentages.items()},
            "model_usage": model_counts,
            "provider_usage": provider_counts,
            "most_used_model": max(model_counts, key=model_counts.get) if model_counts else None,
            "most_used_provider": max(provider_counts, key=provider_counts.get) if provider_counts else None,
        }

    def save_state(self, config_path: str) -> None:
        """Save router configuration state."""
        self.config.save_config(config_path)
        if self.cost_tracker:
            self.cost_tracker.save()

    def get_model_info(self, model_name: str) -> Optional[ModelInfo]:
        """Get information about a specific model."""
        return self.registry.get_model(model_name)

    def list_models_for_tier(self, tier: str) -> List[Dict[str, Any]]:
        """List all models suitable for a tier with their details."""
        models = self.registry.models_for_tier(tier)
        return [
            {
                "name": model.name,
                "provider": model.provider,
                "cost_per_1k_input": model.cost_per_1k_input,
                "cost_per_1k_output": model.cost_per_1k_output,
                "capabilities": model.capabilities,
                "max_tokens": model.max_tokens,
            }
            for model in models
        ]

    # ── Sprint 5: SLA public API ──────────────────────────────────────────

    def get_sla_report(self, since_hours: float = 1.0) -> Dict[str, Any]:
        """Generate an SLA compliance report for the specified look-back period.

        Args:
            since_hours: Number of hours to include in the report (default: 1).

        Returns:
            Dict with compliance_rate, breaches by type, adjustments_made,
            cost_savings_usd, avg_latency_ms, budget_utilization, and
            total_requests.  Returns a zeroed report when SLA is not configured.
        """
        if self._sla_monitor is None:
            return {
                "period_hours": since_hours,
                "compliance_rate": 1.0,
                "breaches": {"latency": 0, "cost": 0, "quality": 0},
                "adjustments_made": 0,
                "cost_savings_usd": 0.0,
                "avg_latency_ms": 0.0,
                "budget_utilization": 0.0,
                "total_requests": 0,
                "note": "No SLA configured.",
            }
        return self._sla_monitor.get_report(since_hours)

    def check_budget_alert(self) -> Dict[str, Any]:
        """Return current budget alert status.

        Returns:
            Dict with status (ok/warning/critical), hourly_spend_usd,
            budget_usd, utilization, projected_hourly_usd, and recommendation.
            Returns an ``"ok"`` status with a note when no SLA is configured.
        """
        if self._sla_monitor is None:
            return {
                "status": "ok",
                "hourly_spend_usd": None,
                "budget_usd": None,
                "utilization": None,
                "projected_hourly_usd": None,
                "recommendation": "No SLA configured.",
            }
        return self._sla_monitor.check_budget_alert()

    def get_cost_optimizations(
        self, estimate_tokens: tuple = (100, 50)
    ) -> List[Dict[str, Any]]:
        """Generate cost optimisation suggestions based on routing history.

        Analyses the recent routing distribution and suggests where cheaper
        models could be used without meaningful quality loss.

        Args:
            estimate_tokens: ``(input_tokens, output_tokens)`` used to estimate
                per-request savings.

        Returns:
            List of suggestion dicts with keys ``suggestion``,
            ``estimated_savings_usd_per_day``, and ``tradeoff``.
        """
        suggestions: List[Dict[str, Any]] = []
        if not self.routing_history:
            return suggestions

        input_tokens, output_tokens = estimate_tokens

        # Count how often each model/tier pair appears
        tier_model_counts: Dict[str, Dict[str, int]] = {}
        for d in self.routing_history:
            tier_model_counts.setdefault(d.tier, {})
            tier_model_counts[d.tier][d.model] = (
                tier_model_counts[d.tier].get(d.model, 0) + 1
            )

        total_decisions = len(self.routing_history)
        # Assume 1-hour of history → scale to 24h for daily estimate
        # Use count directly as "per hour" and multiply by 24
        for tier, model_counts in tier_model_counts.items():
            for model_name, count in model_counts.items():
                model_info = self.registry.get_model(model_name)
                if not model_info:
                    continue

                # Find the cheapest alternative for this tier
                tier_models = self.registry.models_for_tier(tier)
                cheaper = [
                    m for m in tier_models
                    if m.name != model_name
                    and m.calculate_cost(input_tokens, output_tokens)
                    < model_info.calculate_cost(input_tokens, output_tokens)
                ]
                if not cheaper:
                    continue

                cheapest_alt = cheaper[0]  # already sorted by cost
                current_cost = model_info.calculate_cost(input_tokens, output_tokens)
                alt_cost = cheapest_alt.calculate_cost(input_tokens, output_tokens)
                saving_per_req = current_cost - alt_cost

                if saving_per_req <= 0:
                    continue

                # Rough daily saving: count requests × 24 (project from history)
                daily_saving = saving_per_req * count * 24
                if daily_saving < 0.001:
                    continue

                # Quality tradeoff estimate
                # Tiers above simple → flag quality risk
                tradeoff_tiers = ("complex", "expert")
                if tier in tradeoff_tiers:
                    tradeoff = (
                        f"Routing complex/expert tasks to a cheaper model may "
                        f"reduce output quality for high-stakes workloads."
                    )
                else:
                    tradeoff = (
                        f"Minimal quality impact expected; both models handle "
                        f"'{tier}' tier tasks effectively."
                    )

                suggestions.append({
                    "suggestion": (
                        f"Route '{tier}' tasks currently using {model_name} "
                        f"to {cheapest_alt.name} instead"
                    ),
                    "estimated_savings_usd_per_day": round(daily_saving, 4),
                    "tradeoff": tradeoff,
                })

        # Sort by savings descending
        suggestions.sort(key=lambda s: s["estimated_savings_usd_per_day"], reverse=True)
        return suggestions

    def record_sla_quality(self, model: str, score: float) -> None:
        """Record a quality score for SLA tracking.

        Call this after using a model to feed quality data back into the
        SLA monitor so that quality SLA enforcement stays current.

        Args:
            model: Model name.
            score: Quality score 0.0–1.0.
        """
        if self._sla_monitor is not None:
            self._sla_monitor.record_quality(model, score)

    # ── Sprint 5: SLA private helpers ─────────────────────────────────────

    def _apply_sla_and_autoscale(
        self,
        decision: RoutingDecision,
        classification: ClassificationResult,
        effective_tier: str,
        capability: Optional[str],
        estimate_tokens: tuple,
        auto_scale: bool,
    ) -> RoutingDecision:
        """Apply SLA enforcement and auto-scaling to a routing decision.

        Mutates and returns ``decision``.  No-ops when neither SLA nor
        fallback_chain are configured and ``auto_scale`` is *False*.
        """
        input_tokens, output_tokens = estimate_tokens
        adjustments: List[str] = []

        # ── Step 1: Budget SLA → prefer cheaper model ─────────────────────
        if (
            self._sla_monitor is not None
            and self.sla.budget_per_hour_usd is not None
            and self.sla.auto_escalate_on_breach
        ):
            hourly_spend = self._sla_monitor._hourly_spend()
            if hourly_spend > self.sla.budget_per_hour_usd:
                cheaper = self._find_cheaper_model(
                    decision.model, effective_tier, capability, estimate_tokens
                )
                if cheaper:
                    decision = self._swap_model(
                        decision, cheaper, estimate_tokens,
                        reason="budget_sla"
                    )
                    adjustments.append("routed_to_cheaper_model_due_to_budget_sla")

        # ── Step 2: Quality SLA → escalate poor model ─────────────────────
        if (
            self._sla_monitor is not None
            and self.sla.min_quality_score is not None
            and self.sla.auto_escalate_on_breach
        ):
            avg_q = self._sla_monitor.get_avg_quality(decision.model)
            if avg_q is not None and avg_q < self.sla.min_quality_score:
                better = self._find_better_quality_model(
                    decision.model, effective_tier, capability, estimate_tokens
                )
                if better:
                    decision = self._swap_model(
                        decision, better, estimate_tokens,
                        reason="quality_sla"
                    )
                    adjustments.append("escalated_model_due_to_quality_sla")

        # ── Step 3: Latency SLA → prefer faster model ─────────────────────
        if (
            self._sla_monitor is not None
            and self.sla.max_latency_ms is not None
            and self.sla.auto_escalate_on_breach
        ):
            model_info = self.registry.get_model(decision.model)
            if model_info:
                avg_cost_1k = (
                    model_info.cost_per_1k_input + model_info.cost_per_1k_output
                ) / 2
                est_lat = self._sla_monitor.estimate_latency(
                    decision.model, effective_tier, avg_cost_1k
                )
                if est_lat > self.sla.max_latency_ms:
                    faster = self._find_faster_model(
                        decision.model, effective_tier, capability, estimate_tokens
                    )
                    if faster:
                        decision = self._swap_model(
                            decision, faster, estimate_tokens,
                            reason="latency_sla"
                        )
                        adjustments.append("preferred_faster_model_due_to_latency_sla")

        # ── Step 4: Auto-scale via fallback chain ─────────────────────────
        if auto_scale and self.fallback_chain:
            decision, scale_adj = self._apply_fallback_chain(
                decision, effective_tier, capability, estimate_tokens
            )
            adjustments.extend(scale_adj)

        # ── Step 5: SLA evaluation on final model ─────────────────────────
        if self._sla_monitor is not None:
            final_model_info = self.registry.get_model(decision.model)
            avg_cost_1k = 0.0
            if final_model_info:
                avg_cost_1k = (
                    final_model_info.cost_per_1k_input
                    + final_model_info.cost_per_1k_output
                ) / 2

            # Baseline: cost using the most expensive model
            baseline_cost = self._estimate_baseline_cost(
                effective_tier, capability, estimate_tokens
            )

            sla_record = self._sla_monitor.evaluate(
                model=decision.model,
                tier=effective_tier,
                estimated_cost_usd=decision.estimated_cost,
                avg_cost_per_1k=avg_cost_1k,
                baseline_cost_usd=baseline_cost,
            )

            # Merge monitor's adjustments with our pre-swap ones
            all_adjustments = list(dict.fromkeys(adjustments + sla_record.adjustments))
            decision.sla_compliant = sla_record.sla_compliant
            decision.sla_breaches = sla_record.breaches
            decision.sla_adjustments = all_adjustments
        elif adjustments:
            decision.sla_adjustments = adjustments

        return decision

    def _swap_model(
        self,
        decision: RoutingDecision,
        new_model: ModelInfo,
        estimate_tokens: tuple,
        reason: str,
    ) -> RoutingDecision:
        """Replace the model in *decision* with *new_model*.

        Updates model, provider, estimated_cost, reasoning, and evidence.
        """
        input_tokens, output_tokens = estimate_tokens
        decision.model = new_model.name
        decision.provider = new_model.provider
        decision.estimated_cost = new_model.calculate_cost(input_tokens, output_tokens)
        decision.reasoning = decision.reasoning + [
            f"SLA ({reason}): swapped to {new_model.name}"
        ]
        decision.evidence = decision.evidence + [
            f"sla_swap ({reason}): → {new_model.name}"
        ]
        return decision

    def _find_cheaper_model(
        self,
        current_model: str,
        tier: str,
        capability: Optional[str],
        estimate_tokens: tuple,
    ) -> Optional[ModelInfo]:
        """Return the cheapest model for *tier* that is cheaper than *current_model*."""
        current_info = self.registry.get_model(current_model)
        if not current_info:
            return None
        current_cost = current_info.calculate_cost(*estimate_tokens)

        candidates = self.registry.models_for_tier(tier)  # sorted cheapest first
        if capability:
            candidates = [m for m in candidates if m.has_capability(capability)]

        for m in candidates:
            if m.name != current_model and m.calculate_cost(*estimate_tokens) < current_cost:
                return m
        return None

    def _find_better_quality_model(
        self,
        current_model: str,
        tier: str,
        capability: Optional[str],
        estimate_tokens: tuple,
    ) -> Optional[ModelInfo]:
        """Return a higher-cost model for *tier* (escalate quality)."""
        current_info = self.registry.get_model(current_model)
        if not current_info:
            return None
        current_cost = current_info.calculate_cost(*estimate_tokens)

        # Check fallback chain first (respects user's preference ordering)
        if self.fallback_chain:
            for name in self.fallback_chain:
                if name == current_model:
                    continue
                m = self.registry.get_model(name)
                if m and m.calculate_cost(*estimate_tokens) > current_cost:
                    if not capability or m.has_capability(capability):
                        return m

        # Otherwise pick the most expensive model in the tier (highest quality)
        candidates = self.registry.models_for_tier(tier)
        if capability:
            candidates = [m for m in candidates if m.has_capability(capability)]

        # Sorted cheapest→expensive; pick most expensive != current
        for m in reversed(candidates):
            if m.name != current_model:
                return m
        return None

    def _find_faster_model(
        self,
        current_model: str,
        tier: str,
        capability: Optional[str],
        estimate_tokens: tuple,
    ) -> Optional[ModelInfo]:
        """Return a model with lower estimated latency than *current_model*."""
        current_info = self.registry.get_model(current_model)
        if not current_info:
            return None
        current_avg = (
            current_info.cost_per_1k_input + current_info.cost_per_1k_output
        ) / 2

        candidates = self.registry.models_for_tier(tier)  # sorted cheapest first
        if capability:
            candidates = [m for m in candidates if m.has_capability(capability)]

        # Lower cost → lower estimated latency in our heuristic
        for m in candidates:
            if m.name == current_model:
                continue
            m_avg = (m.cost_per_1k_input + m.cost_per_1k_output) / 2
            if m_avg < current_avg:
                return m
        return None

    def _apply_fallback_chain(
        self,
        decision: RoutingDecision,
        tier: str,
        capability: Optional[str],
        estimate_tokens: tuple,
    ) -> tuple:  # (RoutingDecision, List[str])
        """Try fallback chain when primary model is degraded or budget exhausted."""
        adjustments: List[str] = []

        # Determine if we need to switch
        is_degraded = not self._health_tracker.is_available(decision.model)
        budget_exhausted = (
            self._sla_monitor is not None
            and self.sla is not None
            and self.sla.budget_per_hour_usd is not None
            and self._sla_monitor._hourly_spend() > self.sla.budget_per_hour_usd
        )

        if not (is_degraded or budget_exhausted):
            return decision, adjustments

        # Try each model in the chain
        for fallback_name in self.fallback_chain:
            if fallback_name == decision.model:
                continue
            fallback_info = self.registry.get_model(fallback_name)
            if not fallback_info:
                continue
            if capability and not fallback_info.has_capability(capability):
                continue
            if not self._health_tracker.is_available(fallback_name):
                continue

            # If budget-driven: only accept cheaper models
            if budget_exhausted:
                if fallback_info.calculate_cost(*estimate_tokens) >= decision.estimated_cost:
                    continue

            reason = "degraded" if is_degraded else "budget_exhausted"
            decision = self._swap_model(
                decision, fallback_info, estimate_tokens, reason=f"autoscale_{reason}"
            )
            adjustments.append(f"fallback_chain_used_{reason}")
            break
        else:
            # All fallbacks failed or were too expensive → pick cheapest in registry
            if budget_exhausted:
                candidates = self.registry.models_for_tier(tier)
                if capability:
                    candidates = [m for m in candidates if m.has_capability(capability)]
                # Already sorted cheapest first
                for m in candidates:
                    if m.name != decision.model and self._health_tracker.is_available(m.name):
                        decision = self._swap_model(
                            decision, m, estimate_tokens,
                            reason="autoscale_cheapest_available"
                        )
                        adjustments.append("fallback_to_cheapest_available")
                        break

        return decision, adjustments

    def _estimate_baseline_cost(
        self,
        tier: str,
        capability: Optional[str],
        estimate_tokens: tuple,
    ) -> float:
        """Return the cost using the most expensive model for *tier* (for savings calc)."""
        candidates = self.registry.models_for_tier(tier)
        if capability:
            candidates = [m for m in candidates if m.has_capability(capability)]
        if not candidates:
            return 0.0
        # models_for_tier is cheapest-first; most expensive is last
        most_expensive = candidates[-1]
        return most_expensive.calculate_cost(*estimate_tokens)

    # ── Private helpers ───────────────────────────────────────────────────

    def _maybe_escalate(
        self,
        decision: RoutingDecision,
        selected_model: ModelInfo,
        estimate_tokens: tuple,
    ) -> RoutingDecision:
        """Apply confidence-gated escalation logic.

        Modifies and returns the decision (may swap the selected model).
        """
        threshold = self.low_confidence_threshold
        if threshold <= 0.0 or decision.confidence >= threshold:
            return decision
        if not self.escalation_model:
            return decision

        reason = (
            f"confidence {decision.confidence:.2f} < threshold {threshold:.2f}"
        )

        strategy = self.escalation_strategy

        if strategy == ESCALATION_LOG_ONLY:
            _log.info(
                "Escalation considered for %r (strategy=log_only): %s",
                decision.model,
                reason,
            )
            # Don't change the model
            return decision

        if strategy == ESCALATION_ASK:
            # Signal that clarification is needed; preserve original routing
            decision.escalated = True
            decision.original_confidence = decision.confidence
            decision.escalation_reason = f"ask: {reason}"
            # ab_variant left unchanged; selected_model unchanged
            return decision

        # strategy == ESCALATION_ALWAYS — swap model
        esc_model_info = self.registry.get_model(self.escalation_model)
        if not esc_model_info:
            _log.warning("Escalation model %r not found in registry", self.escalation_model)
            return decision

        input_tokens, output_tokens = estimate_tokens
        new_cost = esc_model_info.calculate_cost(input_tokens, output_tokens)

        original_conf = decision.confidence
        # Keep existing fields, update model-specific ones
        decision.original_confidence = original_conf
        decision.escalated = True
        decision.escalation_reason = reason
        decision.model = esc_model_info.name
        decision.provider = esc_model_info.provider
        decision.estimated_cost = new_cost
        decision.reasoning = decision.reasoning + [
            f"Escalated to {esc_model_info.name}: {reason}"
        ]
        # Recalculate evidence to reflect new model
        decision.evidence = decision.evidence + [
            f"escalation: {reason} → routed to {esc_model_info.name}"
        ]

        return decision

    def _determine_confidence_basis(self, classification: ClassificationResult) -> str:
        """Determine the confidence_basis string for a ClassificationResult.

        The main Router uses the keyword-based TaskClassifier, so the basis is
        ``"keyword_density"`` by default; when structural rules dominate the
        signals it falls back to ``"rule_based"``.
        """
        signals = classification.signals or {}
        kw_matches = signals.get("keyword_matches", {})
        total_kw = sum(kw_matches.values()) if kw_matches else 0

        if total_kw > 0:
            return CONFIDENCE_BASIS_KEYWORD
        return CONFIDENCE_BASIS_RULE

    def _generate_evidence(
        self,
        classification: ClassificationResult,
        selected_model: ModelInfo,
        effective_tier: str,
        prompt: str,
    ) -> List[str]:
        """Build a list of human-readable evidence strings for a decision."""
        evidence: List[str] = []
        signals = classification.signals or {}

        # Length
        length = signals.get("length", len(prompt))
        thresholds = self.config.get_length_thresholds()
        trivial_max = thresholds.get("trivial_max", 50)
        simple_max = thresholds.get("simple_max", 200)
        moderate_max = thresholds.get("moderate_max", 1_000)
        complex_max = thresholds.get("complex_max", 3_000)

        if length <= trivial_max:
            evidence.append(f"length: {length} chars → trivial range (≤{trivial_max})")
        elif length <= simple_max:
            evidence.append(f"length: {length} chars → simple range (≤{simple_max})")
        elif length <= moderate_max:
            evidence.append(f"length: {length} chars → moderate range (≤{moderate_max})")
        elif length <= complex_max:
            evidence.append(f"length: {length} chars → complex range (≤{complex_max})")
        else:
            evidence.append(f"length: {length} chars → expert range (>{complex_max})")

        # Keyword matches
        kw_matches = signals.get("keyword_matches", {})
        for tier, count in sorted(kw_matches.items()):
            if count > 0:
                evidence.append(f"keyword match: {count} '{tier}'-tier keyword(s) found")

        # Code signals
        if signals.get("has_code"):
            evidence.append("code detected: present (boosts complex/expert tier weight)")
        code_ind = signals.get("code_indicators", 0)
        if code_ind > 0:
            evidence.append(f"code indicators: {code_ind} programming pattern(s) matched")

        # Structural complexity
        struct = signals.get("structural_complexity", 0)
        if struct > 3:
            evidence.append(f"structural complexity: {struct} (high)")
        elif struct == 0:
            evidence.append("structural complexity: 0 (minimal)")

        # Selected model
        avg_cost = (
            selected_model.cost_per_1k_input + selected_model.cost_per_1k_output
        ) / 2
        evidence.append(
            f"model: {selected_model.name} selected as cheapest for "
            f"'{effective_tier}' tier (${avg_cost:.4f}/1K avg)"
        )

        return evidence

    def _fallback_routing(
        self,
        prompt: str,
        classification: ClassificationResult,
        context: Dict,
        prefer: str,
        capability: str,
        estimate_tokens: tuple,
    ) -> Optional[RoutingDecision]:
        """Attempt fallback routing to higher tiers if no models found."""
        tier_levels = ["trivial", "simple", "moderate", "complex", "expert"]
        current_tier_index = tier_levels.index(classification.tier)

        for tier in tier_levels[current_tier_index + 1:]:
            suitable_models = self.registry.models_for_tier(tier)
            if capability:
                suitable_models = [m for m in suitable_models if m.has_capability(capability)]
            if prefer:
                preferred = [m for m in suitable_models if prefer.lower() in m.provider.lower()]
                if preferred:
                    suitable_models = preferred
            if suitable_models:
                selected_model = suitable_models[0]
                input_tokens, output_tokens = estimate_tokens
                estimated_cost = selected_model.calculate_cost(input_tokens, output_tokens)
                reasoning = [
                    f"Fallback to '{tier}' tier — no models available for '{classification.tier}'",
                    f"Selected {selected_model.name} from {selected_model.provider}",
                    "This may be more expensive than optimal",
                ]
                evidence = [
                    f"fallback: escalated from '{classification.tier}' to '{tier}'",
                    f"model: {selected_model.name} (fallback selection)",
                ]
                return RoutingDecision(
                    model=selected_model.name,
                    provider=selected_model.provider,
                    tier=tier,
                    confidence=classification.confidence * 0.8,
                    reasoning=reasoning,
                    estimated_cost=estimated_cost,
                    fallback_models=[m.name for m in suitable_models[1:4]],
                    classification=classification,
                    confidence_basis=CONFIDENCE_BASIS_RULE,
                    evidence=evidence,
                )

        return None

    def _classify(self, prompt: str, context: Dict = None) -> "ClassificationResult":
        """Invoke the classifier, normalising the result to a :class:`ClassificationResult`.

        When a custom classifier is provided via the ``classifier`` constructor
        argument, it may return a different object (e.g. ``SemanticResult``).
        This helper ensures all downstream code can rely on the full
        :class:`ClassificationResult` interface (``tier``, ``confidence``,
        ``reasoning``, ``signals``).
        """
        # Try to pass context; custom classifiers may not accept it
        try:
            result = self.classifier.classify(prompt, context)
        except TypeError:
            result = self.classifier.classify(prompt)

        # Already a ClassificationResult → return as-is
        if isinstance(result, ClassificationResult):
            return result

        # Adapt: build a ClassificationResult from the minimal contract
        return ClassificationResult(
            tier=result.tier,
            confidence=result.confidence,
            reasoning=list(getattr(result, 'reasoning', [])),
            signals=dict(getattr(result, 'signals', {})),
        )

    def _tier_level(self, tier: str) -> int:
        """Return numeric level for tier comparison (0–4)."""
        tier_levels = ["trivial", "simple", "moderate", "complex", "expert"]
        try:
            return tier_levels.index(tier)
        except ValueError:
            return 2

    def _bump_tier(self, tier: str) -> str:
        """Return the next tier up, capped at 'expert'."""
        tiers = ["trivial", "simple", "moderate", "complex", "expert"]
        idx = self._tier_level(tier)
        return tiers[min(idx + 1, len(tiers) - 1)]

    def _generate_routing_reasoning(
        self,
        classification: ClassificationResult,
        selected_model: ModelInfo,
        effective_tier: str,
        prefer: str,
        capability: str,
    ) -> List[str]:
        """Generate human-readable reasoning for routing decision."""
        reasoning: List[str] = []

        reasoning.extend(classification.reasoning)

        if effective_tier != classification.tier:
            reasoning.append(
                f"Tier elevated from '{classification.tier}' to '{effective_tier}' "
                f"due to minimum requirement"
            )

        reasoning.append(f"Selected '{selected_model.name}' from {selected_model.provider}")
        reasoning.append(f"Cheapest model for '{effective_tier}' tier")

        if prefer:
            reasoning.append(f"Preferred provider: {prefer}")
        if capability:
            reasoning.append(f"Required capability: {capability}")

        avg_cost = (selected_model.cost_per_1k_input + selected_model.cost_per_1k_output) / 2
        if avg_cost == 0:
            reasoning.append("Local model — zero cost")
        elif avg_cost < 0.001:
            reasoning.append("Very low cost model")
        elif avg_cost < 0.01:
            reasoning.append("Low to moderate cost model")
        else:
            reasoning.append("Premium model — high capability")

        return reasoning
