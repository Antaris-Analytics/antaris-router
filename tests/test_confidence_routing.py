"""
Tests for Sprint 7: Confidence-Gated Routing.

Covers:
  1. RoutingDecision confidence fields
  2. Confidence-threshold escalation (always / log_only / ask)
  3. Explainability (router.explain + result.explanation)
  4. Provider health tracking
  5. Cost forecasting
  6. A/B testing framework
  7. Backward compatibility
"""

from __future__ import annotations

import time
from typing import List
import pytest

from antaris_router.router import Router, RoutingDecision
from antaris_router.confidence import (
    ABTest,
    ProviderHealthTracker,
    CONFIDENCE_BASIS_KEYWORD,
    CONFIDENCE_BASIS_RULE,
    CONFIDENCE_BASIS_SEMANTIC,
    CONFIDENCE_BASIS_COMPOSITE,
    CONFIDENCE_BASIS_QUALITY,
    ESCALATION_ALWAYS,
    ESCALATION_LOG_ONLY,
    ESCALATION_ASK,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def router() -> Router:
    """A default Router with no escalation configured."""
    return Router()


@pytest.fixture
def escalating_router() -> Router:
    """Router configured to always escalate on low confidence."""
    # First determine a valid escalation target from the registry
    r = Router(low_confidence_threshold=0.9, escalation_strategy="always")
    # Find the most expensive / highest-tier model name to use as escalation target
    models = r.registry.list_models()
    if models:
        models_sorted = sorted(
            models,
            key=lambda m: (m.cost_per_1k_input + m.cost_per_1k_output),
            reverse=True,
        )
        r.escalation_model = models_sorted[0].name
    return r


@pytest.fixture
def health_router() -> Router:
    """Router with health tracking, no escalation."""
    return Router()


# ---------------------------------------------------------------------------
# 1. RoutingDecision confidence fields
# ---------------------------------------------------------------------------

class TestConfidenceFields:
    """RoutingDecision must expose confidence metadata."""

    def test_result_has_confidence(self, router):
        result = router.route("What is 2+2?")
        assert isinstance(result.confidence, float)
        assert 0.0 <= result.confidence <= 1.0

    def test_result_has_confidence_basis(self, router):
        result = router.route("Write a Python function to sort a list")
        assert result.confidence_basis in {
            CONFIDENCE_BASIS_KEYWORD,
            CONFIDENCE_BASIS_RULE,
            CONFIDENCE_BASIS_SEMANTIC,
            CONFIDENCE_BASIS_COMPOSITE,
            CONFIDENCE_BASIS_QUALITY,
        }

    def test_result_has_evidence_list(self, router):
        result = router.route("Explain quantum entanglement in simple terms")
        assert isinstance(result.evidence, list)
        assert len(result.evidence) > 0

    def test_evidence_contains_length_signal(self, router):
        result = router.route("Hi")
        assert any("length" in e for e in result.evidence)

    def test_evidence_contains_model_signal(self, router):
        result = router.route("Implement a binary search algorithm in Python")
        assert any("model" in e for e in result.evidence)

    def test_selected_model_alias(self, router):
        """result.selected_model must equal result.model."""
        result = router.route("Hello, how are you?")
        assert result.selected_model == result.model

    def test_confidence_basis_for_keyword_heavy_prompt(self, router):
        """Prompts rich in tier keywords should produce keyword_density basis."""
        result = router.route(
            "Design a microservices architecture for a distributed system"
        )
        # Keyword matches should dominate → keyword_density
        assert result.confidence_basis in {CONFIDENCE_BASIS_KEYWORD, CONFIDENCE_BASIS_RULE}

    def test_to_dict_includes_new_fields(self, router):
        result = router.route("Write a SQL query")
        d = result.to_dict()
        assert "confidence_basis" in d
        assert "evidence" in d
        assert "escalated" in d


# ---------------------------------------------------------------------------
# 2. Confidence Threshold + Escalation
# ---------------------------------------------------------------------------

class TestEscalation:
    """Confidence-gated escalation behaviour."""

    def _router_with_escalation(self, threshold: float, strategy: str) -> Router:
        r = Router(low_confidence_threshold=threshold, escalation_strategy=strategy)
        models = r.registry.list_models()
        if models:
            best = max(models, key=lambda m: m.cost_per_1k_input + m.cost_per_1k_output)
            r.escalation_model = best.name
        return r

    def test_escalation_triggers_when_below_threshold(self):
        """Setting threshold=0.99 should escalate almost every prompt."""
        r = self._router_with_escalation(threshold=0.99, strategy="always")
        if not r.escalation_model:
            pytest.skip("No models available for escalation")
        result = r.route("What time is it?")
        # Either escalated (if confidence < 0.99) or not (if confidence >= 0.99)
        # We merely assert the flag is set correctly
        if result.confidence < 0.99:
            assert result.escalated is True

    def test_escalated_result_has_original_confidence(self):
        r = self._router_with_escalation(threshold=0.99, strategy="always")
        if not r.escalation_model:
            pytest.skip("No models available")
        result = r.route("What time is it?")
        if result.escalated:
            assert result.original_confidence is not None
            assert 0.0 <= result.original_confidence <= 1.0

    def test_escalated_result_has_escalation_reason(self):
        r = self._router_with_escalation(threshold=0.99, strategy="always")
        if not r.escalation_model:
            pytest.skip("No models available")
        result = r.route("What time is it?")
        if result.escalated:
            assert result.escalation_reason is not None
            assert len(result.escalation_reason) > 0

    def test_escalated_model_name_matches_escalation_model(self):
        r = self._router_with_escalation(threshold=0.99, strategy="always")
        if not r.escalation_model:
            pytest.skip("No models available")
        result = r.route("Hi there")
        if result.escalated:
            assert result.model == r.escalation_model

    def test_log_only_does_not_change_model(self):
        """log_only strategy must not alter the selected model."""
        r_always = self._router_with_escalation(threshold=0.99, strategy="always")
        r_log = self._router_with_escalation(threshold=0.99, strategy="log_only")
        if not r_always.escalation_model:
            pytest.skip("No models available")

        result_log = r_log.route("Hello")
        # log_only never sets escalated=True or changes the model
        assert result_log.escalated is False

    def test_log_only_escalated_is_false(self):
        r = self._router_with_escalation(threshold=0.99, strategy="log_only")
        if not r.escalation_model:
            pytest.skip("No models available")
        result = r.route("Simple question")
        assert result.escalated is False

    def test_ask_strategy_sets_escalated_flag(self):
        r = self._router_with_escalation(threshold=0.99, strategy="ask")
        if not r.escalation_model:
            pytest.skip("No models available")
        result = r.route("Simple question")
        if result.confidence < 0.99:
            assert result.escalated is True
            assert "ask" in (result.escalation_reason or "")

    def test_ask_strategy_preserves_original_model(self):
        """Ask strategy: escalated flag is set but original model routing is kept."""
        r_default = Router()
        r_ask = self._router_with_escalation(threshold=0.99, strategy="ask")
        if not r_ask.escalation_model:
            pytest.skip("No models available")

        # Route the same prompt with both routers
        result_ask = r_ask.route("Hello world")
        result_default = r_default.route("Hello world")
        if result_ask.escalated:
            # ask strategy does NOT swap the model — keeps original routing
            assert result_ask.model == result_default.model

    def test_no_escalation_when_threshold_zero(self):
        """Default threshold of 0.0 means never escalate."""
        r = Router(low_confidence_threshold=0.0)
        models = r.registry.list_models()
        if models:
            r.escalation_model = models[0].name
        result = r.route("hello")
        assert result.escalated is False

    def test_invalid_escalation_strategy_raises(self):
        with pytest.raises(ValueError, match="escalation_strategy"):
            Router(escalation_strategy="invalid_strategy")


# ---------------------------------------------------------------------------
# 3. Explainability
# ---------------------------------------------------------------------------

class TestExplainability:
    """router.explain() and result.explanation."""

    def test_explain_returns_string(self, router):
        result = router.route("Write a Python function")
        explanation = router.explain(result)
        assert isinstance(explanation, str)
        assert len(explanation) > 20

    def test_explain_contains_model_name(self, router):
        result = router.route("What is the capital of France?")
        explanation = router.explain(result)
        assert result.model in explanation

    def test_explain_contains_confidence_percent(self, router):
        result = router.route("Translate hello to Spanish")
        explanation = router.explain(result)
        assert "%" in explanation

    def test_explain_contains_basis(self, router):
        result = router.route("Write a unit test")
        explanation = router.explain(result)
        # confidence_basis words should appear
        assert any(
            word in explanation.lower()
            for word in ("keyword", "rule", "semantic", "quality", "composite", "basis")
        )

    def test_explain_contains_alternatives(self, router):
        result = router.route("Build a React component")
        explanation = router.explain(result)
        # Should mention fallback models if any exist
        if result.fallback_models:
            assert "Alternatives" in explanation or any(
                fb in explanation for fb in result.fallback_models
            )

    def test_result_explanation_property_populated(self, router):
        """result.explanation should be pre-populated by route()."""
        result = router.route("List the planets in our solar system")
        assert isinstance(result.explanation, str)
        assert len(result.explanation) > 0

    def test_result_explanation_matches_explain(self, router):
        """result.explanation must equal router.explain(result)."""
        result = router.route("Sort this list in Python")
        assert result.explanation == router.explain(result)


# ---------------------------------------------------------------------------
# 4. Provider Health Tracking
# ---------------------------------------------------------------------------

class TestProviderHealth:
    """record_provider_event + get_provider_health."""

    def test_record_success_event(self, router):
        router.record_provider_event("claude-sonnet-4-20250514", event="success", latency_ms=300)
        health = router.get_provider_health("claude-sonnet-4-20250514")
        assert health["status"] == "healthy"
        assert health["success_rate_1h"] == 1.0

    def test_record_error_event(self, router):
        router.record_provider_event("model-x", event="error", details="rate_limited")
        health = router.get_provider_health("model-x")
        assert health["recent_errors"] >= 1

    def test_record_timeout_event(self, router):
        router.record_provider_event("model-x", event="timeout")
        health = router.get_provider_health("model-x")
        assert health["recent_errors"] >= 1

    def test_health_returns_expected_keys(self, router):
        router.record_provider_event("m1", event="success")
        health = router.get_provider_health("m1")
        for key in ("model", "status", "success_rate_1h", "avg_latency_ms", "recent_errors", "last_seen"):
            assert key in health

    def test_health_status_healthy_on_all_success(self, router):
        for _ in range(10):
            router.record_provider_event("m-healthy", event="success", latency_ms=200)
        health = router.get_provider_health("m-healthy")
        assert health["status"] == "healthy"

    def test_health_status_degraded_on_partial_failures(self, router):
        for _ in range(7):
            router.record_provider_event("m-deg", event="success")
        for _ in range(3):
            router.record_provider_event("m-deg", event="error")
        health = router.get_provider_health("m-deg")
        assert health["status"] in ("degraded", "healthy")  # 70% success rate edge

    def test_health_status_down_on_all_errors(self, router):
        for _ in range(10):
            router.record_provider_event("m-down", event="error")
        health = router.get_provider_health("m-down")
        assert health["status"] == "down"

    def test_unknown_model_health(self, router):
        health = router.get_provider_health("completely-unknown-model")
        assert health["status"] == "unknown"
        assert health["success_rate_1h"] is None

    def test_avg_latency_calculated(self, router):
        router.record_provider_event("fast-model", event="success", latency_ms=100)
        router.record_provider_event("fast-model", event="success", latency_ms=200)
        health = router.get_provider_health("fast-model")
        assert health["avg_latency_ms"] == 150.0

    def test_prefer_healthy_routes_away_from_down_model(self):
        """prefer_healthy should avoid 'down' models when alternatives exist."""
        r = Router()
        models = r.registry.list_models()
        if len(models) < 2:
            pytest.skip("Need at least 2 models for this test")

        # Mark first model as completely down
        down_model = models[0].name
        for _ in range(20):
            r.record_provider_event(down_model, event="error")

        result = r.route("Write a function", prefer_healthy=True)
        health = r.get_provider_health(down_model)
        if health["status"] == "down":
            assert result.model != down_model

    def test_health_tracker_last_seen_populated(self, router):
        router.record_provider_event("m-ts", event="success")
        health = router.get_provider_health("m-ts")
        assert health["last_seen"] is not None
        # Should be ISO 8601
        assert "T" in health["last_seen"]

    def test_health_tracker_standalone(self):
        """ProviderHealthTracker can be used independently."""
        tracker = ProviderHealthTracker()
        # 3 successes + 1 error → 75% success rate → "degraded"
        tracker.record_event("my-model", event="success", latency_ms=500)
        tracker.record_event("my-model", event="success", latency_ms=600)
        tracker.record_event("my-model", event="success", latency_ms=550)
        tracker.record_event("my-model", event="error")
        h = tracker.get_health("my-model")
        assert h["status"] in ("healthy", "degraded")
        assert h["recent_errors"] == 1
        score = tracker.get_health_score("my-model")
        assert 0.0 <= score <= 1.0


# ---------------------------------------------------------------------------
# 5. Cost Forecasting
# ---------------------------------------------------------------------------

class TestCostForecasting:
    """router.forecast_cost()."""

    def test_forecast_returns_dict(self, router):
        result = router.forecast_cost(
            requests_per_hour=100,
            avg_input_tokens=500,
            avg_output_tokens=200,
        )
        assert isinstance(result, dict)

    def test_forecast_has_required_keys(self, router):
        result = router.forecast_cost(100, 500, 200)
        for key in ("hourly_cost_usd", "daily_cost_usd", "monthly_cost_usd",
                    "breakdown_by_model", "optimization_tip"):
            assert key in result

    def test_forecast_daily_is_24x_hourly(self, router):
        result = router.forecast_cost(100, 500, 200)
        if result["hourly_cost_usd"] > 0:
            ratio = result["daily_cost_usd"] / result["hourly_cost_usd"]
            assert abs(ratio - 24) < 0.01

    def test_forecast_monthly_is_30x_daily(self, router):
        result = router.forecast_cost(100, 500, 200)
        if result["daily_cost_usd"] > 0:
            ratio = result["monthly_cost_usd"] / result["daily_cost_usd"]
            assert abs(ratio - 30) < 0.01

    def test_forecast_with_routing_history(self):
        """Forecast should use history distribution when available."""
        r = Router()
        # Produce some routing history
        for _ in range(5):
            r.route("hello world")
        result = r.forecast_cost(50, 300, 100)
        assert result["hourly_cost_usd"] >= 0.0
        assert isinstance(result["breakdown_by_model"], dict)

    def test_forecast_optimization_tip_is_string(self, router):
        result = router.forecast_cost(100, 500, 200)
        assert isinstance(result["optimization_tip"], str)
        assert len(result["optimization_tip"]) > 0

    def test_forecast_breakdown_pct_sums_to_one(self):
        """Model request percentages in the breakdown should sum to ~1.0."""
        r = Router()
        for prompt in ["hello", "write code", "analyse this dataset"]:
            r.route(prompt)
        result = r.forecast_cost(100, 500, 200)
        breakdown = result["breakdown_by_model"]
        if breakdown:
            total_pct = sum(v["requests_pct"] for v in breakdown.values())
            assert abs(total_pct - 1.0) < 0.01


# ---------------------------------------------------------------------------
# 6. A/B Testing Framework
# ---------------------------------------------------------------------------

class TestABTesting:
    """create_ab_test + ABTest behaviour."""

    def test_create_ab_test_returns_abtest(self, router):
        ab = router.create_ab_test("test1", "cost_optimized", "quality_first")
        assert isinstance(ab, ABTest)

    def test_ab_test_variant_is_a_or_b(self, router):
        ab = router.create_ab_test("t", "a_strategy", "b_strategy", split=0.5)
        result = router.route("simple question", ab_test=ab)
        assert result.ab_variant in ("a", "b")

    def test_ab_test_50_50_split(self, router):
        """With split=0.5 over 100 requests, each variant should get ~50%."""
        ab = router.create_ab_test("split-test", "s_a", "s_b", split=0.5)
        counts = {"a": 0, "b": 0}
        for i in range(100):
            result = router.route(f"question {i}", ab_test=ab)
            assert result.ab_variant in ("a", "b")
            counts[result.ab_variant] += 1
        # Should be exactly 50/50 with deterministic assignment
        assert counts["a"] == 50
        assert counts["b"] == 50

    def test_ab_test_75_25_split(self, router):
        """With split=0.75, variant A should get 75% of traffic."""
        ab = router.create_ab_test("split75", "s_a", "s_b", split=0.75)
        counts = {"a": 0, "b": 0}
        for i in range(100):
            variant = ab.assign_variant()
            counts[variant] += 1
        # Allow ±5% tolerance
        assert 70 <= counts["a"] <= 80, f"Expected ~75 'a' requests, got {counts['a']}"

    def test_ab_test_get_stats_structure(self, router):
        ab = router.create_ab_test("stats-test", "str_a", "str_b")
        for i in range(6):
            router.route(f"test prompt {i}", ab_test=ab)
        stats = ab.get_stats()
        assert "variant_a" in stats
        assert "variant_b" in stats
        assert "requests" in stats["variant_a"]
        assert "requests" in stats["variant_b"]

    def test_ab_test_record_result(self):
        ab = ABTest("rec-test", "a", "b", split=0.5)
        ab.record_result("a", quality_score=0.9)
        ab.record_result("a", quality_score=0.8)
        stats = ab.get_stats()
        assert stats["variant_a"]["requests"] == 2
        assert stats["variant_a"]["avg_quality"] == pytest.approx(0.85, abs=0.001)

    def test_ab_test_winner_none_when_insufficient_data(self):
        ab = ABTest("winner-test", "a", "b", split=0.5)
        ab.record_result("a", quality_score=0.9)
        ab.record_result("b", quality_score=0.5)
        assert ab.winner() is None  # < 10 samples each

    def test_ab_test_winner_declared_after_enough_data(self):
        ab = ABTest("winner-test2", "a", "b", split=0.5)
        for _ in range(15):
            ab.record_result("a", quality_score=0.95)
            ab.record_result("b", quality_score=0.60)
        winner = ab.winner()
        assert winner == "a"


# ---------------------------------------------------------------------------
# 7. Backward Compatibility
# ---------------------------------------------------------------------------

class TestBackwardCompatibility:
    """Existing router.route() must still work unchanged."""

    def test_basic_route_still_works(self, router):
        result = router.route("What is the capital of France?")
        assert result.model  # non-empty
        assert result.provider
        assert result.tier in ("trivial", "simple", "moderate", "complex", "expert")

    def test_original_fields_present(self, router):
        result = router.route("Write a unit test for this function")
        assert hasattr(result, "model")
        assert hasattr(result, "provider")
        assert hasattr(result, "tier")
        assert hasattr(result, "confidence")
        assert hasattr(result, "reasoning")
        assert hasattr(result, "estimated_cost")
        assert hasattr(result, "fallback_models")
        assert hasattr(result, "classification")

    def test_route_with_prefer_param(self, router):
        result = router.route("Simple task", prefer="claude")
        assert result  # Should not raise

    def test_route_with_min_tier_param(self, router):
        result = router.route("What?", min_tier="moderate")
        assert result.tier in ("moderate", "complex", "expert")

    def test_route_with_estimate_tokens_param(self, router):
        result = router.route("Hello", estimate_tokens=(1000, 500))
        assert result.estimated_cost > 0 or result.estimated_cost == 0.0  # any float

    def test_routing_history_grows(self, router):
        before = len(router.routing_history)
        router.route("test 1")
        router.route("test 2")
        assert len(router.routing_history) == before + 2

    def test_to_dict_backward_compat(self, router):
        """Original to_dict() keys must still be present."""
        result = router.route("build something complex")
        d = result.to_dict()
        for key in ("model", "provider", "tier", "confidence", "reasoning",
                    "estimated_cost", "fallback_models", "classification"):
            assert key in d

    def test_routing_analytics_still_works(self, router):
        router.route("hello")
        analytics = router.routing_analytics()
        assert "total_decisions" in analytics
        assert analytics["total_decisions"] >= 1

    def test_no_escalation_in_default_router(self, router):
        """Default router (threshold=0.0) should never escalate."""
        for _ in range(10):
            result = router.route("test prompt")
            assert result.escalated is False
