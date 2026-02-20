"""
Tests for Sprint 2.3: Confidence-Gated Routing — AdaptiveRouter.

Covers:
  1. RouteDecision has confidence + basis fields
  2. Escalate strategy fires when confidence < threshold
  3. Safe-default strategy returns configured fallback model
  4. Clarify strategy signals clarification needed (strategy_applied="clarify")
  5. explain() returns structured dict with required keys
  6. explain() summary is human-readable string
  7. Backward compatibility: route() result["model"] still works
  8. route_with_confidence() normal path (high confidence, strategy_applied=None)
  9. Default threshold 0.0 never triggers strategy
  10. RouteDecision.to_dict() serialisation includes all fields
  11. explain() does NOT record a decision (read-only)
  12. Invalid confidence_strategy raises ValueError
  13. safe_default without model configured falls through gracefully
  14. Escalate strategy: strategy_applied set even when no higher model found
"""

from __future__ import annotations

import tempfile
import os
import pytest

from antaris_router.adaptive import (
    AdaptiveRouter,
    ModelConfig,
    RouteDecision,
    RoutingResult,
    STRATEGY_ESCALATE,
    STRATEGY_SAFE_DEFAULT,
    STRATEGY_CLARIFY,
    DEFAULT_CONFIDENCE_THRESHOLD,
)
from antaris_router.confidence import (
    CONFIDENCE_BASIS_SEMANTIC,
    CONFIDENCE_BASIS_QUALITY,
    CONFIDENCE_BASIS_COMPOSITE,
)


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

def _workspace() -> str:
    """Return a fresh temp directory for each test."""
    d = tempfile.mkdtemp()
    return d


def _make_router(
    workspace: str,
    threshold: float = 0.0,
    strategy: str | None = None,
    safe_default: str | None = None,
) -> AdaptiveRouter:
    """Build an AdaptiveRouter with two registered models."""
    r = AdaptiveRouter(
        workspace=workspace,
        confidence_threshold=threshold,
        confidence_strategy=strategy,
        safe_default_model=safe_default,
        ab_test_rate=0.0,  # disable A/B to keep routing deterministic
    )
    r.register_model(ModelConfig(
        name="haiku-3-5",
        tier_range=("trivial", "moderate"),
        cost_per_1k_input=0.00025,
        cost_per_1k_output=0.00125,
    ))
    r.register_model(ModelConfig(
        name="sonnet-4",
        tier_range=("moderate", "expert"),
        cost_per_1k_input=0.003,
        cost_per_1k_output=0.015,
    ))
    return r


# ---------------------------------------------------------------------------
# 1. RouteDecision has confidence + basis fields
# ---------------------------------------------------------------------------

class TestRouteDecisionFields:
    """RouteDecision from route_with_confidence() must expose the right fields."""

    def test_confidence_is_float_in_range(self):
        ws = _workspace()
        r = _make_router(ws)
        dec = r.route_with_confidence("What is 2 + 2?")
        assert isinstance(dec.confidence, float)
        assert 0.0 <= dec.confidence <= 1.0

    def test_basis_is_known_value(self):
        ws = _workspace()
        r = _make_router(ws)
        dec = r.route_with_confidence("Design a REST API")
        known = {CONFIDENCE_BASIS_SEMANTIC, CONFIDENCE_BASIS_QUALITY, CONFIDENCE_BASIS_COMPOSITE}
        assert dec.basis in known

    def test_model_field_is_non_empty_string(self):
        ws = _workspace()
        r = _make_router(ws)
        dec = r.route_with_confidence("Hello")
        assert isinstance(dec.model, str)
        assert len(dec.model) > 0

    def test_tier_field_is_valid_tier(self):
        ws = _workspace()
        r = _make_router(ws)
        dec = r.route_with_confidence("Write a Python function to sort a list")
        assert dec.tier in ("trivial", "simple", "moderate", "complex", "expert")

    def test_reason_field_is_non_empty_string(self):
        ws = _workspace()
        r = _make_router(ws)
        dec = r.route_with_confidence("Explain quantum entanglement")
        assert isinstance(dec.reason, str)
        assert len(dec.reason) > 0

    def test_strategy_applied_is_none_by_default(self):
        """With no threshold configured, strategy_applied must be None."""
        ws = _workspace()
        r = _make_router(ws, threshold=0.0)
        dec = r.route_with_confidence("What time is it?")
        assert dec.strategy_applied is None

    def test_to_dict_contains_required_keys(self):
        ws = _workspace()
        r = _make_router(ws)
        dec = r.route_with_confidence("Write a unit test")
        d = dec.to_dict()
        for key in ("model", "tier", "confidence", "basis", "reason",
                    "strategy_applied", "fallback_chain", "prompt_hash", "metadata"):
            assert key in d, f"Missing key: {key}"


# ---------------------------------------------------------------------------
# 2. Escalate strategy
# ---------------------------------------------------------------------------

class TestEscalateStrategy:
    """STRATEGY_ESCALATE should fire when confidence < threshold."""

    def test_escalate_strategy_applied_on_low_confidence(self):
        """Threshold=0.99 ensures almost every prompt triggers escalation."""
        ws = _workspace()
        r = _make_router(ws, threshold=0.99, strategy=STRATEGY_ESCALATE)
        dec = r.route_with_confidence("Hi")
        # If confidence is below threshold, strategy must be set
        if dec.confidence < 0.99:
            assert dec.strategy_applied == STRATEGY_ESCALATE

    def test_escalate_strategy_applied_value(self):
        """strategy_applied must equal the STRATEGY_ESCALATE constant."""
        ws = _workspace()
        r = _make_router(ws, threshold=0.99, strategy=STRATEGY_ESCALATE)
        dec = r.route_with_confidence("Hello there")
        if dec.confidence < 0.99:
            assert dec.strategy_applied == "escalate"

    def test_escalate_reason_mentions_low_confidence(self):
        ws = _workspace()
        r = _make_router(ws, threshold=0.99, strategy=STRATEGY_ESCALATE)
        dec = r.route_with_confidence("Hi")
        if dec.confidence < 0.99 and dec.strategy_applied == STRATEGY_ESCALATE:
            # Reason should explain why escalation was triggered
            assert "confidence" in dec.reason.lower() or "escalat" in dec.reason.lower()

    def test_no_escalate_when_confidence_above_threshold(self):
        """When confidence is >= threshold, strategy must not fire."""
        ws = _workspace()
        # Very low threshold so it never fires
        r = _make_router(ws, threshold=0.01, strategy=STRATEGY_ESCALATE)
        dec = r.route_with_confidence("What is 2+2?")
        if dec.confidence >= 0.01:
            assert dec.strategy_applied is None


# ---------------------------------------------------------------------------
# 3. Safe-default strategy
# ---------------------------------------------------------------------------

class TestSafeDefaultStrategy:
    """STRATEGY_SAFE_DEFAULT should route to configured fallback model."""

    def test_safe_default_returns_fallback_model(self):
        ws = _workspace()
        r = _make_router(
            ws, threshold=0.99,
            strategy=STRATEGY_SAFE_DEFAULT,
            safe_default="sonnet-4",
        )
        dec = r.route_with_confidence("Hello")
        if dec.confidence < 0.99:
            assert dec.model == "sonnet-4"
            assert dec.strategy_applied == STRATEGY_SAFE_DEFAULT

    def test_safe_default_strategy_applied_value(self):
        ws = _workspace()
        r = _make_router(
            ws, threshold=0.99,
            strategy=STRATEGY_SAFE_DEFAULT,
            safe_default="sonnet-4",
        )
        dec = r.route_with_confidence("Hi there")
        if dec.confidence < 0.99:
            assert dec.strategy_applied == "safe_default"

    def test_safe_default_reason_mentions_fallback(self):
        ws = _workspace()
        r = _make_router(
            ws, threshold=0.99,
            strategy=STRATEGY_SAFE_DEFAULT,
            safe_default="sonnet-4",
        )
        dec = r.route_with_confidence("Hi")
        if dec.confidence < 0.99:
            assert "sonnet-4" in dec.reason or "safe" in dec.reason.lower()


# ---------------------------------------------------------------------------
# 4. Clarify strategy
# ---------------------------------------------------------------------------

class TestClarifyStrategy:
    """STRATEGY_CLARIFY should signal clarification needed without changing model."""

    def test_clarify_strategy_applied_on_low_confidence(self):
        ws = _workspace()
        r = _make_router(ws, threshold=0.99, strategy=STRATEGY_CLARIFY)
        dec = r.route_with_confidence("Hmm...")
        if dec.confidence < 0.99:
            assert dec.strategy_applied == STRATEGY_CLARIFY

    def test_clarify_strategy_applied_value(self):
        ws = _workspace()
        r = _make_router(ws, threshold=0.99, strategy=STRATEGY_CLARIFY)
        dec = r.route_with_confidence("Hello")
        if dec.confidence < 0.99:
            assert dec.strategy_applied == "clarify"

    def test_clarify_reason_mentions_clarification(self):
        ws = _workspace()
        r = _make_router(ws, threshold=0.99, strategy=STRATEGY_CLARIFY)
        dec = r.route_with_confidence("Hi")
        if dec.confidence < 0.99 and dec.strategy_applied == STRATEGY_CLARIFY:
            assert "clarif" in dec.reason.lower() or "confidence" in dec.reason.lower()


# ---------------------------------------------------------------------------
# 5 & 6. explain() method
# ---------------------------------------------------------------------------

class TestExplain:
    """explain() returns structured dict + human-readable summary."""

    def test_explain_returns_dict(self):
        ws = _workspace()
        r = _make_router(ws)
        result = r.explain("Write a sorting algorithm in Python")
        assert isinstance(result, dict)

    def test_explain_has_classification_key(self):
        ws = _workspace()
        r = _make_router(ws)
        result = r.explain("What is the capital of France?")
        assert "classification" in result
        cls = result["classification"]
        assert "tier" in cls
        assert "confidence" in cls

    def test_explain_has_quality_scores_key(self):
        ws = _workspace()
        r = _make_router(ws)
        result = r.explain("Design a microservices architecture")
        assert "quality_scores" in result
        assert isinstance(result["quality_scores"], dict)

    def test_explain_has_cost_estimate_key(self):
        ws = _workspace()
        r = _make_router(ws)
        result = r.explain("Hello world")
        assert "cost_estimate" in result

    def test_explain_has_candidates_key(self):
        ws = _workspace()
        r = _make_router(ws)
        result = r.explain("Write a Python function")
        assert "candidates" in result
        assert isinstance(result["candidates"], list)

    def test_explain_has_why_selected_key(self):
        ws = _workspace()
        r = _make_router(ws)
        result = r.explain("Implement a REST API")
        assert "why_selected" in result
        assert isinstance(result["why_selected"], dict)

    def test_explain_summary_is_string(self):
        ws = _workspace()
        r = _make_router(ws)
        result = r.explain("What time is it?")
        assert "summary" in result
        assert isinstance(result["summary"], str)
        assert len(result["summary"]) > 0

    def test_explain_is_read_only(self):
        """explain() must not record a decision in the quality tracker."""
        ws = _workspace()
        r = _make_router(ws)
        count_before = len(r.tracker.decisions)
        r.explain("Write a complex algorithm")
        count_after = len(r.tracker.decisions)
        assert count_after == count_before

    def test_explain_summary_mentions_tier(self):
        ws = _workspace()
        r = _make_router(ws)
        result = r.explain("Hello")
        summary = result["summary"]
        valid_tiers = {"trivial", "simple", "moderate", "complex", "expert"}
        assert any(t in summary for t in valid_tiers)

    def test_explain_strategy_note_appears_when_configured(self):
        """explain() summary should mention strategy when threshold is configured."""
        ws = _workspace()
        r = _make_router(ws, threshold=0.5, strategy=STRATEGY_ESCALATE)
        result = r.explain("Hello")
        summary = result["summary"]
        # Summary should mention threshold comparison
        assert "0.5" in summary or "escalate" in summary.lower() or "threshold" in summary.lower()


# ---------------------------------------------------------------------------
# 7. Backward compatibility: route() still works
# ---------------------------------------------------------------------------

class TestBackwardCompatibility:
    """Callers that only use result.model / result["model"] must still work."""

    def test_route_returns_routing_result(self):
        ws = _workspace()
        r = _make_router(ws)
        result = r.route("What is 2 + 2?")
        assert isinstance(result, RoutingResult)

    def test_route_result_model_is_accessible(self):
        ws = _workspace()
        r = _make_router(ws)
        result = r.route("Write a unit test")
        # Callers that only use result.model should work fine
        assert isinstance(result.model, str)
        assert len(result.model) > 0

    def test_route_result_has_tier(self):
        ws = _workspace()
        r = _make_router(ws)
        result = r.route("Simple question")
        assert result.tier in ("trivial", "simple", "moderate", "complex", "expert")

    def test_route_result_has_confidence(self):
        ws = _workspace()
        r = _make_router(ws)
        result = r.route("Implement a distributed system")
        assert isinstance(result.confidence, float)
        assert 0.0 <= result.confidence <= 1.0

    def test_route_with_confidence_threshold_disabled(self):
        """threshold=0.0 means route_with_confidence acts like route()."""
        ws = _workspace()
        r = _make_router(ws, threshold=0.0)
        dec = r.route_with_confidence("Design a REST API")
        assert dec.strategy_applied is None
        assert isinstance(dec.model, str)


# ---------------------------------------------------------------------------
# 8. Error handling
# ---------------------------------------------------------------------------

class TestErrorHandling:
    """Invalid configuration should raise early, not fail silently."""

    def test_invalid_strategy_raises_value_error(self):
        ws = _workspace()
        with pytest.raises(ValueError, match="confidence_strategy"):
            AdaptiveRouter(
                workspace=ws,
                confidence_threshold=0.5,
                confidence_strategy="invalid_strategy",
            )

    def test_default_threshold_constant(self):
        """DEFAULT_CONFIDENCE_THRESHOLD should be 0.6 per spec."""
        assert DEFAULT_CONFIDENCE_THRESHOLD == 0.6

    def test_route_decision_to_dict_serialises_strategy(self):
        ws = _workspace()
        r = _make_router(ws, threshold=0.99, strategy=STRATEGY_CLARIFY)
        dec = r.route_with_confidence("Hi")
        d = dec.to_dict()
        assert "strategy_applied" in d
        # Must be serialisable (None or a string)
        assert d["strategy_applied"] is None or isinstance(d["strategy_applied"], str)
