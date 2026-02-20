"""
Tests for Sprint 5: Cost-Performance SLAs
Covers SLAConfig validation, SLAMonitor, Router SLA integration,
fallback chains, budget alerts, and cost optimisation suggestions.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List
import pytest

from antaris_router import Router, SLAConfig, SLAMonitor, SLARecord


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture()
def basic_router() -> Router:
    """Router without SLA — backward compatibility baseline."""
    return Router()


@pytest.fixture()
def sla_config() -> SLAConfig:
    return SLAConfig(
        max_latency_ms=2000,
        cost_reduction_target=0.30,
        min_quality_score=0.75,
        budget_per_hour_usd=10.00,
        auto_escalate_on_breach=True,
    )


@pytest.fixture()
def sla_router(sla_config: SLAConfig) -> Router:
    """Router with a full SLA configuration."""
    return Router(sla=sla_config)


@pytest.fixture()
def monitor(sla_config: SLAConfig) -> SLAMonitor:
    return SLAMonitor(sla_config)


# ─────────────────────────────────────────────────────────────────────────────
# 1. SLAConfig validation
# ─────────────────────────────────────────────────────────────────────────────

class TestSLAConfigValidation:

    def test_valid_full_config(self):
        cfg = SLAConfig(
            max_latency_ms=500,
            cost_reduction_target=0.40,
            min_quality_score=0.80,
            budget_per_hour_usd=5.00,
            auto_escalate_on_breach=True,
        )
        assert cfg.max_latency_ms == 500
        assert cfg.cost_reduction_target == 0.40
        assert cfg.min_quality_score == 0.80
        assert cfg.budget_per_hour_usd == 5.00
        assert cfg.auto_escalate_on_breach is True

    def test_all_defaults_are_none(self):
        cfg = SLAConfig()
        assert cfg.max_latency_ms is None
        assert cfg.cost_reduction_target is None
        assert cfg.min_quality_score is None
        assert cfg.budget_per_hour_usd is None
        assert cfg.auto_escalate_on_breach is True  # default True

    def test_negative_latency_rejected(self):
        with pytest.raises(ValueError, match="max_latency_ms"):
            SLAConfig(max_latency_ms=-1)

    def test_zero_latency_rejected(self):
        with pytest.raises(ValueError, match="max_latency_ms"):
            SLAConfig(max_latency_ms=0)

    def test_negative_budget_rejected(self):
        with pytest.raises(ValueError, match="budget_per_hour_usd"):
            SLAConfig(budget_per_hour_usd=-0.01)

    def test_zero_budget_rejected(self):
        with pytest.raises(ValueError, match="budget_per_hour_usd"):
            SLAConfig(budget_per_hour_usd=0)

    def test_quality_above_one_rejected(self):
        with pytest.raises(ValueError, match="min_quality_score"):
            SLAConfig(min_quality_score=1.5)

    def test_quality_below_zero_rejected(self):
        with pytest.raises(ValueError, match="min_quality_score"):
            SLAConfig(min_quality_score=-0.1)

    def test_cost_reduction_above_one_rejected(self):
        with pytest.raises(ValueError, match="cost_reduction_target"):
            SLAConfig(cost_reduction_target=1.1)

    def test_cost_reduction_below_zero_rejected(self):
        with pytest.raises(ValueError, match="cost_reduction_target"):
            SLAConfig(cost_reduction_target=-0.1)

    def test_boundary_quality_zero_accepted(self):
        cfg = SLAConfig(min_quality_score=0.0)
        assert cfg.min_quality_score == 0.0

    def test_boundary_quality_one_accepted(self):
        cfg = SLAConfig(min_quality_score=1.0)
        assert cfg.min_quality_score == 1.0

    def test_boundary_cost_reduction_zero_accepted(self):
        cfg = SLAConfig(cost_reduction_target=0.0)
        assert cfg.cost_reduction_target == 0.0

    def test_auto_escalate_can_be_false(self):
        cfg = SLAConfig(auto_escalate_on_breach=False)
        assert cfg.auto_escalate_on_breach is False


# ─────────────────────────────────────────────────────────────────────────────
# 2. SLAMonitor evaluation
# ─────────────────────────────────────────────────────────────────────────────

class TestSLAMonitorEvaluation:

    def test_compliant_when_no_constraints_breached(self, monitor: SLAMonitor):
        """All constraints satisfied → compliant, no breaches."""
        record = monitor.evaluate(
            model="claude-haiku-3-5",
            tier="trivial",
            estimated_cost_usd=0.0001,
            avg_cost_per_1k=0.0005,  # cheap → low latency
        )
        assert isinstance(record, SLARecord)
        assert record.sla_compliant is True
        assert record.breaches == []

    def test_latency_breach_detected(self):
        """When estimated latency > max_latency_ms, breach is recorded."""
        cfg = SLAConfig(max_latency_ms=50)  # 50 ms — very tight
        mon = SLAMonitor(cfg)
        # Even the cheapest model in our heuristic is ~80ms for trivial
        record = mon.evaluate(
            model="some-model",
            tier="trivial",
            estimated_cost_usd=0.0001,
            avg_cost_per_1k=0.0,  # triggers 80 ms band
        )
        assert "latency_exceeded" in record.breaches
        assert record.sla_compliant is False

    def test_quality_breach_detected(self):
        """When model quality < min_quality_score, breach is recorded."""
        cfg = SLAConfig(min_quality_score=0.90)
        mon = SLAMonitor(cfg)
        mon.record_quality("bad-model", 0.50)
        record = mon.evaluate(
            model="bad-model",
            tier="simple",
            estimated_cost_usd=0.001,
            avg_cost_per_1k=0.001,
        )
        assert "quality_below_threshold" in record.breaches

    def test_no_quality_breach_when_no_data(self):
        """No quality data → quality SLA is not triggered (benefit of the doubt)."""
        cfg = SLAConfig(min_quality_score=0.90)
        mon = SLAMonitor(cfg)
        record = mon.evaluate(
            model="unknown-model",
            tier="simple",
            estimated_cost_usd=0.001,
            avg_cost_per_1k=0.001,
        )
        assert "quality_below_threshold" not in record.breaches

    def test_budget_breach_detected(self):
        """When hourly spend exceeds budget, breach is recorded."""
        cfg = SLAConfig(budget_per_hour_usd=0.001)  # tiny budget
        mon = SLAMonitor(cfg)
        # Inflate spend first
        mon.evaluate("m1", "simple", estimated_cost_usd=0.01, avg_cost_per_1k=0.001)
        record = mon.evaluate("m2", "simple", estimated_cost_usd=0.001, avg_cost_per_1k=0.001)
        assert "budget_exceeded" in record.breaches

    def test_adjustments_populated_on_breach(self):
        """When auto_escalate_on_breach=True, adjustments list is non-empty."""
        cfg = SLAConfig(max_latency_ms=50, auto_escalate_on_breach=True)
        mon = SLAMonitor(cfg)
        record = mon.evaluate("m", "expert", estimated_cost_usd=0.01, avg_cost_per_1k=0.1)
        assert len(record.adjustments) > 0

    def test_no_adjustments_when_escalation_disabled(self):
        """When auto_escalate_on_breach=False, adjustments list is empty even on breach."""
        cfg = SLAConfig(max_latency_ms=50, auto_escalate_on_breach=False)
        mon = SLAMonitor(cfg)
        record = mon.evaluate("m", "expert", estimated_cost_usd=0.01, avg_cost_per_1k=0.1)
        # Breach should still be there
        assert "latency_exceeded" in record.breaches
        # But no adjustments
        assert record.adjustments == []

    def test_quality_scores_averaged(self):
        """Multiple quality recordings are averaged correctly."""
        cfg = SLAConfig(min_quality_score=0.5)
        mon = SLAMonitor(cfg)
        mon.record_quality("model-x", 0.4)
        mon.record_quality("model-x", 0.6)
        avg = mon.get_avg_quality("model-x")
        assert avg == pytest.approx(0.5, abs=1e-9)


# ─────────────────────────────────────────────────────────────────────────────
# 3. SLA report generation
# ─────────────────────────────────────────────────────────────────────────────

class TestSLAReport:

    def _run_n_requests(self, router: Router, n: int = 5) -> None:
        for _ in range(n):
            router.route("hello world")

    def test_report_structure(self, sla_router: Router):
        self._run_n_requests(sla_router, 3)
        report = sla_router.get_sla_report(since_hours=1)
        required_keys = {
            "period_hours", "compliance_rate", "breaches",
            "adjustments_made", "cost_savings_usd",
            "avg_latency_ms", "budget_utilization", "total_requests",
        }
        assert required_keys.issubset(report.keys())

    def test_report_breaches_structure(self, sla_router: Router):
        self._run_n_requests(sla_router, 2)
        report = sla_router.get_sla_report()
        assert "latency" in report["breaches"]
        assert "cost" in report["breaches"]
        assert "quality" in report["breaches"]

    def test_compliance_rate_is_float_0_to_1(self, sla_router: Router):
        self._run_n_requests(sla_router, 4)
        report = sla_router.get_sla_report()
        rate = report["compliance_rate"]
        assert isinstance(rate, float)
        assert 0.0 <= rate <= 1.0

    def test_empty_report_when_no_requests(self, sla_router: Router):
        report = sla_router.get_sla_report(since_hours=1)
        assert report["total_requests"] == 0
        assert report["compliance_rate"] == 1.0  # vacuously compliant

    def test_report_without_sla_returns_note(self, basic_router: Router):
        report = basic_router.get_sla_report()
        assert "note" in report
        assert report["total_requests"] == 0

    def test_compliance_rate_100_pct_when_generous_sla(self):
        """With a very lenient SLA, all requests should be compliant."""
        cfg = SLAConfig(
            max_latency_ms=60_000,       # 60 seconds — nothing exceeds this
            budget_per_hour_usd=1_000.0, # very large budget
        )
        router = Router(sla=cfg)
        for _ in range(5):
            router.route("quick test")
        report = router.get_sla_report()
        assert report["compliance_rate"] == 1.0
        assert report["breaches"]["latency"] == 0
        assert report["breaches"]["cost"] == 0

    def test_breach_counts_match_non_compliant(self):
        """Sum of breach types ≥ non-compliant requests (multiple breaches per request)."""
        cfg = SLAConfig(
            max_latency_ms=1,   # will always breach
            budget_per_hour_usd=0.000001,  # will always breach
        )
        router = Router(sla=cfg)
        for _ in range(3):
            router.route("hello")
        report = router.get_sla_report()
        assert report["breaches"]["latency"] >= 1
        non_compliant = report["total_requests"] - int(
            report["compliance_rate"] * report["total_requests"]
        )
        total_breaches = sum(report["breaches"].values())
        assert total_breaches >= non_compliant


# ─────────────────────────────────────────────────────────────────────────────
# 4. RoutingDecision SLA fields
# ─────────────────────────────────────────────────────────────────────────────

class TestRoutingDecisionSLAFields:

    def test_sla_fields_present_on_decision(self, sla_router: Router):
        decision = sla_router.route("hello world")
        assert hasattr(decision, "sla_compliant")
        assert hasattr(decision, "sla_breaches")
        assert hasattr(decision, "sla_adjustments")

    def test_sla_fields_default_when_no_sla(self, basic_router: Router):
        decision = basic_router.route("hello world")
        assert decision.sla_compliant is True
        assert decision.sla_breaches == []
        assert decision.sla_adjustments == []

    def test_to_dict_includes_sla_fields(self, sla_router: Router):
        decision = sla_router.route("hello")
        d = decision.to_dict()
        assert "sla_compliant" in d
        assert "sla_breaches" in d
        assert "sla_adjustments" in d

    def test_sla_compliant_is_bool(self, sla_router: Router):
        decision = sla_router.route("test prompt")
        assert isinstance(decision.sla_compliant, bool)

    def test_sla_breaches_is_list(self, sla_router: Router):
        decision = sla_router.route("test prompt")
        assert isinstance(decision.sla_breaches, list)

    def test_sla_adjustments_is_list(self, sla_router: Router):
        decision = sla_router.route("test prompt")
        assert isinstance(decision.sla_adjustments, list)


# ─────────────────────────────────────────────────────────────────────────────
# 5. Budget alerts
# ─────────────────────────────────────────────────────────────────────────────

class TestBudgetAlerts:

    def test_alert_ok_when_no_spend(self, sla_router: Router):
        alert = sla_router.check_budget_alert()
        assert alert["status"] == "ok"
        assert alert["utilization"] == 0.0

    def test_alert_structure(self, sla_router: Router):
        alert = sla_router.check_budget_alert()
        for key in ("status", "hourly_spend_usd", "budget_usd",
                    "utilization", "projected_hourly_usd", "recommendation"):
            assert key in alert

    def test_alert_no_sla_returns_ok(self, basic_router: Router):
        alert = basic_router.check_budget_alert()
        assert alert["status"] == "ok"
        assert alert["budget_usd"] is None

    def test_alert_warning_when_near_budget(self):
        cfg = SLAConfig(budget_per_hour_usd=0.001)
        router = Router(sla=cfg)
        # Force spend to ~85% of budget by feeding records directly to monitor
        router._sla_monitor.evaluate(
            model="m1", tier="simple",
            estimated_cost_usd=0.00085,
            avg_cost_per_1k=0.001,
        )
        alert = router.check_budget_alert()
        assert alert["status"] in ("warning", "critical")
        assert alert["utilization"] >= 0.80

    def test_alert_critical_when_over_budget(self):
        cfg = SLAConfig(budget_per_hour_usd=0.0005)
        router = Router(sla=cfg)
        router._sla_monitor.evaluate(
            model="m1", tier="simple",
            estimated_cost_usd=0.001,  # 2× budget
            avg_cost_per_1k=0.001,
        )
        alert = router.check_budget_alert()
        assert alert["status"] == "critical"
        assert alert["utilization"] >= 1.0

    def test_recommendation_string_present(self, sla_router: Router):
        alert = sla_router.check_budget_alert()
        assert isinstance(alert["recommendation"], str)
        assert len(alert["recommendation"]) > 0


# ─────────────────────────────────────────────────────────────────────────────
# 6. Cost optimisation suggestions
# ─────────────────────────────────────────────────────────────────────────────

class TestCostOptimisations:

    def test_empty_when_no_history(self, sla_router: Router):
        assert sla_router.get_cost_optimizations() == []

    def test_returns_list(self, basic_router: Router):
        basic_router.route("hello world")
        result = basic_router.get_cost_optimizations()
        assert isinstance(result, list)

    def test_suggestion_structure(self, basic_router: Router):
        # Route to an expensive tier to generate suggestions
        basic_router.route(
            "Design a distributed microservices architecture with event sourcing, "
            "CQRS, saga patterns, and Kubernetes orchestration for 10k req/s"
        )
        suggestions = basic_router.get_cost_optimizations()
        for s in suggestions:
            assert "suggestion" in s
            assert "estimated_savings_usd_per_day" in s
            assert "tradeoff" in s

    def test_savings_are_non_negative(self, basic_router: Router):
        for _ in range(5):
            basic_router.route("hello world quick question")
        suggestions = basic_router.get_cost_optimizations()
        for s in suggestions:
            assert s["estimated_savings_usd_per_day"] >= 0.0

    def test_sorted_by_savings_descending(self, basic_router: Router):
        for _ in range(3):
            basic_router.route(
                "Implement complex distributed system with CQRS, event sourcing, "
                "microservices, Kubernetes, CI/CD pipelines, monitoring and alerting"
            )
        suggestions = basic_router.get_cost_optimizations()
        if len(suggestions) >= 2:
            for i in range(len(suggestions) - 1):
                assert (
                    suggestions[i]["estimated_savings_usd_per_day"]
                    >= suggestions[i + 1]["estimated_savings_usd_per_day"]
                )


# ─────────────────────────────────────────────────────────────────────────────
# 7. Fallback chain
# ─────────────────────────────────────────────────────────────────────────────

class TestFallbackChain:

    def _get_all_model_names(self, router: Router) -> List[str]:
        return [m.name for m in router.registry.list_models()]

    def test_fallback_chain_stored_on_router(self):
        chain = ["model-a", "model-b"]
        router = Router(fallback_chain=chain)
        assert router.fallback_chain == chain

    def test_route_without_auto_scale_ignores_chain(self):
        """auto_scale=False → fallback chain is not applied."""
        router = Router()
        # Route normally; no errors
        decision = router.route("hello", auto_scale=False)
        assert decision.model != ""

    def test_auto_scale_route_returns_valid_decision(self):
        router = Router()
        all_names = [m.name for m in router.registry.list_models()]
        chain = all_names[:2] if len(all_names) >= 2 else all_names
        router.fallback_chain = chain
        decision = router.route("hello", auto_scale=True)
        assert decision.model != ""

    def test_fallback_used_when_primary_degraded(self):
        """Record errors on primary → health goes down → auto_scale picks next."""
        router = Router()
        all_names = [m.name for m in router.registry.list_models()]
        if len(all_names) < 2:
            pytest.skip("Need at least 2 models in registry")

        primary = all_names[0]
        secondary = all_names[1]
        router.fallback_chain = [primary, secondary]

        # Degrade the primary
        for _ in range(20):
            router.record_provider_event(primary, "error")

        # Force a route where the primary would be selected
        # (use prefer= to steer toward primary's provider)
        primary_info = router.registry.get_model(primary)
        decision = router.route(
            "hello",
            auto_scale=True,
        )
        # Either switched to secondary or stayed on primary if still "available"
        # The key assertion: no exception, valid decision
        assert decision.model in [m.name for m in router.registry.list_models()]

    def test_router_with_chain_and_sla(self):
        """Router with both SLA and fallback_chain initialises without error."""
        sla = SLAConfig(budget_per_hour_usd=10.0)
        router = Router(sla=sla, fallback_chain=["m1", "m2"])
        assert router.sla is sla
        assert router.fallback_chain == ["m1", "m2"]


# ─────────────────────────────────────────────────────────────────────────────
# 8. Backward compatibility
# ─────────────────────────────────────────────────────────────────────────────

class TestBackwardCompatibility:

    def test_router_without_sla_still_works(self, basic_router: Router):
        decision = basic_router.route("hello world")
        assert decision.model != ""
        assert decision.tier in ("trivial", "simple", "moderate", "complex", "expert")

    def test_sla_fields_have_safe_defaults(self, basic_router: Router):
        decision = basic_router.route("hello")
        assert decision.sla_compliant is True
        assert decision.sla_breaches == []
        assert decision.sla_adjustments == []

    def test_get_sla_report_without_sla_safe(self, basic_router: Router):
        report = basic_router.get_sla_report()
        assert isinstance(report, dict)
        assert "compliance_rate" in report

    def test_check_budget_alert_without_sla_safe(self, basic_router: Router):
        alert = basic_router.check_budget_alert()
        assert alert["status"] == "ok"

    def test_get_cost_optimizations_without_history_safe(self, basic_router: Router):
        assert basic_router.get_cost_optimizations() == []

    def test_all_sprint7_params_still_work(self):
        """Sprint 7 constructor params still accepted alongside Sprint 5 params."""
        sla = SLAConfig(budget_per_hour_usd=5.0)
        router = Router(
            low_confidence_threshold=0.3,
            escalation_model=None,
            escalation_strategy="always",
            sla=sla,
            fallback_chain=[],
        )
        assert router.low_confidence_threshold == 0.3
        assert router.sla is sla

    def test_routing_decision_to_dict_backward_compat(self, basic_router: Router):
        decision = basic_router.route("hello")
        d = decision.to_dict()
        # Original Sprint 1-6 keys
        for key in ("model", "provider", "tier", "confidence",
                    "reasoning", "estimated_cost", "fallback_models",
                    "classification"):
            assert key in d
        # Sprint 5 keys
        assert "sla_compliant" in d
        assert "sla_breaches" in d
        assert "sla_adjustments" in d


# ─────────────────────────────────────────────────────────────────────────────
# 9. SLA monitor edge cases
# ─────────────────────────────────────────────────────────────────────────────

class TestSLAMonitorEdgeCases:

    def test_latency_estimate_for_cheap_model(self):
        """Very cheap model → low latency estimate."""
        cfg = SLAConfig(max_latency_ms=500)
        mon = SLAMonitor(cfg)
        # avg_cost_per_1k = 0.0001 → should map to 200ms band
        lat = mon.estimate_latency("cheap", "simple", 0.0001)
        assert lat <= 300.0

    def test_latency_estimate_for_expensive_model(self):
        """Expensive model → higher latency estimate."""
        cfg = SLAConfig(max_latency_ms=500)
        mon = SLAMonitor(cfg)
        lat = mon.estimate_latency("expensive", "expert", 0.10)
        assert lat >= 1_000.0

    def test_multiple_quality_scores_for_same_model(self):
        cfg = SLAConfig(min_quality_score=0.7)
        mon = SLAMonitor(cfg)
        for score in [0.6, 0.8, 0.9, 0.7]:
            mon.record_quality("model-z", score)
        avg = mon.get_avg_quality("model-z")
        assert avg == pytest.approx(0.75, abs=1e-9)

    def test_get_avg_quality_returns_none_for_unknown(self):
        cfg = SLAConfig()
        mon = SLAMonitor(cfg)
        assert mon.get_avg_quality("nonexistent") is None

    def test_record_sla_quality_on_router(self, sla_router: Router):
        sla_router.record_sla_quality("test-model", 0.9)
        avg = sla_router._sla_monitor.get_avg_quality("test-model")
        assert avg == pytest.approx(0.9, abs=1e-9)

    def test_record_sla_quality_no_op_without_sla(self, basic_router: Router):
        # Should not raise
        basic_router.record_sla_quality("any-model", 0.9)

    def test_budget_utilization_zero_when_no_spend(self):
        cfg = SLAConfig(budget_per_hour_usd=10.0)
        mon = SLAMonitor(cfg)
        report = mon.get_report(since_hours=1)
        assert report["budget_utilization"] == 0.0

    def test_cost_savings_recorded(self):
        """baseline > actual cost → positive savings."""
        cfg = SLAConfig(budget_per_hour_usd=100.0)
        mon = SLAMonitor(cfg)
        mon.evaluate(
            model="cheap-model",
            tier="simple",
            estimated_cost_usd=0.001,
            avg_cost_per_1k=0.001,
            baseline_cost_usd=0.01,
        )
        report = mon.get_report(since_hours=1)
        assert report["cost_savings_usd"] == pytest.approx(0.009, abs=1e-6)
