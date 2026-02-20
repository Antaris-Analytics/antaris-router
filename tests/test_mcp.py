"""
Sprint 2.8: antaris-router MCP server tests.

Tests cover:
- create_server() raises ImportError when mcp not installed (mocked)
- create_server() returns FastMCP instance when mcp is available (mocked)
- route() tool returns expected keys
- route() tool handles different strategies
- explain() tool returns string explanation
- record_outcome() tool validates outcome values
- record_outcome() returns recorded=True for valid outcomes
- record_outcome() returns recorded=False for invalid outcome
- get_provider_health() tool returns health dict
- All tools are present in the server

Run with: pytest tests/test_mcp.py -v
"""

import sys
import pytest
from unittest.mock import MagicMock, patch


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_mock_decision(
    model="gpt-4o-mini",
    provider="openai",
    tier="simple",
    confidence=0.85,
    estimated_cost=0.00012,
):
    """Build a mock RoutingDecision."""
    decision = MagicMock()
    decision.model = model
    decision.provider = provider
    decision.tier = tier
    decision.confidence = confidence
    decision.estimated_cost = estimated_cost
    decision.fallback_models = ["gpt-4o", "claude-haiku-3-5"]
    decision.reasoning = ["Simple task", "Low cost model selected"]
    decision.explanation = f"Model selected: {model} (confidence: {int(confidence*100)}%)"
    decision.sla_compliant = True
    decision.sla_breaches = []
    decision.sla_adjustments = []
    return decision


# ── Module-level: server creation ─────────────────────────────────────────────

class TestCreateServer:
    """create_server() produces a server when mcp is available."""

    def test_create_server_raises_when_mcp_unavailable(self):
        """create_server() raises ImportError when MCP_AVAILABLE is False."""
        with patch("antaris_router.mcp_server.MCP_AVAILABLE", False):
            from antaris_router.mcp_server import create_server
            with pytest.raises(ImportError, match="mcp"):
                create_server()

    def test_create_server_with_mock_mcp(self):
        """create_server() returns a server object when mcp is mocked."""
        mock_mcp = MagicMock()
        mock_mcp.tool.return_value = lambda f: f  # pass-through decorator

        with patch("antaris_router.mcp_server.MCP_AVAILABLE", True), \
             patch("antaris_router.mcp_server.FastMCP", return_value=mock_mcp):
            from antaris_router.mcp_server import create_server
            server = create_server()
            assert server is mock_mcp

    def test_create_server_registers_four_tools(self):
        """create_server() calls mcp.tool() at least 4 times (one per tool)."""
        call_count = 0

        def counting_decorator(f):
            nonlocal call_count
            call_count += 1
            return f

        mock_mcp = MagicMock()
        mock_mcp.tool.side_effect = lambda: counting_decorator
        mock_mcp.name = "antaris-router"

        with patch("antaris_router.mcp_server.MCP_AVAILABLE", True), \
             patch("antaris_router.mcp_server.FastMCP", return_value=mock_mcp):
            from antaris_router.mcp_server import create_server
            server = create_server()

        # tool() should be called once per tool (route, explain, record_outcome, get_provider_health)
        assert mock_mcp.tool.call_count >= 4


# ── route() tool ──────────────────────────────────────────────────────────────

class TestRouteTool:
    """route() tool returns correct structure."""

    @pytest.fixture(autouse=True)
    def mock_router(self):
        """Patch Router.route() for all tests in this class."""
        self.decision = _make_mock_decision()
        with patch("antaris_router.mcp_server._get_router") as mock_get:
            mock_router = MagicMock()
            mock_router.route.return_value = self.decision
            mock_get.return_value = mock_router
            self.mock_router = mock_router
            yield

    def _get_route_fn(self):
        """Extract the route function by registering it with a real FastMCP mock."""
        registered = {}

        def capturing_decorator(f):
            registered[f.__name__] = f
            return f

        mock_mcp = MagicMock()
        mock_mcp.tool.return_value = capturing_decorator

        with patch("antaris_router.mcp_server.MCP_AVAILABLE", True), \
             patch("antaris_router.mcp_server.FastMCP", return_value=mock_mcp):
            from antaris_router.mcp_server import create_server
            create_server()

        return registered

    def test_route_returns_dict(self):
        """route() returns a dict."""
        fns = self._get_route_fn()
        result = fns["route"](request="Hello world")
        assert isinstance(result, dict)

    def test_route_has_required_keys(self):
        """route() result has all required keys."""
        fns = self._get_route_fn()
        result = fns["route"](request="Hello world")
        for key in ["model", "provider", "tier", "confidence", "estimated_cost",
                    "fallback_models", "reasoning"]:
            assert key in result, f"Missing key: {key}"

    def test_route_model_is_string(self):
        """route() result has string model."""
        fns = self._get_route_fn()
        result = fns["route"](request="Build a REST API")
        assert isinstance(result["model"], str)
        assert len(result["model"]) > 0

    def test_route_confidence_is_float(self):
        """route() result has float confidence."""
        fns = self._get_route_fn()
        result = fns["route"](request="Test confidence")
        assert isinstance(result["confidence"], float)
        assert 0.0 <= result["confidence"] <= 1.0

    def test_route_estimated_cost_is_float(self):
        """route() result has float estimated_cost."""
        fns = self._get_route_fn()
        result = fns["route"](request="Cost test")
        assert isinstance(result["estimated_cost"], float)
        assert result["estimated_cost"] >= 0.0

    def test_route_fallback_models_is_list(self):
        """route() result has fallback_models as list."""
        fns = self._get_route_fn()
        result = fns["route"](request="Fallback test")
        assert isinstance(result["fallback_models"], list)

    def test_route_strategy_none(self):
        """route() works with strategy=None (default)."""
        fns = self._get_route_fn()
        result = fns["route"](request="No strategy", strategy=None)
        assert isinstance(result, dict)

    def test_route_strategy_quality(self):
        """route() works with strategy='quality'."""
        fns = self._get_route_fn()
        result = fns["route"](request="Quality strategy", strategy="quality")
        assert isinstance(result, dict)

    def test_route_strategy_cost(self):
        """route() works with strategy='cost'."""
        fns = self._get_route_fn()
        result = fns["route"](request="Cost strategy", strategy="cost")
        assert isinstance(result, dict)


# ── explain() tool ────────────────────────────────────────────────────────────

class TestExplainTool:
    """explain() tool returns string explanation."""

    def _get_explain_fn(self):
        registered = {}

        def capturing_decorator(f):
            registered[f.__name__] = f
            return f

        mock_mcp = MagicMock()
        mock_mcp.tool.return_value = capturing_decorator

        decision = _make_mock_decision()
        mock_router = MagicMock()
        mock_router.route.return_value = decision
        mock_router.explain.return_value = "Model selected: gpt-4o-mini (confidence: 85%)"

        with patch("antaris_router.mcp_server.MCP_AVAILABLE", True), \
             patch("antaris_router.mcp_server.FastMCP", return_value=mock_mcp), \
             patch("antaris_router.mcp_server._get_router", return_value=mock_router):
            from antaris_router.mcp_server import create_server
            create_server()

        return registered

    def test_explain_returns_string(self):
        """explain() returns a string."""
        fns = self._get_explain_fn()
        result = fns["explain"](request="Explain this")
        assert isinstance(result, str)

    def test_explain_returns_non_empty(self):
        """explain() returns non-empty string."""
        fns = self._get_explain_fn()
        result = fns["explain"](request="Test explanation")
        assert len(result) > 0


# ── record_outcome() tool ─────────────────────────────────────────────────────

class TestRecordOutcomeTool:
    """record_outcome() validates outcomes and records events."""

    def _get_record_fn(self):
        registered = {}

        def capturing_decorator(f):
            registered[f.__name__] = f
            return f

        mock_mcp = MagicMock()
        mock_mcp.tool.return_value = capturing_decorator

        mock_router = MagicMock()

        with patch("antaris_router.mcp_server.MCP_AVAILABLE", True), \
             patch("antaris_router.mcp_server.FastMCP", return_value=mock_mcp), \
             patch("antaris_router.mcp_server._get_router", return_value=mock_router):
            from antaris_router.mcp_server import create_server
            create_server()

        return registered, mock_router

    def test_record_outcome_success(self):
        """record_outcome() records 'success' and returns recorded=True."""
        fns, _ = self._get_record_fn()
        result = fns["record_outcome"](model="gpt-4o-mini", outcome="success")
        assert result["recorded"] is True
        assert result["model"] == "gpt-4o-mini"

    def test_record_outcome_error(self):
        """record_outcome() records 'error' and returns recorded=True."""
        fns, _ = self._get_record_fn()
        result = fns["record_outcome"](model="claude-haiku-3-5", outcome="error")
        assert result["recorded"] is True

    def test_record_outcome_timeout(self):
        """record_outcome() records 'timeout' and returns recorded=True."""
        fns, _ = self._get_record_fn()
        result = fns["record_outcome"](model="gpt-4o", outcome="timeout")
        assert result["recorded"] is True

    def test_record_outcome_invalid(self):
        """record_outcome() rejects invalid outcome and returns recorded=False."""
        fns, _ = self._get_record_fn()
        result = fns["record_outcome"](model="gpt-4o-mini", outcome="invalid_outcome")
        assert result["recorded"] is False
        assert "error" in result

    def test_record_outcome_with_latency(self):
        """record_outcome() accepts optional latency_ms."""
        fns, _ = self._get_record_fn()
        result = fns["record_outcome"](
            model="gpt-4o-mini", outcome="success", latency_ms=250.0
        )
        assert result["recorded"] is True

    def test_record_outcome_calls_router(self):
        """record_outcome() calls router.record_provider_event()."""
        registered = {}

        def capturing_decorator(f):
            registered[f.__name__] = f
            return f

        mock_mcp = MagicMock()
        mock_mcp.tool.return_value = capturing_decorator

        mock_router_instance = MagicMock()

        with patch("antaris_router.mcp_server.MCP_AVAILABLE", True), \
             patch("antaris_router.mcp_server.FastMCP", return_value=mock_mcp), \
             patch("antaris_router.mcp_server._get_router", return_value=mock_router_instance):
            from antaris_router.mcp_server import create_server
            create_server()
            # Call the tool while the patch is still active
            result = registered["record_outcome"](
                model="test-model", outcome="success", latency_ms=100.0
            )

        assert result["recorded"] is True
        mock_router_instance.record_provider_event.assert_called_once()


# ── get_provider_health() tool ────────────────────────────────────────────────

class TestGetProviderHealthTool:
    """get_provider_health() returns health dict."""

    def _get_health_fn(self):
        registered = {}

        def capturing_decorator(f):
            registered[f.__name__] = f
            return f

        mock_mcp = MagicMock()
        mock_mcp.tool.return_value = capturing_decorator

        mock_router = MagicMock()
        mock_router.get_provider_health.return_value = {
            "model": "gpt-4o-mini",
            "status": "ok",
            "success_rate_1h": 0.99,
            "avg_latency_ms": 230.0,
            "recent_errors": 0,
        }

        with patch("antaris_router.mcp_server.MCP_AVAILABLE", True), \
             patch("antaris_router.mcp_server.FastMCP", return_value=mock_mcp), \
             patch("antaris_router.mcp_server._get_router", return_value=mock_router):
            from antaris_router.mcp_server import create_server
            create_server()

        return registered

    def test_get_provider_health_returns_dict(self):
        """get_provider_health() returns a dict."""
        fns = self._get_health_fn()
        result = fns["get_provider_health"](provider="gpt-4o-mini")
        assert isinstance(result, dict)

    def test_get_provider_health_has_status(self):
        """get_provider_health() result has 'status' key."""
        fns = self._get_health_fn()
        result = fns["get_provider_health"](provider="gpt-4o-mini")
        assert "status" in result

    def test_get_provider_health_model_key(self):
        """get_provider_health() result has 'model' key."""
        fns = self._get_health_fn()
        result = fns["get_provider_health"](provider="gpt-4o-mini")
        assert "model" in result

    def test_get_provider_health_unknown_provider(self):
        """get_provider_health() works for unknown provider (returns dict)."""
        registered = {}

        def capturing_decorator(f):
            registered[f.__name__] = f
            return f

        mock_mcp = MagicMock()
        mock_mcp.tool.return_value = capturing_decorator
        mock_router = MagicMock()
        mock_router.get_provider_health.return_value = {
            "model": "unknown-provider",
            "status": "unknown",
        }

        with patch("antaris_router.mcp_server.MCP_AVAILABLE", True), \
             patch("antaris_router.mcp_server.FastMCP", return_value=mock_mcp), \
             patch("antaris_router.mcp_server._get_router", return_value=mock_router):
            from antaris_router.mcp_server import create_server
            create_server()

        result = registered["get_provider_health"](provider="unknown-provider")
        assert isinstance(result, dict)
