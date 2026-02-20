"""
Antaris Router MCP Server

Exposes antaris-router as MCP tools for any MCP-enabled agent.

Tools:
  - route(request, strategy?) → model selection decision
  - explain(request)          → human-readable routing explanation
  - record_outcome(model, outcome, latency_ms?) → feed back real performance
  - get_provider_health(provider) → health stats for a specific provider

Usage:
    python -m antaris_router.mcp_server
    # or
    from antaris_router.mcp_server import create_server
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Any, Dict, Optional

# ---------------------------------------------------------------------------
# Graceful MCP availability check
# ---------------------------------------------------------------------------
try:
    from mcp.server.fastmcp import FastMCP
    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False
    FastMCP = None  # type: ignore

from antaris_router import Router


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_router() -> Router:
    """Create a Router instance (stateless; creates fresh per call)."""
    return Router(enable_cost_tracking=True)


# ---------------------------------------------------------------------------
# Server factory
# ---------------------------------------------------------------------------

def create_server() -> "FastMCP":
    """Create and return the FastMCP server with antaris-router tools.

    Returns:
        A configured ``FastMCP`` instance ready to run.

    Raises:
        ImportError: If the ``mcp`` package is not installed.
    """
    if not MCP_AVAILABLE:
        raise ImportError(
            "The 'mcp' package is required to run the antaris-router MCP server. "
            "Install it with: pip install mcp"
        )

    mcp = FastMCP(
        name="antaris-router",
        instructions=(
            "Antaris Router — smart model routing. "
            "Use route() to get a model recommendation, explain() for reasoning, "
            "record_outcome() to feed performance back, and get_provider_health() "
            "to check provider status."
        ),
    )

    # ------------------------------------------------------------------
    # Tool: route
    # ------------------------------------------------------------------
    @mcp.tool()
    def route(
        request: str,
        strategy: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Route a request to the most appropriate model.

        Args:
            request: The text of the request to route.
            strategy: Optional routing hint. Supported values:
                ``"cost"`` (prefer cheapest), ``"quality"`` (prefer best),
                ``"balanced"`` (default, cost-performance balance).

        Returns:
            Dict with keys: model, provider, tier, confidence, estimated_cost,
            fallback_models, reasoning, explanation.
        """
        router = _get_router()

        # Map strategy hint to prefer_healthy / min_tier
        prefer_healthy = strategy in ("quality", "balanced", None)
        min_tier = None
        if strategy == "quality":
            min_tier = "moderate"

        decision = router.route(
            prompt=request,
            min_tier=min_tier,
            prefer_healthy=prefer_healthy,
        )
        return {
            "model": decision.model,
            "provider": decision.provider,
            "tier": decision.tier,
            "confidence": round(decision.confidence, 4),
            "estimated_cost": round(decision.estimated_cost, 6),
            "fallback_models": decision.fallback_models[:3],
            "reasoning": decision.reasoning[:5],
            "explanation": decision.explanation,
            "sla_compliant": decision.sla_compliant,
        }

    # ------------------------------------------------------------------
    # Tool: explain
    # ------------------------------------------------------------------
    @mcp.tool()
    def explain(request: str) -> str:
        """Generate a human-readable explanation of how this request would be routed.

        Args:
            request: The text of the request to analyse.

        Returns:
            Multi-line explanation string with model selection rationale,
            classification tier, cost breakdown, and alternatives.
        """
        router = _get_router()
        decision = router.route(prompt=request)
        return router.explain(decision)

    # ------------------------------------------------------------------
    # Tool: record_outcome
    # ------------------------------------------------------------------
    @mcp.tool()
    def record_outcome(
        model: str,
        outcome: str,
        latency_ms: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Record a real-world outcome for a model to improve future routing.

        Args:
            model: The model name (e.g. ``"claude-haiku-3-5"``).
            outcome: ``"success"``, ``"error"``, or ``"timeout"``.
            latency_ms: Optional observed latency in milliseconds.

        Returns:
            Dict with keys: model, outcome, recorded (bool).
        """
        valid_outcomes = {"success", "error", "timeout"}
        if outcome not in valid_outcomes:
            return {
                "model": model,
                "outcome": outcome,
                "recorded": False,
                "error": f"outcome must be one of {sorted(valid_outcomes)}",
            }

        router = _get_router()
        router.record_provider_event(
            model=model,
            event=outcome,
            latency_ms=latency_ms,
        )
        return {"model": model, "outcome": outcome, "recorded": True}

    # ------------------------------------------------------------------
    # Tool: get_provider_health
    # ------------------------------------------------------------------
    @mcp.tool()
    def get_provider_health(provider: str) -> Dict[str, Any]:
        """Return real-time health statistics for a provider or model.

        Args:
            provider: Provider or model name (e.g. ``"claude-haiku-3-5"``).

        Returns:
            Dict with keys: model, status, success_rate_1h, avg_latency_ms,
            recent_errors, last_seen.
        """
        router = _get_router()
        health = router.get_provider_health(provider)
        return health

    return mcp


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """Run the antaris-router MCP server (stdio transport by default)."""
    parser = argparse.ArgumentParser(
        description="Antaris Router MCP Server — expose model routing over MCP."
    )
    parser.add_argument(
        "--transport",
        choices=["stdio", "sse"],
        default="stdio",
        help="MCP transport (default: stdio).",
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Host for SSE transport (default: 127.0.0.1).",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8766,
        help="Port for SSE transport (default: 8766).",
    )
    args = parser.parse_args()

    if not MCP_AVAILABLE:
        print(
            "ERROR: The 'mcp' package is not installed.\n"
            "Install it with: pip install mcp",
            file=sys.stderr,
        )
        sys.exit(1)

    server = create_server()

    if args.transport == "stdio":
        server.run(transport="stdio")
    else:
        server.settings.host = args.host
        server.settings.port = args.port
        server.run(transport="sse")


if __name__ == "__main__":
    main()
