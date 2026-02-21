"""
Antaris Router — Adaptive model router for LLM cost optimization.

v2.0: Semantic classification (TF-IDF), quality tracking with outcome
learning, fallback chains, A/B testing, and context-aware routing.
Zero external dependencies. All state is file-based.

Usage:
    from antaris_router import AdaptiveRouter, ModelConfig

    router = AdaptiveRouter("./config")
    router.register_model(ModelConfig(
        name="gpt-4o",
        tier_range=("moderate", "expert"),
        cost_per_1k_input=0.0025,
        cost_per_1k_output=0.01,
    ))
    router.register_model(ModelConfig(
        name="gpt-4o-mini",
        tier_range=("trivial", "moderate"),
        cost_per_1k_input=0.00015,
        cost_per_1k_output=0.0006,
    ))

    result = router.route("Implement a distributed task queue")
    print(f"Use {result.model} ({result.tier}, confidence: {result.confidence:.2f})")

    # Report outcome to help the router learn
    router.report_outcome(result.prompt_hash, quality_score=0.9, success=True)

Legacy v1 API (keyword-based) is still available:
    from antaris_router import Router
    router = Router("./config")

Sprint 5 — Cost-Performance SLAs:
    from antaris_router import Router, SLAConfig

    sla = SLAConfig(max_latency_ms=200, budget_per_hour_usd=5.00)
    router = Router(sla=sla, fallback_chain=["claude-opus-4-6", "claude-haiku-3-5"])
    result = router.route("Hello", auto_scale=True)
"""

__version__ = "4.0.0"

# v2.0 API — adaptive routing with semantic classification
# Sprint 2.3 adds RouteDecision and confidence strategy constants
from .adaptive import (
    AdaptiveRouter,
    RoutingResult,
    ModelConfig,
    RouteDecision,
    STRATEGY_ESCALATE,
    STRATEGY_SAFE_DEFAULT,
    STRATEGY_CLARIFY,
    VALID_CONFIDENCE_STRATEGIES,
    DEFAULT_CONFIDENCE_THRESHOLD,
)
from .semantic import SemanticClassifier, SemanticResult, TFIDFVectorizer
from .quality import QualityTracker, RoutingDecision as QualityDecision

# v1.0 API — legacy keyword-based routing (backward compatible)
from .router import Router, RoutingDecision
from .classifier import TaskClassifier, ClassificationResult
from .registry import ModelRegistry, ModelInfo
from .costs import CostTracker, UsageRecord
from .config import Config

# Sprint 5 — SLA
from .sla import SLAConfig, SLAMonitor, SLARecord

__all__ = [
    # v2.0
    "AdaptiveRouter",
    "RoutingResult",
    "ModelConfig",
    "SemanticClassifier",
    "SemanticResult",
    "TFIDFVectorizer",
    "QualityTracker",
    "QualityDecision",
    # Sprint 2.3 — confidence-gated routing
    "RouteDecision",
    "STRATEGY_ESCALATE",
    "STRATEGY_SAFE_DEFAULT",
    "STRATEGY_CLARIFY",
    "VALID_CONFIDENCE_STRATEGIES",
    "DEFAULT_CONFIDENCE_THRESHOLD",

    # v1.0 (legacy)
    "Router",
    "RoutingDecision",
    "TaskClassifier",
    "ClassificationResult",
    "ModelRegistry",
    "ModelInfo",
    "CostTracker",
    "UsageRecord",
    "Config",

    # Sprint 5 — SLA
    "SLAConfig",
    "SLAMonitor",
    "SLARecord",
]
