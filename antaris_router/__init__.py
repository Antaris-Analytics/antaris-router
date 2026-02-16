"""
Antaris Router — Rule-based model router for LLM cost optimization.

A deterministic router that picks the cheapest model capable of handling
a given task complexity. Zero API calls, zero dependencies, pure file-based
decision making.

Usage:
    from antaris_router import Router

    router = Router("./config")
    decision = router.route("What's 2+2?")
    print(f"Use {decision.model} (${decision.estimated_cost:.4f})")
    
    # Log actual usage for cost tracking
    router.log_usage(decision, input_tokens=50, output_tokens=20)
"""

__version__ = "0.3.0"

from .router import Router, RoutingDecision
from .classifier import TaskClassifier, ClassificationResult
from .registry import ModelRegistry, ModelInfo
from .costs import CostTracker, UsageRecord
from .config import Config

__all__ = [
    "Router",
    "RoutingDecision",
    "TaskClassifier",
    "ClassificationResult",
    "ModelRegistry",
    "ModelInfo",
    "CostTracker",
    "UsageRecord",
    "Config",
]