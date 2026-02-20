"""
Adaptive router for Antaris Router v2.0.

Combines semantic classification, quality tracking, and context awareness
into an intelligent routing system that learns and improves over time.
Supports fallback chains, A/B testing, and multi-objective optimization.

Sprint 2.3: Confidence-gated routing with escalate/safe_default/clarify
strategies, RouteDecision dataclass with explainability, and explain()
method that returns structured decision analysis from a request string.

Zero external dependencies. All state is file-based.
"""

import hashlib
import json
import logging
import os
import random
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

from .semantic import SemanticClassifier, SemanticResult
from .quality import QualityTracker, RoutingDecision
from .confidence import (
    ProviderHealthTracker,
    CONFIDENCE_BASIS_SEMANTIC,
    CONFIDENCE_BASIS_QUALITY,
    CONFIDENCE_BASIS_COMPOSITE,
)

_log = logging.getLogger(__name__)

# ── Confidence strategy constants ─────────────────────────────────────────────
STRATEGY_ESCALATE     = "escalate"
STRATEGY_SAFE_DEFAULT = "safe_default"
STRATEGY_CLARIFY      = "clarify"

VALID_CONFIDENCE_STRATEGIES = {STRATEGY_ESCALATE, STRATEGY_SAFE_DEFAULT, STRATEGY_CLARIFY}

# Default confidence threshold for low-confidence strategies
DEFAULT_CONFIDENCE_THRESHOLD = 0.6


@dataclass
class RouteDecision:
    """Enhanced routing decision for Sprint 2.3 — confidence-gated routing.

    Attributes:
        model: Selected model name.
        tier: Complexity tier (trivial/simple/moderate/complex/expert).
        confidence: Routing confidence 0.0–1.0.
        basis: How confidence was determined
            (semantic_classifier / quality_tracker / composite / rule_based).
        reason: Human-readable explanation of the routing choice.
        strategy_applied: None when routing proceeded normally;
            ``"escalated"``, ``"safe_default"``, or ``"clarify"`` when the
            low-confidence strategy fired.
        fallback_chain: Ordered list of alternative models.
        prompt_hash: MD5 prefix of the original prompt (for outcome tracking).
        metadata: Additional signals from the classification step.
    """

    model: str
    tier: str
    confidence: float
    basis: str
    reason: str
    strategy_applied: Optional[str]  # None | "escalated" | "safe_default" | "clarify"
    fallback_chain: List[str] = field(default_factory=list)
    prompt_hash: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialise to plain dict (Pipeline telemetry-ready)."""
        return {
            "model": self.model,
            "tier": self.tier,
            "confidence": self.confidence,
            "basis": self.basis,
            "reason": self.reason,
            "strategy_applied": self.strategy_applied,
            "fallback_chain": self.fallback_chain,
            "prompt_hash": self.prompt_hash,
            "metadata": self.metadata,
        }


@dataclass
class RoutingResult:
    """Result of an adaptive routing decision."""
    model: str
    tier: str
    confidence: float
    reasoning: List[str]
    fallback_chain: List[str]  # Models to try if primary fails
    ab_test: bool = False  # Whether this is an A/B test route
    prompt_hash: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ModelConfig:
    """Configuration for a model in the routing registry."""
    name: str
    tier_range: Tuple[str, str]  # (min_tier, max_tier) this model handles
    cost_per_1k_input: float
    cost_per_1k_output: float
    avg_latency_ms: float = 1000
    max_context: int = 128000
    provider: str = ""
    tags: List[str] = field(default_factory=list)
    supports_streaming: bool = True


# Tier ordering for comparison
TIER_ORDER = {'trivial': 0, 'simple': 1, 'moderate': 2, 'complex': 3, 'expert': 4}


class AdaptiveRouter:
    """Intelligent model router that learns from outcomes.
    
    Features:
    - Semantic classification (TF-IDF, not keywords)
    - Quality tracking with outcome learning
    - Fallback chains with automatic escalation
    - A/B testing (routes X% to premium to validate cheap routing)
    - Multi-objective optimization (quality, cost, speed)
    - Context-aware routing (conversation history, iteration count)
    - All file-based, all deterministic (given same state)
    """

    def __init__(
        self,
        workspace: str = ".",
        ab_test_rate: float = 0.05,
        # ── Sprint 2.3: confidence-gated routing ──────────────────────────
        confidence_threshold: float = 0.0,
        confidence_strategy: Optional[str] = None,
        safe_default_model: Optional[str] = None,
    ):
        """Initialize the adaptive router.

        Args:
            workspace: Directory for persistent state files.
            ab_test_rate: Fraction of requests to route to premium for
                validation (0.0–1.0).
            confidence_threshold: When confidence falls below this value
                the ``confidence_strategy`` fires.  Set to ``0.0``
                (default) to disable confidence-gating.
            confidence_strategy: One of ``"escalate"``, ``"safe_default"``,
                or ``"clarify"``.  Required when ``confidence_threshold > 0``.

                - ``"escalate"`` — bump to the next model tier.
                - ``"safe_default"`` — route to ``safe_default_model``.
                - ``"clarify"`` — keep the routing but set
                  ``strategy_applied="clarify"`` so callers know the
                  request needs clarification.

            safe_default_model: Model name to use when
                ``confidence_strategy="safe_default"`` and confidence is
                low.  Ignored for other strategies.
        """
        if confidence_strategy is not None and confidence_strategy not in VALID_CONFIDENCE_STRATEGIES:
            raise ValueError(
                f"confidence_strategy must be one of "
                f"{sorted(VALID_CONFIDENCE_STRATEGIES)}, "
                f"got {confidence_strategy!r}"
            )

        self.workspace = os.path.abspath(workspace)
        self.classifier = SemanticClassifier(workspace)
        self.tracker = QualityTracker(workspace)
        self.ab_test_rate = ab_test_rate
        self._health_tracker = ProviderHealthTracker()

        # Sprint 2.3
        self.confidence_threshold: float = confidence_threshold
        self.confidence_strategy: Optional[str] = confidence_strategy
        self.safe_default_model: Optional[str] = safe_default_model

        self.models: Dict[str, ModelConfig] = {}
        self._config_path = os.path.join(workspace, "router_config.json")
        self._load_config()

    def _load_config(self):
        """Load router configuration."""
        if os.path.exists(self._config_path):
            try:
                with open(self._config_path) as f:
                    data = json.load(f)
                for name, cfg in data.get('models', {}).items():
                    self.models[name] = ModelConfig(
                        name=name,
                        tier_range=tuple(cfg.get('tier_range', ['trivial', 'expert'])),
                        cost_per_1k_input=cfg.get('cost_per_1k_input', 0.0),
                        cost_per_1k_output=cfg.get('cost_per_1k_output', 0.0),
                        avg_latency_ms=cfg.get('avg_latency_ms', 1000),
                        max_context=cfg.get('max_context', 128000),
                        provider=cfg.get('provider', ''),
                        tags=cfg.get('tags', []),
                        supports_streaming=cfg.get('supports_streaming', True),
                    )
                self.ab_test_rate = data.get('ab_test_rate', self.ab_test_rate)
            except json.JSONDecodeError as exc:
                # Config file exists but is malformed — hard fail rather than
                # silently routing with zero models (Gemini review fix).
                raise ValueError(
                    f"Router config at {self._config_path} is corrupt (invalid JSON): {exc}. "
                    "Delete or repair the file to reset to defaults."
                ) from exc
            except Exception as exc:
                _log.warning(
                    "Could not load router config from %s: %s — using empty config",
                    self._config_path, exc,
                )
                self.models = {}

    def register_model(self, config: ModelConfig):
        """Register a model for routing.
        
        Args:
            config: Model configuration
        """
        self.models[config.name] = config
        self._save_config()

    def _save_config(self):
        """Save router configuration."""
        os.makedirs(self.workspace, exist_ok=True)
        data = {
            'models': {
                name: {
                    'tier_range': list(cfg.tier_range),
                    'cost_per_1k_input': cfg.cost_per_1k_input,
                    'cost_per_1k_output': cfg.cost_per_1k_output,
                    'avg_latency_ms': cfg.avg_latency_ms,
                    'max_context': cfg.max_context,
                    'provider': cfg.provider,
                    'tags': cfg.tags,
                    'supports_streaming': cfg.supports_streaming,
                }
                for name, cfg in self.models.items()
            },
            'ab_test_rate': self.ab_test_rate,
        }
        from .utils import atomic_write_json
        atomic_write_json(self._config_path, data)

    def route(self, prompt: str, context: Dict = None,
              optimize: str = "balanced") -> RoutingResult:
        """Route a prompt to the best model.
        
        Args:
            prompt: The prompt text to route
            context: Optional context (conversation history, iteration count, etc.)
            optimize: Optimization goal - "quality", "cost", "speed", "balanced"
            
        Returns:
            RoutingResult with model selection, reasoning, and fallback chain
        """
        if not self.models:
            return RoutingResult(
                model="",
                tier="moderate",
                confidence=0.0,
                reasoning=["No models registered"],
                fallback_chain=[],
            )
        
        prompt_hash = hashlib.md5(prompt.encode()).hexdigest()[:12]
        
        # Step 1: Semantic classification
        classification = self.classifier.classify(prompt)
        tier = classification.tier
        reasoning = [f"Semantic classification: {tier} (confidence: {classification.confidence:.2f})"]
        
        # Step 2: Context adjustment
        if context:
            tier, ctx_reason = self._adjust_for_context(tier, context)
            if ctx_reason:
                reasoning.append(ctx_reason)
        
        # Step 3: Find eligible models for this tier
        eligible = self._eligible_models(tier)
        if not eligible:
            # Fall back to any model
            eligible = list(self.models.keys())
            reasoning.append("No tier-specific models; using all available")

        # Filter out rate-limited models
        non_rate_limited = [
            m for m in eligible
            if not self._health_tracker.is_rate_limited(m)
        ]
        if non_rate_limited:
            if len(non_rate_limited) < len(eligible):
                skipped = set(eligible) - set(non_rate_limited)
                reasoning.append(
                    f"Skipping rate-limited model(s): {', '.join(sorted(skipped))}"
                )
            eligible = non_rate_limited
        else:
            reasoning.append("All eligible models are rate-limited; using full set")
        
        # Step 4: Check quality history — skip models with poor track record
        filtered = []
        for model_name in eligible:
            if not self.tracker.should_escalate(model_name, tier):
                filtered.append(model_name)
            else:
                reasoning.append(f"Skipping {model_name} (poor quality history on {tier})")
        
        if not filtered:
            # Prefer models with no data (neutral 0.5) over models with bad data
            # Sort eligible by score descending — untested models rank higher than failed ones
            filtered = sorted(eligible, 
                            key=lambda m: self.tracker.get_model_score(m, tier),
                            reverse=True)
            reasoning.append("All models have poor history; preferring least-bad option")
        
        # Step 5: Select best model based on optimization goal
        if len(filtered) == 1:
            selected = filtered[0]
            confidence = classification.confidence
        else:
            selected, rec_confidence = self.tracker.recommend_model(
                tier, filtered, optimize=optimize
            )
            if not selected:
                selected = filtered[0]
            confidence = min(classification.confidence, rec_confidence) if rec_confidence > 0 else classification.confidence
            reasoning.append(f"Quality tracker recommends {selected} ({optimize} optimization)")
        
        # Step 6: A/B testing — occasionally route to premium to validate
        ab_test = False
        if (self.ab_test_rate > 0 and 
            random.random() < self.ab_test_rate and
            tier in ('trivial', 'simple', 'moderate')):
            premium = self._get_premium_model()
            if premium and premium != selected:
                reasoning.append(f"A/B test: routing to {premium} instead of {selected}")
                selected = premium
                ab_test = True
        
        # Step 7: Build fallback chain
        fallback_chain = self._build_fallback_chain(selected, tier, eligible)
        
        # Step 8: Record decision
        decision = RoutingDecision(
            prompt_hash=prompt_hash,
            tier=tier,
            model=selected,
            timestamp=time.time(),
        )
        self.tracker.record(decision)
        
        return RoutingResult(
            model=selected,
            tier=tier,
            confidence=confidence,
            reasoning=reasoning,
            fallback_chain=fallback_chain,
            ab_test=ab_test,
            prompt_hash=prompt_hash,
            metadata={
                'classification': {
                    'tier_scores': classification.signals.get('tier_scores', {}),
                    'similar_examples': [
                        (text[:80], round(sim, 3))
                        for text, sim in classification.similar_examples
                    ],
                },
                'optimize': optimize,
            }
        )

    # ── Sprint 2.3: confidence-gated routing ──────────────────────────────

    def route_with_confidence(
        self,
        prompt: str,
        context: Dict = None,
        optimize: str = "balanced",
    ) -> RouteDecision:
        """Route a prompt and return a :class:`RouteDecision` with confidence metadata.

        Applies the configured ``confidence_strategy`` when confidence falls
        below ``confidence_threshold``.

        Args:
            prompt: The prompt text.
            context: Optional context dict (same keys as :meth:`route`).
            optimize: Optimization goal — ``"quality"``, ``"cost"``,
                ``"speed"``, or ``"balanced"``.

        Returns:
            :class:`RouteDecision` with model, tier, confidence, basis,
            reason, strategy_applied, and fallback_chain.
        """
        # Delegate core routing to the existing route() method
        base = self.route(prompt, context=context, optimize=optimize)

        # Determine confidence basis
        basis = self._determine_basis(base)

        # Build initial reason string
        reason_parts = [
            f"Semantic classification: {base.tier} tier "
            f"(confidence {base.confidence:.2f})"
        ]
        if base.reasoning:
            reason_parts.extend(base.reasoning[1:])  # skip duplicate first entry
        reason = "; ".join(reason_parts[:3])  # keep it concise

        strategy_applied: Optional[str] = None

        # Apply low-confidence strategy if configured
        if (
            self.confidence_threshold > 0.0
            and base.confidence < self.confidence_threshold
            and self.confidence_strategy is not None
        ):
            if self.confidence_strategy == STRATEGY_ESCALATE:
                escalated_model = self._escalate_model(base.model, base.tier)
                if escalated_model and escalated_model != base.model:
                    orig_confidence = base.confidence
                    orig_model = base.model
                    base = self.route(prompt, context=context, optimize="quality")
                    # Re-derive basis/reason from new result but keep strategy marker
                    basis = self._determine_basis(base)
                    reason = (
                        f"Escalated from {orig_model!r}: low confidence "
                        f"({orig_confidence:.2f} < {self.confidence_threshold:.2f}); "
                        f"using {base.model} for {base.tier} tier"
                    )
                strategy_applied = STRATEGY_ESCALATE

            elif self.confidence_strategy == STRATEGY_SAFE_DEFAULT:
                if self.safe_default_model:
                    reason = (
                        f"Safe default applied: low confidence "
                        f"({base.confidence:.2f} < {self.confidence_threshold:.2f}); "
                        f"routing to configured fallback {self.safe_default_model!r}"
                    )
                    # Return a decision pointing at the safe default
                    return RouteDecision(
                        model=self.safe_default_model,
                        tier=base.tier,
                        confidence=base.confidence,
                        basis=basis,
                        reason=reason,
                        strategy_applied=STRATEGY_SAFE_DEFAULT,
                        fallback_chain=base.fallback_chain,
                        prompt_hash=base.prompt_hash,
                        metadata=base.metadata,
                    )
                strategy_applied = STRATEGY_SAFE_DEFAULT

            elif self.confidence_strategy == STRATEGY_CLARIFY:
                reason = (
                    f"Clarification needed: confidence {base.confidence:.2f} < "
                    f"threshold {self.confidence_threshold:.2f}; "
                    f"model {base.model!r} selected but input may be ambiguous"
                )
                strategy_applied = STRATEGY_CLARIFY

        return RouteDecision(
            model=base.model,
            tier=base.tier,
            confidence=base.confidence,
            basis=basis,
            reason=reason,
            strategy_applied=strategy_applied,
            fallback_chain=base.fallback_chain,
            prompt_hash=base.prompt_hash,
            metadata=base.metadata,
        )

    def explain(self, request: str, context: Dict = None) -> Dict[str, Any]:
        """Return a structured explanation of how *request* would be routed.

        Unlike :meth:`route_with_confidence` this method is **read-only** —
        it does not record a decision in the quality tracker.

        Args:
            request: The prompt text to explain.
            context: Optional context dict.

        Returns:
            Dict with keys:
                - ``classification``: tier, confidence, tier_scores
                - ``quality_scores``: per-model quality score for candidate models
                - ``cost_estimate``: cheapest candidate model and its cost/1K tokens
                - ``candidates``: list of candidate model names for the tier
                - ``why_selected``: dict explaining why each candidate was or wasn't chosen
                - ``summary``: human-readable multi-line explanation string
        """
        classification = self.classifier.classify(request)
        tier = classification.tier

        # Cost estimate: cost-per-1k-input for eligible models
        eligible = self._eligible_models(tier)
        cost_estimate: Dict[str, Any] = {}
        if eligible:
            cheapest = min(
                eligible,
                key=lambda m: self.models[m].cost_per_1k_input
                if m in self.models else 0.0,
                default=None,
            )
            if cheapest and cheapest in self.models:
                cfg = self.models[cheapest]
                cost_estimate = {
                    "model": cheapest,
                    "cost_per_1k_input": cfg.cost_per_1k_input,
                    "cost_per_1k_output": cfg.cost_per_1k_output,
                }

        # Quality scores for eligible candidates
        quality_scores: Dict[str, float] = {
            m: round(self.tracker.get_model_score(m, tier), 4)
            for m in eligible
        }

        # Why each candidate was/wasn't selected
        why_selected: Dict[str, str] = {}
        for m in eligible:
            score = quality_scores.get(m, 0.5)
            if self.tracker.should_escalate(m, tier):
                why_selected[m] = f"skipped — quality score {score:.2f} below threshold"
            else:
                why_selected[m] = f"eligible — quality score {score:.2f}"

        # Summary
        conf_pct = int(round(classification.confidence * 100))
        basis = self._determine_basis_from_classification(classification)
        lines = [
            f"Request classified as '{tier}' tier ({conf_pct}% confidence, basis: {basis}).",
            f"Eligible models: {', '.join(eligible) if eligible else 'none registered'}.",
        ]
        if cost_estimate:
            lines.append(
                f"Cheapest candidate: {cost_estimate['model']} "
                f"(${cost_estimate['cost_per_1k_input']:.5f}/1K input, "
                f"${cost_estimate['cost_per_1k_output']:.5f}/1K output)."
            )
        if self.confidence_threshold > 0.0 and self.confidence_strategy:
            if classification.confidence < self.confidence_threshold:
                lines.append(
                    f"Confidence {classification.confidence:.2f} < threshold "
                    f"{self.confidence_threshold:.2f}: strategy '{self.confidence_strategy}' "
                    f"would fire."
                )
            else:
                lines.append(
                    f"Confidence {classification.confidence:.2f} >= threshold "
                    f"{self.confidence_threshold:.2f}: normal routing."
                )

        return {
            "classification": {
                "tier": tier,
                "confidence": round(classification.confidence, 4),
                "tier_scores": classification.signals.get("tier_scores", {}),
            },
            "quality_scores": quality_scores,
            "cost_estimate": cost_estimate,
            "candidates": eligible,
            "why_selected": why_selected,
            "summary": "\n".join(lines),
        }

    # ── Sprint 2.3 private helpers ─────────────────────────────────────────

    def _determine_basis(self, result: "RoutingResult") -> str:
        """Choose the confidence basis label for a RoutingResult."""
        has_quality_data = any(
            self.tracker.get_model_score(result.model, result.tier) != 0.5
            for _ in [1]  # single-iteration to call once
        )
        if has_quality_data:
            return CONFIDENCE_BASIS_COMPOSITE
        return CONFIDENCE_BASIS_SEMANTIC

    def _determine_basis_from_classification(self, classification: "SemanticResult") -> str:
        """Choose the confidence basis from a raw SemanticResult."""
        return CONFIDENCE_BASIS_SEMANTIC

    def _escalate_model(self, current_model: str, current_tier: str) -> Optional[str]:
        """Return the next-tier-up model name, or None if already at max."""
        tier_list = ["trivial", "simple", "moderate", "complex", "expert"]
        try:
            idx = tier_list.index(current_tier)
        except ValueError:
            idx = 2
        for higher_tier in tier_list[idx + 1:]:
            candidates = self._eligible_models(higher_tier)
            if candidates:
                return candidates[0]
        return None

    # ── End Sprint 2.3 ────────────────────────────────────────────────────

    def report_outcome(self, prompt_hash: str, quality_score: float = None,
                       success: bool = None, latency_ms: float = None):
        """Report the outcome of a routing decision.

        Call this after using the routed model to help the router learn.

        Args:
            prompt_hash: The prompt_hash from RoutingResult
            quality_score: 0.0-1.0 quality rating
            success: Whether the task was completed successfully
            latency_ms: Response time in milliseconds
        """
        self.tracker.record_outcome(
            prompt_hash=prompt_hash,
            quality_score=quality_score,
            success=success,
            latency_ms=latency_ms,
        )

    def escalate(self, prompt_hash: str) -> Optional[str]:
        """Escalate a failed routing to the next model in the fallback chain.
        
        Records the escalation and returns the next model to try.
        
        Args:
            prompt_hash: The prompt_hash from the original routing
            
        Returns:
            Next model name, or None if no fallbacks remain
        """
        # Find the original decision
        for decision in reversed(self.tracker.decisions):
            if decision['prompt_hash'] == prompt_hash:
                decision['escalated'] = True
                current_model = decision['model']
                tier = decision['tier']

                # Find next model in capability order — only within eligible set
                eligible = self._eligible_models(tier)
                current_idx = TIER_ORDER.get(tier, 2)

                # Try the next tier up, filtering to eligible models only
                for higher_tier in ['moderate', 'complex', 'expert']:
                    if TIER_ORDER[higher_tier] > current_idx:
                        for model_name in eligible:
                            if model_name != current_model:
                                config = self.models[model_name]
                                min_tier = TIER_ORDER.get(config.tier_range[0], 0)
                                max_tier = TIER_ORDER.get(config.tier_range[1], 4)
                                if min_tier <= TIER_ORDER[higher_tier] <= max_tier:
                                    return model_name
                break

        return None

    def teach(self, prompt: str, correct_tier: str):
        """Teach the classifier about a prompt's correct tier.
        
        Use this when the semantic classifier gets it wrong.
        The correction is permanently learned.
        
        Args:
            prompt: The prompt text
            correct_tier: The correct complexity tier
        """
        self.classifier.learn(prompt, correct_tier)

    def _adjust_for_context(self, tier: str, context: Dict) -> Tuple[str, str]:
        """Adjust tier based on conversation context.
        
        Args:
            tier: Current tier from classifier
            context: Context dict with optional keys:
                - iteration: How many times this has been attempted
                - conversation_length: Number of prior messages
                - user_expertise: "beginner", "intermediate", "expert"
                - urgency: "low", "normal", "high"
                
        Returns:
            Tuple of (adjusted_tier, reasoning_string)
        """
        tier_idx = TIER_ORDER.get(tier, 2)
        reason = ""
        
        # Multiple iterations → escalate (user is struggling)
        iteration = context.get('iteration', 1)
        if iteration >= 3:
            tier_idx = min(tier_idx + 1, 4)
            reason = f"Iteration {iteration}: escalating tier"
        
        # Long conversation → probably complex
        conv_len = context.get('conversation_length', 0)
        if conv_len > 10 and tier_idx < 2:
            tier_idx = 2
            reason = f"Long conversation ({conv_len} msgs): minimum moderate"
        
        # User expertise affects routing
        expertise = context.get('user_expertise', 'intermediate')
        if expertise == 'expert' and tier_idx < 2:
            tier_idx = max(tier_idx, 2)
            reason = "Expert user: minimum moderate tier"
        
        # Urgency boost
        if context.get('urgency') == 'high':
            tier_idx = min(tier_idx + 1, 4)
            reason = "High urgency: escalating tier"
        
        # Convert back to tier name
        idx_to_tier = {v: k for k, v in TIER_ORDER.items()}
        new_tier = idx_to_tier.get(tier_idx, tier)
        
        return new_tier, reason

    def _eligible_models(self, tier: str) -> List[str]:
        """Get models eligible for a given tier."""
        tier_idx = TIER_ORDER.get(tier, 2)
        eligible = []
        
        for name, config in self.models.items():
            min_idx = TIER_ORDER.get(config.tier_range[0], 0)
            max_idx = TIER_ORDER.get(config.tier_range[1], 4)
            if min_idx <= tier_idx <= max_idx:
                eligible.append(name)
        
        return eligible

    def _get_premium_model(self) -> Optional[str]:
        """Get the highest-capability model for A/B testing."""
        best = None
        best_max = -1
        for name, config in self.models.items():
            max_idx = TIER_ORDER.get(config.tier_range[1], 0)
            if max_idx > best_max:
                best = name
                best_max = max_idx
        return best

    def _build_fallback_chain(self, primary: str, tier: str,
                               eligible: List[str]) -> List[str]:
        """Build ordered fallback chain from eligible models.
        
        Order: cheapest eligible → most expensive eligible, excluding primary.
        """
        chain = []
        models_by_cost = sorted(
            [(name, self.models[name].cost_per_1k_input) for name in eligible if name != primary],
            key=lambda x: x[1]
        )
        chain = [name for name, _ in models_by_cost]
        return chain

    def save(self):
        """Persist all state to disk."""
        self.classifier.save()
        self.tracker.save()
        self._save_config()

    def get_stats(self) -> Dict:
        """Get comprehensive router statistics."""
        return {
            'classifier': self.classifier.get_stats(),
            'quality_tracker': self.tracker.get_stats(),
            'models_registered': len(self.models),
            'ab_test_rate': self.ab_test_rate,
        }
