"""
Adaptive router for Antaris Router v2.0.

Combines semantic classification, quality tracking, and context awareness
into an intelligent routing system that learns and improves over time.
Supports fallback chains, A/B testing, and multi-objective optimization.

Zero external dependencies. All state is file-based.
"""

import hashlib
import json
import os
import random
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

from .semantic import SemanticClassifier, SemanticResult
from .quality import QualityTracker, RoutingDecision


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

    def __init__(self, workspace: str = ".", ab_test_rate: float = 0.05):
        """Initialize the adaptive router.
        
        Args:
            workspace: Directory for persistent state files
            ab_test_rate: Fraction of requests to route to premium for validation (0.0-1.0)
        """
        self.workspace = os.path.abspath(workspace)
        self.classifier = SemanticClassifier(workspace)
        self.tracker = QualityTracker(workspace)
        self.ab_test_rate = ab_test_rate
        
        self.models: Dict[str, ModelConfig] = {}
        self._config_path = os.path.join(workspace, "router_config.json")
        self._load_config()

    def _load_config(self):
        """Load router configuration."""
        if os.path.exists(self._config_path):
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
                    )
                self.ab_test_rate = data.get('ab_test_rate', self.ab_test_rate)

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
                
                # Find next model in capability order
                eligible = self._eligible_models(tier)
                current_idx = TIER_ORDER.get(tier, 2)
                
                # Try the next tier up
                for higher_tier in ['moderate', 'complex', 'expert']:
                    if TIER_ORDER[higher_tier] > current_idx:
                        for model_name, config in self.models.items():
                            if model_name != current_model:
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
