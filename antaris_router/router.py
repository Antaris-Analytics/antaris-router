"""
Main routing interface for Antaris Router.

Combines classification, model registry, and cost tracking to provide
intelligent model selection based on task complexity and cost optimization.
"""

import hashlib
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

from .config import Config
from .classifier import TaskClassifier, ClassificationResult
from .registry import ModelRegistry, ModelInfo
from .costs import CostTracker


@dataclass
class RoutingDecision:
    """Result of routing decision with model selection and metadata."""
    model: str
    provider: str
    tier: str
    confidence: float
    reasoning: List[str]
    estimated_cost: float
    fallback_models: List[str]
    classification: ClassificationResult
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
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
                "signals": self.classification.signals
            }
        }


class Router:
    """Main router interface for intelligent model selection."""
    
    def __init__(self, config_path: str = None, enable_cost_tracking: bool = True):
        """Initialize the router.
        
        Args:
            config_path: Path to configuration directory
            enable_cost_tracking: Whether to enable cost tracking
        """
        self.config = Config(config_path)
        self.classifier = TaskClassifier(self.config)
        self.registry = ModelRegistry(self.config)
        self.cost_tracker = CostTracker() if enable_cost_tracking else None
        self.routing_history: List[RoutingDecision] = []
    
    def route(self, prompt: str, context: Dict = None, prefer: str = None, 
              min_tier: str = None, capability: str = None,
              estimate_tokens: tuple = (100, 50)) -> RoutingDecision:
        """Route a prompt to the most appropriate model.
        
        Args:
            prompt: The text prompt to route
            context: Optional context dictionary for classification
            prefer: Preferred model provider (e.g., "claude", "openai")
            min_tier: Minimum complexity tier to consider
            capability: Required capability (e.g., "vision", "code")
            estimate_tokens: Tuple of (input_tokens, output_tokens) for cost estimation
            
        Returns:
            RoutingDecision with selected model and metadata
        """
        # Classify the prompt
        classification = self.classifier.classify(prompt, context)
        
        # Override tier if min_tier is specified and higher than classified
        effective_tier = classification.tier
        if min_tier and self._tier_level(min_tier) > self._tier_level(classification.tier):
            effective_tier = min_tier
        
        # Get suitable models for the tier
        suitable_models = self.registry.models_for_tier(effective_tier)
        
        # Filter by capability if specified
        if capability:
            suitable_models = [m for m in suitable_models if m.has_capability(capability)]
        
        # Filter by provider preference if specified
        if prefer:
            preferred_models = [m for m in suitable_models if prefer.lower() in m.provider.lower()]
            if preferred_models:
                suitable_models = preferred_models
        
        if not suitable_models:
            # Fallback: try higher tiers
            fallback_decision = self._fallback_routing(prompt, classification, context, 
                                                     prefer, capability, estimate_tokens)
            if fallback_decision:
                return fallback_decision
            
            # No suitable models found
            raise ValueError(f"No suitable models found for tier '{effective_tier}'" + 
                           (f" with capability '{capability}'" if capability else "") +
                           (f" from provider '{prefer}'" if prefer else ""))
        
        # Select the cheapest suitable model
        selected_model = suitable_models[0]  # Already sorted by cost in registry
        
        # Calculate estimated cost
        input_tokens, output_tokens = estimate_tokens
        estimated_cost = selected_model.calculate_cost(input_tokens, output_tokens)
        
        # Prepare fallback options (next 2-3 cheapest models)
        fallback_models = [m.name for m in suitable_models[1:4]]
        
        # Generate routing reasoning
        reasoning = self._generate_routing_reasoning(
            classification, selected_model, effective_tier, prefer, capability
        )
        
        # Create routing decision
        decision = RoutingDecision(
            model=selected_model.name,
            provider=selected_model.provider,
            tier=effective_tier,
            confidence=classification.confidence,
            reasoning=reasoning,
            estimated_cost=estimated_cost,
            fallback_models=fallback_models,
            classification=classification
        )
        
        # Track the decision
        self.routing_history.append(decision)
        
        return decision
    
    def log_usage(self, decision: RoutingDecision, input_tokens: int, 
                  output_tokens: int) -> float:
        """Log actual usage for cost tracking.
        
        Args:
            decision: The RoutingDecision that was used
            input_tokens: Actual input tokens consumed
            output_tokens: Actual output tokens generated
            
        Returns:
            Actual cost incurred
        """
        if not self.cost_tracker:
            return 0.0
        
        model = self.registry.get_model(decision.model)
        if not model:
            return 0.0
        
        # Create prompt hash for deduplication tracking
        prompt_hash = hashlib.md5(
            f"{decision.tier}:{decision.confidence}".encode()
        ).hexdigest()[:12]
        
        return self.cost_tracker.log_usage(
            model=model,
            tier=decision.tier,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            prompt_hash=prompt_hash,
            confidence=decision.confidence
        )
    
    def cost_report(self, period: str = "week") -> Dict[str, Any]:
        """Generate cost report for the specified period.
        
        Args:
            period: Time period ("day", "week", "month", "all")
            
        Returns:
            Cost report dictionary
        """
        if not self.cost_tracker:
            return {"error": "Cost tracking is disabled"}
        
        return self.cost_tracker.report(period, self.registry)
    
    def savings_estimate(self, comparison_model: str = "gpt-4o") -> Dict[str, Any]:
        """Estimate cost savings compared to always using an expensive model.
        
        Args:
            comparison_model: Model to compare against
            
        Returns:
            Savings analysis dictionary
        """
        if not self.cost_tracker:
            return {"error": "Cost tracking is disabled"}
        
        return self.cost_tracker.savings_estimate(comparison_model, self.registry)
    
    def routing_analytics(self) -> Dict[str, Any]:
        """Get analytics on routing decisions.
        
        Returns:
            Analytics dictionary with routing patterns and performance
        """
        if not self.routing_history:
            return {"error": "No routing history available"}
        
        # Basic stats
        total_decisions = len(self.routing_history)
        tier_counts = {}
        model_counts = {}
        provider_counts = {}
        avg_confidence = 0.0
        
        for decision in self.routing_history:
            # Count by tier
            tier_counts[decision.tier] = tier_counts.get(decision.tier, 0) + 1
            
            # Count by model
            model_counts[decision.model] = model_counts.get(decision.model, 0) + 1
            
            # Count by provider
            provider_counts[decision.provider] = provider_counts.get(decision.provider, 0) + 1
            
            # Sum confidence for average
            avg_confidence += decision.confidence
        
        avg_confidence /= total_decisions
        
        # Calculate percentages
        tier_percentages = {tier: (count / total_decisions * 100) 
                          for tier, count in tier_counts.items()}
        
        return {
            "total_decisions": total_decisions,
            "avg_confidence": round(avg_confidence, 3),
            "tier_distribution": tier_counts,
            "tier_percentages": {k: round(v, 1) for k, v in tier_percentages.items()},
            "model_usage": model_counts,
            "provider_usage": provider_counts,
            "most_used_model": max(model_counts, key=model_counts.get) if model_counts else None,
            "most_used_provider": max(provider_counts, key=provider_counts.get) if provider_counts else None
        }
    
    def _fallback_routing(self, prompt: str, classification: ClassificationResult,
                         context: Dict, prefer: str, capability: str,
                         estimate_tokens: tuple) -> Optional[RoutingDecision]:
        """Attempt fallback routing to higher tiers if no models found.
        
        Args:
            prompt: Original prompt
            classification: Classification result
            context: Optional context
            prefer: Preferred provider
            capability: Required capability
            estimate_tokens: Token estimates
            
        Returns:
            RoutingDecision if fallback successful, None otherwise
        """
        tier_levels = ['trivial', 'simple', 'moderate', 'complex', 'expert']
        current_tier_index = tier_levels.index(classification.tier)
        
        # Try higher tiers
        for tier in tier_levels[current_tier_index + 1:]:
            suitable_models = self.registry.models_for_tier(tier)
            
            if capability:
                suitable_models = [m for m in suitable_models if m.has_capability(capability)]
            
            if prefer:
                preferred_models = [m for m in suitable_models if prefer.lower() in m.provider.lower()]
                if preferred_models:
                    suitable_models = preferred_models
            
            if suitable_models:
                selected_model = suitable_models[0]
                input_tokens, output_tokens = estimate_tokens
                estimated_cost = selected_model.calculate_cost(input_tokens, output_tokens)
                
                reasoning = [
                    f"Fallback to '{tier}' tier - no models available for '{classification.tier}'",
                    f"Selected {selected_model.name} from {selected_model.provider}",
                    "This may be more expensive than optimal"
                ]
                
                fallback_models = [m.name for m in suitable_models[1:4]]
                
                return RoutingDecision(
                    model=selected_model.name,
                    provider=selected_model.provider,
                    tier=tier,
                    confidence=classification.confidence * 0.8,  # Reduced confidence for fallback
                    reasoning=reasoning,
                    estimated_cost=estimated_cost,
                    fallback_models=fallback_models,
                    classification=classification
                )
        
        return None
    
    def _tier_level(self, tier: str) -> int:
        """Get numeric level for tier comparison.
        
        Args:
            tier: Tier name
            
        Returns:
            Numeric level (0-4)
        """
        tier_levels = ['trivial', 'simple', 'moderate', 'complex', 'expert']
        try:
            return tier_levels.index(tier)
        except ValueError:
            return 2  # Default to moderate
    
    def _generate_routing_reasoning(self, classification: ClassificationResult,
                                  selected_model: ModelInfo, effective_tier: str,
                                  prefer: str, capability: str) -> List[str]:
        """Generate human-readable reasoning for routing decision.
        
        Args:
            classification: Classification result
            selected_model: Selected model
            effective_tier: Effective tier used for routing
            prefer: Preferred provider filter
            capability: Required capability filter
            
        Returns:
            List of reasoning strings
        """
        reasoning = []
        
        # Classification reasoning
        reasoning.extend(classification.reasoning)
        
        # Tier override reasoning
        if effective_tier != classification.tier:
            reasoning.append(f"Tier elevated from '{classification.tier}' to '{effective_tier}' due to minimum requirement")
        
        # Model selection reasoning
        reasoning.append(f"Selected '{selected_model.name}' from {selected_model.provider}")
        reasoning.append(f"Cheapest model for '{effective_tier}' tier")
        
        # Filter reasoning
        if prefer:
            reasoning.append(f"Preferred provider: {prefer}")
        if capability:
            reasoning.append(f"Required capability: {capability}")
        
        # Cost reasoning
        avg_cost = (selected_model.cost_per_1k_input + selected_model.cost_per_1k_output) / 2
        if avg_cost == 0:
            reasoning.append("Local model - zero cost")
        elif avg_cost < 0.001:
            reasoning.append("Very low cost model")
        elif avg_cost < 0.01:
            reasoning.append("Low to moderate cost model")
        else:
            reasoning.append("Premium model - high capability")
        
        return reasoning
    
    def save_state(self, config_path: str) -> None:
        """Save router configuration state.
        
        Args:
            config_path: Path to save configuration
        """
        self.config.save_config(config_path)
        
        if self.cost_tracker:
            self.cost_tracker.save()
    
    def get_model_info(self, model_name: str) -> Optional[ModelInfo]:
        """Get information about a specific model.
        
        Args:
            model_name: Name of the model
            
        Returns:
            ModelInfo if found, None otherwise
        """
        return self.registry.get_model(model_name)
    
    def list_models_for_tier(self, tier: str) -> List[Dict[str, Any]]:
        """List all models suitable for a tier with their details.
        
        Args:
            tier: Complexity tier
            
        Returns:
            List of model information dictionaries
        """
        models = self.registry.models_for_tier(tier)
        return [
            {
                "name": model.name,
                "provider": model.provider,
                "cost_per_1k_input": model.cost_per_1k_input,
                "cost_per_1k_output": model.cost_per_1k_output,
                "capabilities": model.capabilities,
                "max_tokens": model.max_tokens
            }
            for model in models
        ]