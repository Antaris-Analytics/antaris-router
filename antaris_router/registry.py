"""
Model registry for Antaris Router.

Manages model definitions, capabilities, costs, and tier assignments.
Provides methods to query models by tier and sort by cost.
"""

import json
import os
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

from .config import Config


@dataclass
class ModelInfo:
    """Information about a language model."""
    name: str
    provider: str
    cost_per_1k_input: float
    cost_per_1k_output: float
    capabilities: List[str]
    max_tokens: int
    tier: List[str]
    
    def calculate_cost(self, input_tokens: int, output_tokens: int) -> float:
        """Calculate cost for given token usage.
        
        Args:
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens
            
        Returns:
            Total cost in dollars
        """
        input_cost = (input_tokens / 1000) * self.cost_per_1k_input
        output_cost = (output_tokens / 1000) * self.cost_per_1k_output
        return input_cost + output_cost
    
    def supports_tier(self, tier: str) -> bool:
        """Check if model supports a given complexity tier.
        
        Args:
            tier: Complexity tier to check
            
        Returns:
            True if model supports the tier
        """
        return tier in self.tier
    
    def has_capability(self, capability: str) -> bool:
        """Check if model has a specific capability.
        
        Args:
            capability: Capability to check (e.g., 'vision', 'code')
            
        Returns:
            True if model has the capability
        """
        return capability in self.capabilities


class ModelRegistry:
    """Registry for managing language model definitions."""
    
    def __init__(self, config: Config):
        """Initialize model registry.
        
        Args:
            config: Configuration instance with model definitions
        """
        self.config = config
        self.models = self._load_models()
    
    def _load_models(self) -> Dict[str, ModelInfo]:
        """Load models from configuration.
        
        Returns:
            Dictionary mapping model names to ModelInfo objects
        """
        models = {}
        for model_def in self.config.get_models():
            model = ModelInfo(
                name=model_def['name'],
                provider=model_def['provider'],
                cost_per_1k_input=model_def['cost_per_1k_input'],
                cost_per_1k_output=model_def['cost_per_1k_output'],
                capabilities=model_def['capabilities'],
                max_tokens=model_def['max_tokens'],
                tier=model_def['tier']
            )
            models[model.name] = model
        return models
    
    def get_model(self, name: str) -> Optional[ModelInfo]:
        """Get model by name.
        
        Args:
            name: Model name
            
        Returns:
            ModelInfo if found, None otherwise
        """
        return self.models.get(name)
    
    def list_models(self) -> List[ModelInfo]:
        """Get all registered models.
        
        Returns:
            List of all ModelInfo objects
        """
        return list(self.models.values())
    
    def models_for_tier(self, tier: str) -> List[ModelInfo]:
        """Get models that support a given complexity tier, sorted by cost.
        
        Args:
            tier: Complexity tier ('trivial', 'simple', 'moderate', 'complex', 'expert')
            
        Returns:
            List of ModelInfo objects sorted by cost (cheapest first)
        """
        suitable_models = [
            model for model in self.models.values()
            if model.supports_tier(tier)
        ]
        
        # Sort by average cost per 1k tokens (input + output)
        suitable_models.sort(key=lambda m: (m.cost_per_1k_input + m.cost_per_1k_output) / 2)
        
        return suitable_models
    
    def models_with_capability(self, capability: str) -> List[ModelInfo]:
        """Get models that have a specific capability.
        
        Args:
            capability: Required capability (e.g., 'vision', 'code')
            
        Returns:
            List of ModelInfo objects with the capability
        """
        return [
            model for model in self.models.values()
            if model.has_capability(capability)
        ]
    
    def cheapest_for_tier(self, tier: str, capability: str = None) -> Optional[ModelInfo]:
        """Get the cheapest model for a given tier and optional capability.
        
        Args:
            tier: Complexity tier
            capability: Optional required capability
            
        Returns:
            Cheapest ModelInfo that meets criteria, or None if none found
        """
        models = self.models_for_tier(tier)
        
        if capability:
            models = [m for m in models if m.has_capability(capability)]
        
        return models[0] if models else None
    
    def add_model(self, model_info: ModelInfo) -> None:
        """Add or update a model in the registry.
        
        Args:
            model_info: ModelInfo object to add
        """
        self.models[model_info.name] = model_info
        
        # Update config
        model_def = {
            'name': model_info.name,
            'provider': model_info.provider,
            'cost_per_1k_input': model_info.cost_per_1k_input,
            'cost_per_1k_output': model_info.cost_per_1k_output,
            'capabilities': model_info.capabilities,
            'max_tokens': model_info.max_tokens,
            'tier': model_info.tier
        }
        self.config.add_model(model_def)
    
    def remove_model(self, name: str) -> bool:
        """Remove a model from the registry.
        
        Args:
            name: Name of model to remove
            
        Returns:
            True if model was removed, False if not found
        """
        if name in self.models:
            del self.models[name]
            return self.config.remove_model(name)
        return False
    
    def get_providers(self) -> List[str]:
        """Get list of all providers in the registry.
        
        Returns:
            List of unique provider names
        """
        providers = set()
        for model in self.models.values():
            providers.add(model.provider)
        return sorted(list(providers))
    
    def models_by_provider(self, provider: str) -> List[ModelInfo]:
        """Get models from a specific provider.
        
        Args:
            provider: Provider name
            
        Returns:
            List of ModelInfo objects from the provider
        """
        return [
            model for model in self.models.values()
            if model.provider.lower() == provider.lower()
        ]
    
    def estimate_costs(self, tier: str, input_tokens: int, output_tokens: int) -> Dict[str, float]:
        """Estimate costs for all models supporting a tier.
        
        Args:
            tier: Complexity tier
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens
            
        Returns:
            Dictionary mapping model names to estimated costs
        """
        costs = {}
        for model in self.models_for_tier(tier):
            costs[model.name] = model.calculate_cost(input_tokens, output_tokens)
        return costs
    
    def model_comparison(self, tier: str) -> List[Dict[str, Any]]:
        """Get a comparison table for all models supporting a tier.
        
        Args:
            tier: Complexity tier
            
        Returns:
            List of dictionaries with model comparison data
        """
        comparison = []
        for model in self.models_for_tier(tier):
            comparison.append({
                'name': model.name,
                'provider': model.provider,
                'cost_per_1k_input': model.cost_per_1k_input,
                'cost_per_1k_output': model.cost_per_1k_output,
                'avg_cost_per_1k': (model.cost_per_1k_input + model.cost_per_1k_output) / 2,
                'max_tokens': model.max_tokens,
                'capabilities': model.capabilities,
                'supported_tiers': model.tier
            })
        return comparison
    
    def save_to_file(self, file_path: str) -> None:
        """Save registry to a JSON file.
        
        Args:
            file_path: Path to save the registry
        """
        data = {
            'models': []
        }
        
        for model in self.models.values():
            model_data = {
                'name': model.name,
                'provider': model.provider,
                'cost_per_1k_input': model.cost_per_1k_input,
                'cost_per_1k_output': model.cost_per_1k_output,
                'capabilities': model.capabilities,
                'max_tokens': model.max_tokens,
                'tier': model.tier
            }
            data['models'].append(model_data)
        
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=2)
    
    def load_from_file(self, file_path: str) -> None:
        """Load registry from a JSON file.
        
        Args:
            file_path: Path to load the registry from
        """
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        self.models = {}
        for model_def in data.get('models', []):
            model = ModelInfo(
                name=model_def['name'],
                provider=model_def['provider'],
                cost_per_1k_input=model_def['cost_per_1k_input'],
                cost_per_1k_output=model_def['cost_per_1k_output'],
                capabilities=model_def['capabilities'],
                max_tokens=model_def['max_tokens'],
                tier=model_def['tier']
            )
            self.models[model.name] = model