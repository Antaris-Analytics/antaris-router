"""
Configuration management for Antaris Router.

Loads routing rules, model definitions, and classification parameters
from JSON configuration files.
"""

import json
import os
from pathlib import Path
from typing import Dict, Any, List


class Config:
    """Configuration manager for routing rules and model definitions."""
    
    def __init__(self, config_path: str = None):
        """Initialize configuration.
        
        Args:
            config_path: Path to config directory. If None, uses defaults.
        """
        self.config_path = config_path
        self.config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from file or defaults."""
        if self.config_path and os.path.exists(os.path.join(self.config_path, 'config.json')):
            config_file = os.path.join(self.config_path, 'config.json')
            with open(config_file, 'r') as f:
                return json.load(f)
        else:
            # Load defaults from package
            defaults_path = Path(__file__).parent / 'defaults.json'
            with open(defaults_path, 'r') as f:
                return json.load(f)
    
    def get_models(self) -> List[Dict[str, Any]]:
        """Get model definitions."""
        return self.config.get('models', [])
    
    def get_classification_rules(self) -> Dict[str, Any]:
        """Get classification rules and keywords."""
        return self.config.get('classification_rules', {})
    
    def get_trivial_keywords(self) -> List[str]:
        """Get keywords that indicate trivial complexity."""
        return self.get_classification_rules().get('trivial_keywords', [])
    
    def get_simple_keywords(self) -> List[str]:
        """Get keywords that indicate simple complexity."""
        return self.get_classification_rules().get('simple_keywords', [])
    
    def get_moderate_keywords(self) -> List[str]:
        """Get keywords that indicate moderate complexity."""
        return self.get_classification_rules().get('moderate_keywords', [])
    
    def get_complex_keywords(self) -> List[str]:
        """Get keywords that indicate complex tasks."""
        return self.get_classification_rules().get('complex_keywords', [])
    
    def get_expert_keywords(self) -> List[str]:
        """Get keywords that indicate expert-level tasks."""
        return self.get_classification_rules().get('expert_keywords', [])
    
    def get_code_indicators(self) -> List[str]:
        """Get indicators that suggest code-related tasks."""
        return self.get_classification_rules().get('code_indicators', [])
    
    def get_length_thresholds(self) -> Dict[str, int]:
        """Get length thresholds for different complexity tiers."""
        return self.get_classification_rules().get('length_thresholds', {})
    
    def save_config(self, config_path: str) -> None:
        """Save current configuration to file.
        
        Args:
            config_path: Path to config directory
        """
        os.makedirs(config_path, exist_ok=True)
        config_file = os.path.join(config_path, 'config.json')
        from .utils import atomic_write_json
        atomic_write_json(config_file, self.config)
    
    def add_model(self, model_def: Dict[str, Any]) -> None:
        """Add or update a model definition.
        
        Args:
            model_def: Model definition dict with required fields
        """
        models = self.config.get('models', [])
        # Remove existing model with same name
        models = [m for m in models if m.get('name') != model_def.get('name')]
        models.append(model_def)
        self.config['models'] = models
    
    def remove_model(self, model_name: str) -> bool:
        """Remove a model definition.
        
        Args:
            model_name: Name of model to remove
            
        Returns:
            True if model was found and removed, False otherwise
        """
        models = self.config.get('models', [])
        original_count = len(models)
        self.config['models'] = [m for m in models if m.get('name') != model_name]
        return len(self.config['models']) < original_count
    
    def update_classification_rules(self, rules: Dict[str, Any]) -> None:
        """Update classification rules.
        
        Args:
            rules: New classification rules to merge
        """
        current_rules = self.config.get('classification_rules', {})
        current_rules.update(rules)
        self.config['classification_rules'] = current_rules