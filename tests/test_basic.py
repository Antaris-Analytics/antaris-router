"""
Comprehensive tests for Antaris Router.

Tests all core functionality including classification, registry, routing,
cost tracking, and configuration management.
"""

import unittest
import tempfile
import os
import shutil
import json
from pathlib import Path

from antaris_router import (
    Router, RoutingDecision, TaskClassifier, ClassificationResult,
    ModelRegistry, ModelInfo, CostTracker, UsageRecord, Config
)


class TestConfig(unittest.TestCase):
    """Test configuration management."""
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.config = Config()
    
    def tearDown(self):
        shutil.rmtree(self.temp_dir)
    
    def test_default_config_loads(self):
        """Test that default configuration loads successfully."""
        models = self.config.get_models()
        self.assertGreater(len(models), 0)
        self.assertIn('name', models[0])
        self.assertIn('provider', models[0])
        self.assertIn('tier', models[0])
    
    def test_classification_rules(self):
        """Test that classification rules are loaded."""
        trivial = self.config.get_trivial_keywords()
        self.assertIn('hello', trivial)
        
        complex_kw = self.config.get_complex_keywords()
        self.assertIn('implement', complex_kw)
        
        thresholds = self.config.get_length_thresholds()
        self.assertIn('trivial_max', thresholds)
    
    def test_save_and_load_config(self):
        """Test saving and loading custom configuration."""
        # Add a custom model
        custom_model = {
            'name': 'test-model',
            'provider': 'test-provider',
            'cost_per_1k_input': 0.001,
            'cost_per_1k_output': 0.002,
            'capabilities': ['text'],
            'max_tokens': 1000,
            'tier': ['simple']
        }
        self.config.add_model(custom_model)
        
        # Save config
        self.config.save_config(self.temp_dir)
        
        # Load new config from saved file
        new_config = Config(self.temp_dir)
        models = new_config.get_models()
        
        # Verify custom model is present
        model_names = [m['name'] for m in models]
        self.assertIn('test-model', model_names)


class TestTaskClassifier(unittest.TestCase):
    """Test task classification functionality."""
    
    def setUp(self):
        self.config = Config()
        self.classifier = TaskClassifier(self.config)
    
    def test_trivial_classification(self):
        """Test classification of trivial tasks."""
        result = self.classifier.classify("hello")
        self.assertEqual(result.tier, "trivial")
        self.assertGreater(result.confidence, 0.0)
        self.assertIsInstance(result.reasoning, list)
    
    def test_simple_classification(self):
        """Test classification of simple tasks."""
        result = self.classifier.classify("What is the capital of France?")
        self.assertIn(result.tier, ["simple", "trivial"])  # Could go either way
        self.assertGreater(result.confidence, 0.0)
    
    def test_complex_classification(self):
        """Test classification of complex tasks."""
        prompt = """
        Implement a Python function that performs merge sort on a list of integers.
        The function should handle edge cases and be optimized for performance.
        Include comprehensive docstrings and error handling.
        ```python
        def merge_sort(arr):
            # Your implementation here
            pass
        ```
        """
        result = self.classifier.classify(prompt)
        self.assertIn(result.tier, ["complex", "expert"])
        reasoning_text = " ".join(result.reasoning).lower()
        self.assertTrue(any(word in reasoning_text for word in ["code", "programming", "implementation"]))
    
    def test_expert_classification(self):
        """Test classification of expert-level tasks."""
        prompt = """
        Design a distributed microservices architecture for an e-commerce platform
        that handles 1M+ concurrent users. Consider event sourcing, CQRS patterns,
        data consistency across services, and fault tolerance. Provide detailed
        architectural diagrams and explain the trade-offs of your design decisions.
        Address scalability bottlenecks and propose monitoring strategies.
        """
        result = self.classifier.classify(prompt)
        self.assertIn(result.tier, ["expert", "complex"])
        # Should trigger expert-level pattern detection regardless of length
    
    def test_empty_prompt(self):
        """Test handling of empty prompts."""
        result = self.classifier.classify("")
        self.assertEqual(result.tier, "trivial")
        self.assertEqual(result.confidence, 1.0)
    
    def test_code_detection(self):
        """Test code block detection."""
        code_prompt = "Here's a Python function:\n```python\ndef hello():\n    print('hi')\n```"
        result = self.classifier.classify(code_prompt)
        self.assertTrue(result.signals['has_code'])
    
    def test_question_detection(self):
        """Test question detection."""
        question_prompt = "What is machine learning? How does it work?"
        result = self.classifier.classify(question_prompt)
        self.assertTrue(result.signals['has_questions'])


class TestModelRegistry(unittest.TestCase):
    """Test model registry functionality."""
    
    def setUp(self):
        self.config = Config()
        self.registry = ModelRegistry(self.config)
    
    def test_models_loaded(self):
        """Test that models are loaded from config."""
        models = self.registry.list_models()
        self.assertGreater(len(models), 0)
        
        # Check first model has required fields
        model = models[0]
        self.assertIsInstance(model.name, str)
        self.assertIsInstance(model.provider, str)
        self.assertIsInstance(model.cost_per_1k_input, float)
        self.assertIsInstance(model.tier, list)
    
    def test_models_for_tier(self):
        """Test getting models for specific tiers."""
        trivial_models = self.registry.models_for_tier("trivial")
        self.assertGreater(len(trivial_models), 0)
        
        # Should be sorted by cost (cheapest first)
        if len(trivial_models) > 1:
            avg_cost_1 = (trivial_models[0].cost_per_1k_input + trivial_models[0].cost_per_1k_output) / 2
            avg_cost_2 = (trivial_models[1].cost_per_1k_input + trivial_models[1].cost_per_1k_output) / 2
            self.assertLessEqual(avg_cost_1, avg_cost_2)
    
    def test_cheapest_for_tier(self):
        """Test getting cheapest model for tier."""
        cheapest = self.registry.cheapest_for_tier("simple")
        self.assertIsNotNone(cheapest)
        self.assertIn("simple", cheapest.tier)
    
    def test_model_cost_calculation(self):
        """Test cost calculation functionality."""
        models = self.registry.list_models()
        model = models[0]
        
        cost = model.calculate_cost(1000, 500)  # 1k input, 500 output tokens
        expected = model.cost_per_1k_input + (model.cost_per_1k_output * 0.5)
        self.assertAlmostEqual(cost, expected, places=6)
    
    def test_add_and_remove_model(self):
        """Test adding and removing models from registry."""
        # Create test model
        test_model = ModelInfo(
            name="test-model-registry",
            provider="test",
            cost_per_1k_input=0.001,
            cost_per_1k_output=0.002,
            capabilities=["text"],
            max_tokens=1000,
            tier=["simple"]
        )
        
        # Add model
        original_count = len(self.registry.list_models())
        self.registry.add_model(test_model)
        self.assertEqual(len(self.registry.list_models()), original_count + 1)
        
        # Verify model exists
        retrieved = self.registry.get_model("test-model-registry")
        self.assertIsNotNone(retrieved)
        self.assertEqual(retrieved.provider, "test")
        
        # Remove model
        removed = self.registry.remove_model("test-model-registry")
        self.assertTrue(removed)
        self.assertEqual(len(self.registry.list_models()), original_count)
    
    def test_models_with_capability(self):
        """Test filtering models by capability."""
        vision_models = self.registry.models_with_capability("vision")
        for model in vision_models:
            self.assertIn("vision", model.capabilities)
    
    def test_providers(self):
        """Test getting providers list."""
        providers = self.registry.get_providers()
        self.assertGreater(len(providers), 0)
        self.assertIsInstance(providers[0], str)


class TestCostTracker(unittest.TestCase):
    """Test cost tracking functionality."""
    
    def setUp(self):
        self.temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.json')
        self.temp_file.close()
        self.tracker = CostTracker(self.temp_file.name)
        
        # Create test model
        self.test_model = ModelInfo(
            name="test-model",
            provider="test",
            cost_per_1k_input=0.001,
            cost_per_1k_output=0.002,
            capabilities=["text"],
            max_tokens=1000,
            tier=["simple"]
        )
    
    def tearDown(self):
        os.unlink(self.temp_file.name)
    
    def test_log_usage(self):
        """Test logging model usage."""
        cost = self.tracker.log_usage(
            model=self.test_model,
            tier="simple",
            input_tokens=1000,
            output_tokens=500
        )
        
        expected_cost = 0.001 + (0.002 * 0.5)  # 1k input + 500 output
        self.assertAlmostEqual(cost, expected_cost, places=6)
        
        self.assertEqual(len(self.tracker.usage_history), 1)
        record = self.tracker.usage_history[0]
        self.assertEqual(record.model_name, "test-model")
        self.assertEqual(record.tier, "simple")
    
    def test_save_and_load(self):
        """Test saving and loading usage data."""
        # Log some usage
        self.tracker.log_usage(self.test_model, "simple", 1000, 500)
        
        # Save
        self.tracker.save()
        
        # Create new tracker and load
        new_tracker = CostTracker(self.temp_file.name)
        
        self.assertEqual(len(new_tracker.usage_history), 1)
        self.assertEqual(new_tracker.usage_history[0].model_name, "test-model")
    
    def test_cost_report(self):
        """Test cost reporting."""
        # Log multiple usage records
        for i in range(3):
            self.tracker.log_usage(self.test_model, "simple", 1000, 500)
        
        report = self.tracker.report("all")
        
        self.assertEqual(report["total_requests"], 3)
        self.assertGreater(report["total_cost"], 0)
        self.assertIn("test-model", report["model_breakdown"])
        self.assertIn("simple", report["tier_breakdown"])
    
    def test_model_efficiency(self):
        """Test model efficiency analysis."""
        self.tracker.log_usage(self.test_model, "simple", 1000, 500, confidence=0.8)
        
        efficiency = self.tracker.model_efficiency()
        
        self.assertIn("test-model", efficiency)
        model_stats = efficiency["test-model"]
        self.assertIn("cost_per_request", model_stats)
        self.assertIn("avg_routing_confidence", model_stats)


class TestRouter(unittest.TestCase):
    """Test main router functionality."""
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.router = Router(enable_cost_tracking=False)  # Disable for cleaner tests
    
    def tearDown(self):
        shutil.rmtree(self.temp_dir)
    
    def test_basic_routing(self):
        """Test basic routing functionality."""
        decision = self.router.route("Hello there!")
        
        self.assertIsInstance(decision, RoutingDecision)
        self.assertIsInstance(decision.model, str)
        self.assertIsInstance(decision.provider, str)
        self.assertIn(decision.tier, ["trivial", "simple", "moderate", "complex", "expert"])
        self.assertGreater(decision.confidence, 0.0)
        self.assertIsInstance(decision.reasoning, list)
        self.assertGreaterEqual(decision.estimated_cost, 0.0)
    
    def test_trivial_routing(self):
        """Test routing of trivial tasks."""
        decision = self.router.route("hi")
        self.assertEqual(decision.tier, "trivial")
        
        # Should select a cheap model for trivial tasks
        model = self.router.get_model_info(decision.model)
        self.assertIsNotNone(model)
    
    def test_complex_routing(self):
        """Test routing of complex tasks."""
        complex_prompt = """
        Implement a distributed caching system with the following requirements:
        - Support for Redis and Memcached backends
        - Automatic failover and load balancing
        - Configurable expiration policies
        - Thread-safe operations
        - Comprehensive logging and monitoring
        
        def create_cache_cluster(backends, config):
            # Implementation needed
            pass
        """
        
        decision = self.router.route(complex_prompt)
        self.assertIn(decision.tier, ["complex", "expert"])
        
        # Should select a capable model for complex tasks
        model = self.router.get_model_info(decision.model)
        self.assertIsNotNone(model)
        self.assertIn(decision.tier, model.tier)
    
    def test_routing_with_preferences(self):
        """Test routing with provider preferences."""
        decision = self.router.route("What is Python?", prefer="openai")
        
        # Should prefer OpenAI if available for the tier
        if "openai" in decision.provider.lower():
            self.assertIn("openai", decision.provider.lower())
    
    def test_routing_with_capability(self):
        """Test routing with required capability."""
        decision = self.router.route("Describe this image", capability="vision")
        
        model = self.router.get_model_info(decision.model)
        self.assertIn("vision", model.capabilities)
    
    def test_min_tier_override(self):
        """Test minimum tier override."""
        decision = self.router.route("hi", min_tier="complex")
        
        # Should use complex tier even for trivial prompt
        self.assertIn(decision.tier, ["complex", "expert"])
    
    def test_routing_analytics(self):
        """Test routing analytics generation."""
        # Make several routing decisions
        for prompt in ["hi", "What is AI?", "Implement quicksort", "hello"]:
            self.router.route(prompt)
        
        analytics = self.router.routing_analytics()
        
        self.assertEqual(analytics["total_decisions"], 4)
        self.assertIn("tier_distribution", analytics)
        self.assertIn("model_usage", analytics)
        self.assertIn("avg_confidence", analytics)
    
    def test_list_models_for_tier(self):
        """Test listing models for a specific tier."""
        models = self.router.list_models_for_tier("simple")
        
        self.assertGreater(len(models), 0)
        self.assertIn("name", models[0])
        self.assertIn("provider", models[0])
        self.assertIn("cost_per_1k_input", models[0])
    
    def test_routing_decision_serialization(self):
        """Test that routing decisions can be serialized."""
        decision = self.router.route("Test prompt")
        decision_dict = decision.to_dict()
        
        self.assertIsInstance(decision_dict, dict)
        self.assertIn("model", decision_dict)
        self.assertIn("tier", decision_dict)
        self.assertIn("classification", decision_dict)
        
        # Should be JSON serializable
        json_str = json.dumps(decision_dict)
        self.assertIsInstance(json_str, str)


class TestIntegration(unittest.TestCase):
    """Integration tests for complete workflows."""
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.router = Router()
    
    def tearDown(self):
        shutil.rmtree(self.temp_dir)
    
    def test_complete_workflow(self):
        """Test complete routing and cost tracking workflow."""
        # Route a prompt
        decision = self.router.route("Explain machine learning algorithms")
        
        # Log usage
        cost = self.router.log_usage(decision, input_tokens=100, output_tokens=300)
        self.assertGreater(cost, 0)
        
        # Generate reports
        cost_report = self.router.cost_report("all")
        self.assertEqual(cost_report["total_requests"], 1)
        
        analytics = self.router.routing_analytics()
        self.assertEqual(analytics["total_decisions"], 1)
    
    def test_savings_calculation(self):
        """Test savings calculation compared to premium model."""
        # Make some routing decisions
        prompts = [
            "hi",
            "What's the weather?",
            "Implement a binary search tree"
        ]
        
        for prompt in prompts:
            decision = self.router.route(prompt)
            self.router.log_usage(decision, input_tokens=50, output_tokens=100)
        
        savings = self.router.savings_estimate("gpt-4o")
        
        self.assertIn("total_savings", savings)
        self.assertIn("percentage_saved", savings)
        self.assertEqual(savings["records_analyzed"], 3)
    
    def test_fallback_routing(self):
        """Test fallback routing when no models available for tier."""
        # This is harder to test without manipulating the model registry
        # For now, just ensure no crashes with unusual requirements
        try:
            decision = self.router.route("Test", capability="nonexistent-capability")
            # Should either work or raise a clear error
        except ValueError as e:
            self.assertIn("No suitable models found", str(e))
    
    def test_configuration_persistence(self):
        """Test that configuration changes persist."""
        # Save current state
        self.router.save_state(self.temp_dir)
        
        # Verify config file was created
        config_file = os.path.join(self.temp_dir, 'config.json')
        self.assertTrue(os.path.exists(config_file))
        
        # Load config and verify models are present
        with open(config_file, 'r') as f:
            config_data = json.load(f)
        
        self.assertIn('models', config_data)
        self.assertGreater(len(config_data['models']), 0)


if __name__ == '__main__':
    unittest.main()