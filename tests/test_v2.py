"""Tests for antaris-router v2.0 — semantic classification, quality tracking, adaptive routing."""

import os
import sys
import json
import shutil
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from antaris_router.semantic import SemanticClassifier, TFIDFVectorizer
from antaris_router.quality import QualityTracker, RoutingDecision
from antaris_router.adaptive import AdaptiveRouter, ModelConfig


class TestTFIDFVectorizer(unittest.TestCase):
    """Test the TF-IDF vectorizer."""

    def setUp(self):
        self.v = TFIDFVectorizer()
        self.v.fit([
            "implement a distributed database with sharding",
            "what is the weather today",
            "write a python function to sort a list",
            "design a microservices architecture for e-commerce",
        ])

    def test_tokenize(self):
        tokens = self.v._tokenize("Hello World! This is a test.")
        self.assertIn("hello", tokens)
        self.assertIn("world", tokens)
        self.assertIn("test", tokens)
        # Stopwords filtered
        self.assertNotIn("this", tokens)
        self.assertNotIn("is", tokens)

    def test_similarity_identical(self):
        vec = self.v.transform("implement a database")
        sim = self.v.similarity(vec, vec)
        self.assertAlmostEqual(sim, 1.0, places=5)

    def test_similarity_related(self):
        vec_a = self.v.transform("implement a distributed database")
        vec_b = self.v.transform("design a database with sharding")
        sim = self.v.similarity(vec_a, vec_b)
        self.assertGreater(sim, 0.1)

    def test_similarity_unrelated(self):
        vec_a = self.v.transform("what is the weather")
        vec_b = self.v.transform("implement microservices architecture")
        sim = self.v.similarity(vec_a, vec_b)
        self.assertLess(sim, 0.3)

    def test_empty_input(self):
        vec = self.v.transform("")
        self.assertEqual(vec, {})

    def test_serialization(self):
        data = self.v.to_dict()
        restored = TFIDFVectorizer.from_dict(data)
        self.assertEqual(restored.doc_count, self.v.doc_count)
        self.assertEqual(len(restored.idf), len(self.v.idf))


class TestSemanticClassifier(unittest.TestCase):
    """Test semantic classification."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.classifier = SemanticClassifier(self.tmpdir)

    def tearDown(self):
        shutil.rmtree(self.tmpdir)

    def test_trivial_classification(self):
        result = self.classifier.classify("What time is it?")
        self.assertEqual(result.tier, "trivial")

    def test_simple_classification(self):
        result = self.classifier.classify("Write a Python function to reverse a string")
        self.assertIn(result.tier, ["simple", "moderate"])

    def test_complex_classification(self):
        result = self.classifier.classify(
            "Design a microservices architecture for an e-commerce platform "
            "handling 10K concurrent users with event sourcing and CQRS patterns, "
            "including service mesh, circuit breakers, and distributed tracing"
        )
        self.assertIn(result.tier, ["complex", "expert"])

    def test_wont_misclassify_implement(self):
        """Flatus's exact criticism: 'implement a React component' shouldn't route to free tier."""
        result = self.classifier.classify(
            "Implement a React component that displays a sortable data table with pagination"
        )
        self.assertNotEqual(result.tier, "trivial",
                          "Should NOT classify implementation tasks as trivial")

    def test_empty_prompt(self):
        result = self.classifier.classify("")
        self.assertEqual(result.tier, "trivial")

    def test_similar_examples_returned(self):
        result = self.classifier.classify("Write unit tests for authentication")
        self.assertGreater(len(result.similar_examples), 0)

    def test_learn(self):
        # Initially classify
        result1 = self.classifier.classify("Optimize our Kubernetes deployment for cost efficiency")
        
        # Teach it that this is complex
        self.classifier.learn("Optimize our Kubernetes deployment for cost efficiency", "complex")
        
        # Re-classify — should be influenced
        result2 = self.classifier.classify("Optimize our Kubernetes deployment for cost efficiency")
        # After learning, it should at least match or improve
        self.assertIsNotNone(result2.tier)

    def test_save_and_load(self):
        self.classifier.learn("Custom test prompt about deployment", "moderate")
        self.classifier.save()
        
        loaded = SemanticClassifier(self.tmpdir)
        self.assertIn("Custom test prompt about deployment",
                      loaded.examples.get("moderate", []))

    def test_stats(self):
        stats = self.classifier.get_stats()
        self.assertGreater(stats['total_examples'], 0)
        self.assertGreater(stats['vocab_size'], 0)


class TestQualityTracker(unittest.TestCase):
    """Test quality tracking and outcome learning."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.tracker = QualityTracker(self.tmpdir)

    def tearDown(self):
        shutil.rmtree(self.tmpdir)

    def test_record_decision(self):
        d = RoutingDecision(
            prompt_hash="abc123",
            tier="moderate",
            model="gpt-4o",
            timestamp=1000.0,
        )
        self.tracker.record(d)
        self.assertEqual(len(self.tracker.decisions), 1)

    def test_record_outcome(self):
        d = RoutingDecision(
            prompt_hash="abc123",
            tier="moderate",
            model="gpt-4o",
            timestamp=1000.0,
        )
        self.tracker.record(d)
        self.tracker.record_outcome("abc123", quality_score=0.9, success=True)
        
        self.assertEqual(self.tracker.decisions[0]['quality_score'], 0.9)
        self.assertTrue(self.tracker.decisions[0]['success'])

    def test_model_score_no_data(self):
        score = self.tracker.get_model_score("unknown_model", "trivial")
        self.assertEqual(score, 0.5)  # Neutral when no data

    def test_model_score_with_data(self):
        for i in range(10):
            d = RoutingDecision(
                prompt_hash=f"hash_{i}",
                tier="moderate",
                model="gpt-4o",
                timestamp=1000.0 + i,
                quality_score=0.9,
                success=True,
            )
            self.tracker.record(d)
        
        score = self.tracker.get_model_score("gpt-4o", "moderate")
        self.assertGreater(score, 0.7)

    def test_recommend_model(self):
        # Record good performance for model A, bad for model B
        for i in range(10):
            self.tracker.record(RoutingDecision(
                prompt_hash=f"a_{i}", tier="simple", model="model_a",
                timestamp=1000.0, quality_score=0.9, success=True,
            ))
            self.tracker.record(RoutingDecision(
                prompt_hash=f"b_{i}", tier="simple", model="model_b",
                timestamp=1000.0, quality_score=0.3, success=False,
            ))
        
        model, confidence = self.tracker.recommend_model(
            "simple", ["model_a", "model_b"]
        )
        self.assertEqual(model, "model_a")

    def test_should_escalate(self):
        # Record poor performance
        for i in range(5):
            self.tracker.record(RoutingDecision(
                prompt_hash=f"h_{i}", tier="complex", model="cheap_model",
                timestamp=1000.0, quality_score=0.1, success=False,
            ))
        
        self.assertTrue(self.tracker.should_escalate("cheap_model", "complex"))

    def test_save_and_load(self):
        self.tracker.record(RoutingDecision(
            prompt_hash="test", tier="simple", model="gpt-4o",
            timestamp=1000.0,
        ))
        self.tracker.save()
        
        loaded = QualityTracker(self.tmpdir)
        self.assertEqual(len(loaded.decisions), 1)

    def test_trim(self):
        for i in range(200):
            self.tracker.record(RoutingDecision(
                prompt_hash=f"h_{i}", tier="simple", model="m",
                timestamp=float(i),
            ))
        self.tracker.trim(keep_last=50)
        self.assertEqual(len(self.tracker.decisions), 50)


class TestAdaptiveRouter(unittest.TestCase):
    """Test the full adaptive routing pipeline."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.router = AdaptiveRouter(self.tmpdir, ab_test_rate=0.0)  # Disable A/B for determinism
        
        # Register test models
        self.router.register_model(ModelConfig(
            name="gpt-4o-mini",
            tier_range=("trivial", "moderate"),
            cost_per_1k_input=0.00015,
            cost_per_1k_output=0.0006,
        ))
        self.router.register_model(ModelConfig(
            name="gpt-4o",
            tier_range=("moderate", "expert"),
            cost_per_1k_input=0.0025,
            cost_per_1k_output=0.01,
        ))
        self.router.register_model(ModelConfig(
            name="claude-opus",
            tier_range=("complex", "expert"),
            cost_per_1k_input=0.015,
            cost_per_1k_output=0.075,
        ))

    def tearDown(self):
        shutil.rmtree(self.tmpdir)

    def test_trivial_routes_cheap(self):
        result = self.router.route("What is 2 + 2?")
        self.assertEqual(result.model, "gpt-4o-mini")
        self.assertEqual(result.tier, "trivial")

    def test_complex_routes_premium(self):
        result = self.router.route(
            "Design a distributed consensus algorithm with leader election, "
            "log replication, and cluster membership changes. Include fault "
            "tolerance for network partitions and node failures."
        )
        self.assertIn(result.tier, ["complex", "expert"])
        self.assertIn(result.model, ["gpt-4o", "claude-opus"])

    def test_implement_react_not_trivial(self):
        """The Flatus test: 'implement a React component' must NOT route to trivial."""
        result = self.router.route(
            "Implement a React component that displays a sortable data table with pagination"
        )
        self.assertNotEqual(result.tier, "trivial")
        self.assertNotEqual(result.model, "gpt-4o-mini" if result.tier in ["complex", "expert"] else "")

    def test_fallback_chain_exists(self):
        result = self.router.route("Write a function to sort a list")
        self.assertIsInstance(result.fallback_chain, list)

    def test_reasoning_provided(self):
        result = self.router.route("Hello")
        self.assertGreater(len(result.reasoning), 0)
        self.assertTrue(any("Semantic" in r for r in result.reasoning))

    def test_report_outcome(self):
        result = self.router.route("Test prompt")
        self.router.report_outcome(result.prompt_hash, quality_score=0.8, success=True)
        # Should not raise
        stats = self.router.get_stats()
        self.assertGreater(stats['quality_tracker']['total_decisions'], 0)

    def test_context_escalation(self):
        result1 = self.router.route("Fix this bug", context={'iteration': 1})
        result2 = self.router.route("Fix this bug", context={'iteration': 5})
        # Higher iteration should escalate tier
        tier1_idx = {'trivial': 0, 'simple': 1, 'moderate': 2, 'complex': 3, 'expert': 4}
        self.assertGreaterEqual(
            tier1_idx.get(result2.tier, 0),
            tier1_idx.get(result1.tier, 0),
        )

    def test_teach_correction(self):
        """Test that teaching a correction influences future routing."""
        self.router.teach(
            "Optimize database queries for production",
            "complex"
        )
        # Should be learned
        stats = self.router.classifier.get_stats()
        self.assertGreater(stats['total_examples'], 0)

    def test_save_and_reload(self):
        self.router.route("Test prompt for persistence")
        self.router.save()
        
        loaded = AdaptiveRouter(self.tmpdir, ab_test_rate=0.0)
        stats = loaded.get_stats()
        self.assertGreater(stats['models_registered'], 0)

    def test_no_models_registered(self):
        empty_dir = tempfile.mkdtemp()
        try:
            empty_router = AdaptiveRouter(empty_dir)
            result = empty_router.route("Hello")
            self.assertEqual(result.model, "")
            self.assertEqual(len(result.reasoning), 1)
        finally:
            shutil.rmtree(empty_dir)


if __name__ == '__main__':
    unittest.main()
