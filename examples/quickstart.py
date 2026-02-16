#!/usr/bin/env python3
"""
Antaris Router Quickstart Example

Demonstrates core routing functionality, cost tracking, and analytics.
"""

from antaris_router import Router


def main():
    """Run quickstart demonstration."""
    print("=== Antaris Router Quickstart ===\n")
    
    # Initialize router
    print("1. Initializing router...")
    router = Router()
    print(f"   Loaded {len(router.registry.list_models())} models")
    print(f"   Available providers: {', '.join(router.registry.get_providers())}")
    print()
    
    # Test different types of prompts
    test_prompts = [
        ("Trivial", "Hello!"),
        ("Simple", "What is the capital of France?"),
        ("Moderate", "Explain the differences between REST and GraphQL APIs"),
        ("Complex", """
Implement a Python class for a thread-safe LRU cache with the following requirements:
- Generic key-value storage
- Configurable maximum size
- Thread-safe operations using locks
- O(1) get and put operations
- Automatic eviction of least recently used items

class LRUCache:
    def __init__(self, capacity: int):
        pass
    
    def get(self, key):
        pass
    
    def put(self, key, value):
        pass
"""),
        ("Expert", """
Design a distributed system architecture for a real-time collaborative document editor 
(like Google Docs) that can handle 100,000+ concurrent users. Your design should address:

1. Real-time synchronization and conflict resolution
2. Scalable data storage and retrieval
3. Operational transformations for concurrent edits
4. Fault tolerance and disaster recovery
5. Geographic distribution and latency optimization
6. Security and access control
7. Performance monitoring and analytics

Provide detailed architectural diagrams, technology choices, and explain the 
trade-offs in your design decisions. Consider both the initial implementation 
and scaling strategies for growth.
""")
    ]
    
    routing_results = []
    
    print("2. Testing routing for different prompt types...")
    for prompt_type, prompt in test_prompts:
        print(f"\n   {prompt_type} prompt:")
        print(f"   Input: {prompt[:80]}{'...' if len(prompt) > 80 else ''}")
        
        # Route the prompt
        decision = router.route(prompt)
        routing_results.append((prompt_type, decision, prompt))
        
        print(f"   → Model: {decision.model} ({decision.provider})")
        print(f"   → Tier: {decision.tier} (confidence: {decision.confidence:.2f})")
        print(f"   → Estimated cost: ${decision.estimated_cost:.4f}")
        print(f"   → Reasoning: {decision.reasoning[-1]}")  # Last reasoning point
        
        # Simulate usage logging with realistic token counts
        token_estimates = {
            "Trivial": (10, 5),
            "Simple": (50, 30),
            "Moderate": (200, 150),
            "Complex": (800, 400),
            "Expert": (1500, 800)
        }
        
        input_tokens, output_tokens = token_estimates[prompt_type]
        actual_cost = router.log_usage(decision, input_tokens, output_tokens)
        print(f"   → Actual cost: ${actual_cost:.4f}")
    
    print("\n" + "="*60)
    
    # Show routing analytics
    print("\n3. Routing Analytics:")
    analytics = router.routing_analytics()
    
    print(f"   Total decisions: {analytics['total_decisions']}")
    print(f"   Average confidence: {analytics['avg_confidence']:.2f}")
    print(f"   Most used model: {analytics['most_used_model']}")
    print(f"   Most used provider: {analytics['most_used_provider']}")
    
    print(f"\n   Tier distribution:")
    for tier, count in analytics['tier_distribution'].items():
        percentage = analytics['tier_percentages'][tier]
        print(f"     {tier}: {count} ({percentage}%)")
    
    # Show cost report
    print("\n4. Cost Report:")
    cost_report = router.cost_report("all")
    
    print(f"   Total cost: ${cost_report['total_cost']:.4f}")
    print(f"   Average per request: ${cost_report['avg_cost_per_request']:.4f}")
    print(f"   Total tokens: {cost_report['total_input_tokens']:,} in, {cost_report['total_output_tokens']:,} out")
    
    print(f"\n   Cost by model:")
    for model, data in cost_report['model_breakdown'].items():
        print(f"     {model}: ${data['total_cost']:.4f} ({data['requests']} requests)")
    
    # Show savings estimate
    print("\n5. Savings Analysis:")
    savings = router.savings_estimate("gpt-4o")
    
    if "error" not in savings:
        print(f"   Cost with smart routing: ${savings['actual_cost']:.4f}")
        print(f"   Cost if always used GPT-4o: ${savings['comparison_cost']:.4f}")
        print(f"   Total savings: ${savings['total_savings']:.4f} ({savings['percentage_saved']:.1f}%)")
    else:
        print(f"   {savings['error']}")
    
    # Demonstrate model registry features
    print("\n6. Model Registry Features:")
    print(f"\n   Models available for 'simple' tasks:")
    simple_models = router.list_models_for_tier("simple")
    for i, model in enumerate(simple_models[:3]):  # Show top 3
        print(f"     {i+1}. {model['name']} ({model['provider']}) - "
              f"${model['cost_per_1k_input']:.4f}/${model['cost_per_1k_output']:.4f} per 1K tokens")
    
    # Show capability-based filtering
    print(f"\n   Models with vision capability:")
    vision_models = router.registry.models_with_capability("vision")
    if vision_models:
        for model in vision_models[:3]:
            print(f"     - {model.name} ({model.provider})")
    else:
        print("     No vision-capable models found in current registry")
    
    # Demonstrate advanced routing
    print("\n7. Advanced Routing Features:")
    
    # Provider preference
    print(f"\n   Routing with provider preference (OpenAI):")
    openai_decision = router.route("Explain quantum computing", prefer="openai")
    print(f"     → {openai_decision.model} ({openai_decision.provider})")
    
    # Minimum tier requirement
    print(f"\n   Routing with minimum tier (complex for simple prompt):")
    complex_decision = router.route("What's 2+2?", min_tier="complex")
    print(f"     → {complex_decision.model} (tier: {complex_decision.tier})")
    simple_cost = router.route("What's 2+2?").estimated_cost
    print(f"     → Cost difference: simple would be ~${simple_cost:.4f}, "
          f"complex is ${complex_decision.estimated_cost:.4f}")
    
    # Capability requirement
    if vision_models:
        print(f"\n   Routing with capability requirement (vision):")
        vision_decision = router.route("Describe this image", capability="vision")
        print(f"     → {vision_decision.model} (has vision capability)")
    
    print("\n8. Classification Details:")
    print(f"\n   Let's examine how the complex prompt was classified:")
    complex_prompt = test_prompts[3][1]  # The LRU cache implementation prompt
    classification = router.classifier.classify(complex_prompt)
    
    print(f"     Tier: {classification.tier}")
    print(f"     Confidence: {classification.confidence:.2f}")
    print(f"     Signals detected:")
    print(f"       - Length: {classification.signals['length']} characters")
    print(f"       - Has code: {classification.signals['has_code']}")
    print(f"       - Structural complexity: {classification.signals['structural_complexity']}")
    print(f"       - Code indicators: {classification.signals['code_indicators']}")
    
    for tier, matches in classification.signals['keyword_matches'].items():
        if matches > 0:
            print(f"       - {tier.title()} keywords: {matches}")
    
    print(f"\n     Full reasoning:")
    for reason in classification.reasoning:
        print(f"       - {reason}")
    
    print("\n" + "="*60)
    print("Quickstart complete! 🚀")
    print("\nKey takeaways:")
    print("- Antaris Router automatically selects the cheapest model for each task")
    print("- Classification is deterministic and based on keywords + structure")
    print("- Cost tracking helps you monitor and optimize LLM expenses")
    print("- No API calls needed for routing decisions - pure file-based logic")
    print("\nNext steps:")
    print("- Customize the model registry for your specific needs")
    print("- Adjust classification rules in the config")
    print("- Integrate with your LLM client code")
    print("- Run regular cost reports to track savings")


if __name__ == "__main__":
    main()