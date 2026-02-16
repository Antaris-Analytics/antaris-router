#!/usr/bin/env python3
"""Live routing tests for README examples."""

from antaris_router import Router
import json

def main():
    # Initialize router
    router = Router()

    # Test different prompt types with cost tracking
    test_prompts = [
        "Hi there!",
        "What is machine learning?", 
        "Explain the architecture of microservices with load balancing and caching",
        "Implement a React component with TypeScript for user authentication",
        """Design a distributed system architecture for handling 100k concurrent users.
Include database sharding, Redis caching, load balancers, and monitoring.
Consider fault tolerance, auto-scaling, and cost optimization."""
    ]

    print("🧪 LIVE ROUTING TESTS\n")

    total_cost = 0
    baseline_cost = 0  # If we used GPT-4o for everything

    for i, prompt in enumerate(test_prompts, 1):
        decision = router.route(prompt)
        
        # Simulate realistic token usage
        if decision.tier == "trivial":
            in_tokens, out_tokens = 10, 25
        elif decision.tier == "simple":
            in_tokens, out_tokens = 25, 80
        elif decision.tier == "moderate":
            in_tokens, out_tokens = 60, 200
        elif decision.tier == "complex":
            in_tokens, out_tokens = 150, 400
        else:  # expert
            in_tokens, out_tokens = 300, 800
        
        # Calculate actual costs
        actual_cost = router.log_usage(decision, input_tokens=in_tokens, output_tokens=out_tokens)
        total_cost += actual_cost
        
        # Calculate baseline (GPT-4o for everything: $2.50 input, $10.00 output per 1M tokens)
        baseline_cost += (in_tokens * 2.50 + out_tokens * 10.00) / 1000
        
        print(f"Test {i}: {decision.tier.upper()}")
        print(f'  Prompt: "{prompt[:60]}{"..." if len(prompt) > 60 else ""}"')
        print(f"  Model: {decision.model}")
        print(f"  Cost: ${actual_cost:.6f} (vs GPT-4o: ${(in_tokens * 2.50 + out_tokens * 10.00) / 1000:.6f})")
        print(f"  Reasoning: {', '.join(decision.reasoning[:2])}")
        print()

    print("📊 COST ANALYSIS")
    print(f"Total with router: ${total_cost:.4f}")  
    print(f"GPT-4o for everything: ${baseline_cost:.4f}")
    savings_amount = baseline_cost - total_cost
    savings_percent = (savings_amount / baseline_cost * 100) if baseline_cost > 0 else 0
    print(f"Savings: ${savings_amount:.4f} ({savings_percent:.1f}%)")

    # Generate a monthly projection
    monthly_requests = 10000
    monthly_with_router = total_cost * (monthly_requests / len(test_prompts))
    monthly_baseline = baseline_cost * (monthly_requests / len(test_prompts))

    print(f"\n📈 MONTHLY PROJECTION (10,000 requests)")
    print(f"With antaris-router: ${monthly_with_router:.2f}")
    print(f"GPT-4o only: ${monthly_baseline:.2f}") 
    print(f"Monthly savings: ${monthly_baseline - monthly_with_router:.2f}")
    
    # Return data for README
    return {
        "test_results": [
            {
                "prompt": prompt[:60] + ("..." if len(prompt) > 60 else ""),
                "tier": router.route(prompt).tier,
                "model": router.route(prompt).model,
                "cost": router.log_usage(router.route(prompt), 
                                       input_tokens=(10, 25, 60, 150, 300)[i-1], 
                                       output_tokens=(25, 80, 200, 400, 800)[i-1])
            } for i, prompt in enumerate(test_prompts, 1)
        ],
        "monthly_savings": monthly_baseline - monthly_with_router,
        "savings_percent": savings_percent
    }

if __name__ == "__main__":
    main()