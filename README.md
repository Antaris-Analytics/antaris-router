# Antaris Router

**File-based model router for LLM cost optimization. Zero dependencies.**

Route prompts to the cheapest capable model using deterministic keyword matching and structural analysis. No API calls for routing decisions, no vector databases, no infrastructure.

[![Tests](https://img.shields.io/badge/tests-passing-brightgreen)](https://github.com/Antaris-Analytics/antaris-router)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-green.svg)](https://python.org)
[![License](https://img.shields.io/badge/license-Apache%202.0-orange.svg)](LICENSE)

## What It Does

- Classifies prompts into complexity tiers (trivial, simple, moderate, complex, expert) using deterministic rules
- Routes to the **cheapest model** capable of handling each complexity level
- Tracks actual usage costs and provides savings estimates vs. premium models
- Supports provider preferences, capability requirements, and tier overrides
- Runs fully offline — zero network calls, zero API keys, zero model inference for routing

## What It Doesn't Do

- **Not a model proxy** — returns routing decisions, doesn't make API calls
- **Not semantic** — classification uses keyword matching and structural patterns, not embeddings
- **Not adaptive** — routing rules are deterministic and don't learn from outcomes
- **Not a rate limiter** — handles routing logic only, not request management

## Design Goals

| Goal | Rationale |
|------|-----------|
| Deterministic | Same prompt → same routing decision. No model variance. |
| Cost-optimized | Always picks the cheapest model for each complexity tier. |
| Offline | No network, no API keys, no model calls for routing. |
| Transparent | Plain JSON config. Inspect routing rules with any text editor. |
| Zero dependencies | Pure Python standard library. |

## Install

```bash
pip install antaris-router
```

## Quick Start

```python
from antaris_router import Router

# Initialize router
router = Router("./config")

# Route different types of prompts
trivial = router.route("Hello!")
# → RoutingDecision(model="gpt-4o-mini", tier="trivial", estimated_cost=0.0001)

complex_task = router.route("""
Implement a distributed caching system with Redis clustering,
automatic failover, and comprehensive monitoring.
""")
# → RoutingDecision(model="claude-sonnet-3-5", tier="complex", estimated_cost=0.12)

# Track actual usage for cost analysis
actual_cost = router.log_usage(trivial, input_tokens=10, output_tokens=5)
print(f"Actual cost: ${actual_cost:.4f}")

# Generate cost reports
report = router.cost_report(period="week")
print(f"Total cost: ${report['total_cost']:.2f}")
print(f"Savings vs always using GPT-4: ${router.savings_estimate()['total_savings']:.2f}")
```

## Classification Tiers

| Tier | Examples | Typical Models |
|------|----------|----------------|
| **Trivial** | Greetings, yes/no, acknowledgments | gpt-4o-mini, gemini-flash |
| **Simple** | Factual questions, basic explanations | gpt-4o-mini, claude-haiku |
| **Moderate** | Analysis, summarization, planning | claude-haiku, gpt-4o |
| **Complex** | Code generation, architecture, debugging | claude-sonnet, gpt-4o |
| **Expert** | Research synthesis, novel problem solving | claude-opus, gpt-4o |

Classification is based on:
- **Keywords** — domain-specific terms for each complexity level
- **Length** — longer prompts typically indicate higher complexity  
- **Structure** — code blocks, lists, questions increase complexity
- **Context** — optional metadata can influence classification

## Advanced Routing

```python
# Provider preference
decision = router.route("Explain quantum computing", prefer="anthropic")

# Minimum complexity tier
decision = router.route("Hello", min_tier="complex")  # Forces expensive model

# Capability requirements  
decision = router.route("Describe this image", capability="vision")

# Multiple constraints
decision = router.route(
    "Generate React component code",
    prefer="openai", 
    min_tier="moderate",
    capability="code"
)

# Inspect routing reasoning
for reason in decision.reasoning:
    print(f"- {reason}")
```

## Cost Tracking

```python
# Automatic cost tracking (enabled by default)
router = Router()

# Log actual usage
for prompt in ["Hi", "Explain ML", "Write Python code"]:
    decision = router.route(prompt)
    router.log_usage(decision, input_tokens=100, output_tokens=50)

# Analyze costs
report = router.cost_report(period="month")
print(f"Requests: {report['total_requests']}")
print(f"Total cost: ${report['total_cost']:.4f}")

# Compare to always using expensive models
savings = router.savings_estimate(comparison_model="gpt-4o")
print(f"Saved: ${savings['total_savings']:.2f} ({savings['percentage_saved']:.1f}%)")

# Model efficiency analysis
efficiency = router.cost_tracker.model_efficiency()
for model, stats in efficiency.items():
    print(f"{model}: ${stats['cost_per_request']:.4f}/request")
```

## Configuration

All configuration is stored in JSON files. Customize models, costs, and classification rules:

```json
{
  "models": [
    {
      "name": "gpt-4o-mini",
      "provider": "openai", 
      "cost_per_1k_input": 0.00015,
      "cost_per_1k_output": 0.0006,
      "capabilities": ["text", "code", "reasoning"],
      "max_tokens": 128000,
      "tier": ["trivial", "simple", "moderate"]
    }
  ],
  "classification_rules": {
    "trivial_keywords": ["hello", "hi", "thanks", "yes", "no"],
    "complex_keywords": ["implement", "architecture", "algorithm"],
    "length_thresholds": {
      "trivial_max": 50,
      "simple_max": 200,
      "moderate_max": 1000,
      "complex_max": 3000
    }
  }
}
```

## Storage Format

Router state and cost tracking data are stored in JSON:

```json
{
  "version": "1.0.0",
  "saved_at": "2026-02-15T14:30:00",
  "usage_history": [
    {
      "timestamp": "2026-02-15T10:00:00",
      "model_name": "gpt-4o-mini",
      "tier": "simple",
      "input_tokens": 50,
      "output_tokens": 30,
      "actual_cost": 0.0000825,
      "routing_confidence": 0.87
    }
  ]
}
```

## Architecture

```
Router
├── TaskClassifier     — Prompt → complexity tier (trivial...expert)
├── ModelRegistry      — Manage model definitions, costs, capabilities
├── CostTracker        — Log usage, generate reports, calculate savings
└── Config             — Load/save JSON configuration files
```

**Data flow:** `prompt → classify → find cheapest model for tier → return decision`

## Zero Dependencies

Uses only the Python standard library. No external packages, no network calls, no hidden infrastructure requirements.

## Comparison

| | Antaris Router | OpenRouter | LiteLLM | LangChain |
|---|---|---|---|---|
| Routing logic | ✅ File-based | ❌ API required | ❌ API required | ❌ Model calls |
| Cost optimization | ✅ Built-in | ⚠️ Manual | ⚠️ Manual | ❌ |
| Deterministic | ✅ Always | ❌ Load balancing | ❌ Load balancing | ❌ Model variance |
| Offline routing | ✅ | ❌ | ❌ | ❌ |
| Usage tracking | ✅ Local files | ✅ Dashboard | ⚠️ Basic | ❌ |
| Zero setup | ✅ | ❌ API keys | ❌ API keys | ❌ Model setup |

**OpenRouter** and **LiteLLM** are excellent model proxies but require API setup and make network calls for routing decisions. Antaris Router makes routing decisions locally and returns them to your code.

## Model Registry

The router ships with sensible defaults for popular models:

- **OpenAI**: GPT-4o, GPT-4o-mini
- **Anthropic**: Claude Opus, Sonnet, Haiku  
- **Google**: Gemini Pro, Gemini Flash
- **Local**: Llama models (zero cost)

Add your own models:

```python
from antaris_router import ModelInfo

custom_model = ModelInfo(
    name="my-custom-model",
    provider="custom", 
    cost_per_1k_input=0.001,
    cost_per_1k_output=0.002,
    capabilities=["text", "code"],
    max_tokens=4096,
    tier=["simple", "moderate"]
)

router.registry.add_model(custom_model)
```

## Use Cases

- **API cost optimization** — Route expensive tasks to capable models, cheap tasks to efficient models
- **Development workflows** — Use cheap models for iteration, expensive models for production
- **Multi-tenant systems** — Different routing rules per customer tier
- **Batch processing** — Classify and route large document collections efficiently
- **A/B testing** — Compare routing strategies without changing application logic

## License

Licensed under the Apache License 2.0. See [LICENSE](LICENSE) for details.