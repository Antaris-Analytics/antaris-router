# Antaris Router

**Adaptive model router for LLM applications. Learns from outcomes. Zero dependencies.**

Routes prompts to the cheapest capable model using semantic classification (TF-IDF), not keyword matching. Tracks outcomes to learn which models actually perform well on which tasks. All state stored in plain JSON files.

[![PyPI](https://img.shields.io/pypi/v/antaris-router)](https://pypi.org/project/antaris-router/)
[![Tests](https://img.shields.io/badge/tests-67%20passing-brightgreen)](https://github.com/Antaris-Analytics/antaris-router)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-green.svg)](https://python.org)
[![License](https://img.shields.io/badge/license-Apache%202.0-orange.svg)](LICENSE)

## What It Does

- **Semantic classification** — TF-IDF vectors + cosine similarity, not keyword matching
- **Outcome learning** — tracks routing decisions and their results, builds per-model quality profiles
- **Fallback chains** — automatic escalation when cheap models fail
- **A/B testing** — routes a configurable % to premium models to validate cheap routing
- **Context-aware** — adjusts routing based on iteration count, conversation length, user expertise
- **Multi-objective** — optimize for quality, cost, speed, or balanced
- Runs fully offline — zero network calls, zero tokens, zero API keys

## Demo

```
Prompt                                                          Tier       Model
──────────────────────────────────────────────────────────────────────────────────
What is 2 + 2?                                                  trivial    gpt-4o-mini
Translate hello to French                                       trivial    gpt-4o-mini
Write a Python function to reverse a string                     simple     gpt-4o-mini
Implement a React component with sortable table and pagination  moderate   claude-sonnet
Write a class that manages a connection pool with retry logic   moderate   claude-sonnet
Design microservices for e-commerce with 10K users and CQRS     complex    claude-sonnet
Fix this bug (iteration 1)                                      trivial    gpt-4o-mini
Fix this bug (iteration 5)                                      simple     claude-sonnet
```

Note: "Implement a React component" routes to **moderate**, not trivial. The semantic classifier understands that implementation tasks require real capability, regardless of prompt length.

## Performance

```
Routing latency (100 calls): median 0.04ms, p99 0.17ms
Classification: ~50 seed examples across 5 tiers, TF-IDF with cosine similarity
Memory: <5MB for typical workloads
Storage: 3 JSON files (examples, model, decisions)
```

Measured on Apple M4, Python 3.14. Your numbers will vary.

## What It Doesn't Do

- **Not a proxy** — doesn't forward requests to models. It tells you *which* model to use.
- **Not semantic search** — no embeddings, no vector DB. Uses TF-IDF (bag-of-words with term weighting).
- **Not real-time market data** — doesn't track live model pricing or availability.
- **Classification is statistical, not perfect** — edge cases exist. Use `teach()` to correct them.
- **Quality tracking requires your feedback** — call `report_outcome()` after using the model, or the router can't learn.

## Install

```bash
pip install antaris-router
```

## Quick Start

```python
from antaris_router import AdaptiveRouter, ModelConfig

router = AdaptiveRouter("./routing_data")

# Register your models with their capability ranges
router.register_model(ModelConfig(
    name="gpt-4o-mini",
    tier_range=("trivial", "moderate"),
    cost_per_1k_input=0.00015,
    cost_per_1k_output=0.0006,
))
router.register_model(ModelConfig(
    name="claude-sonnet",
    tier_range=("simple", "complex"),
    cost_per_1k_input=0.003,
    cost_per_1k_output=0.015,
))
router.register_model(ModelConfig(
    name="claude-opus",
    tier_range=("complex", "expert"),
    cost_per_1k_input=0.015,
    cost_per_1k_output=0.075,
))

# Route a prompt
result = router.route("Implement a distributed task queue with priority scheduling")
print(f"Use {result.model} (tier: {result.tier}, confidence: {result.confidence:.2f})")
# → Use claude-sonnet (tier: complex, confidence: 0.50)

# Report outcome so the router learns
router.report_outcome(result.prompt_hash, quality_score=0.9, success=True)
```

## Outcome Learning

The router gets smarter over time. When a cheap model consistently fails on a task type, the router learns to skip it.

```python
# Initial: routes to cheap model
result = router.route("Write a regex to validate emails")
# → cheap (score: 0.50)

# After reporting 5 failures on cheap:
router.report_outcome(hash, quality_score=0.15, success=False)
# ... repeat ...

# After reporting 4 successes on premium:
# → premium (cheap score: 0.28, premium score: 0.89)
```

Quality scores are computed per model per tier:
```
score = 0.4 × success_rate + 0.4 × avg_quality + 0.2 × (1 - escalation_rate)
```

Models below the escalation threshold (default 0.30) are automatically skipped.

## Context-Aware Routing

Pass context to influence routing decisions:

```python
# First attempt — routes normally
result = router.route("Fix this bug", context={"iteration": 1})
# → trivial → cheap model

# Fifth attempt — escalates (user is struggling)
result = router.route("Fix this bug", context={"iteration": 5})
# → simple → better model

# Long conversation — minimum moderate
result = router.route("What do you think?", context={"conversation_length": 15})

# Expert user — don't waste time with weak models
result = router.route("Optimize this", context={"user_expertise": "expert"})
```

## A/B Testing

Validate cheap routing by occasionally sending requests to premium:

```python
router = AdaptiveRouter("./data", ab_test_rate=0.05)  # 5% of requests

# 95% of trivial/simple/moderate prompts route normally
# 5% route to the premium model for quality comparison
# Track outcomes to confirm cheap routing is actually good enough
```

## Fallback Chains

Every routing result includes a fallback chain:

```python
result = router.route("Write unit tests for authentication")
print(result.model)           # → gpt-4o-mini
print(result.fallback_chain)  # → ['claude-sonnet', 'claude-opus']

# If the primary model fails:
next_model = router.escalate(result.prompt_hash)
print(next_model)  # → claude-sonnet
```

## Teaching Corrections

When the classifier gets it wrong, teach it:

```python
# Classifier thinks this is simple, but it's actually complex
router.teach(
    "Optimize our Kubernetes deployment for cost efficiency",
    "complex"
)
# Correction is learned permanently and improves future classifications
```

## Semantic Classification

The classifier uses TF-IDF (term frequency-inverse document frequency) with cosine similarity, not keyword matching.

**How it works:**
1. Ships with ~50 labeled examples across 5 tiers (trivial → expert)
2. Builds TF-IDF vectors from the example corpus
3. For each new prompt, computes similarity to all examples
4. Scores each tier by average similarity to its top-3 closest examples
5. Applies structural adjustments (long prompts can't be trivial, code presence boosts tier)

**Why not embeddings?** Embeddings would be better but require either an API call (defeats offline goal) or a model file (~100MB+). TF-IDF gets 80% of the benefit with zero dependencies.

## Storage Format

```
routing_data/
├── routing_examples.json    # Labeled examples (seed + learned)
├── routing_model.json       # TF-IDF model (IDF weights, vocab)
├── routing_decisions.json   # Decision history for outcome learning
├── model_profiles.json      # Per-model per-tier quality scores
└── router_config.json       # Model registry and settings
```

All plain JSON. Inspect or edit with any text editor.

## Architecture

```
AdaptiveRouter
├── SemanticClassifier
│   └── TFIDFVectorizer     — Term weighting + cosine similarity
├── QualityTracker
│   ├── RoutingDecision      — Decision + outcome records
│   └── ModelProfiles        — Per-model per-tier quality scores
├── ContextAdjuster          — Iteration, conversation, expertise signals
├── FallbackChain            — Ordered model escalation
└── ABTester                 — Validation routing (configurable %)
```

**Routing flow:** `prompt → semantic classify → context adjust → quality filter → select model → record decision → return`

## Tiers

| Tier | Description | Examples |
|------|-------------|----------|
| trivial | One-line answers, lookups | "What is 2+2?", "Define photosynthesis" |
| simple | Short tasks, basic code | "Reverse a string", "Explain TCP vs UDP" |
| moderate | Multi-step implementation | "Build a REST API with auth", "Write a caching layer" |
| complex | Architecture, multi-system | "Design microservices for e-commerce", "Build a custom ORM" |
| expert | Full system design | "Architect a globally distributed database", "Design HFT platform" |

## Comparison

| | Antaris Router | OpenRouter | LiteLLM | RouteLLM |
|---|---|---|---|---|
| Classification | TF-IDF semantic | API proxy | API proxy | Embeddings |
| Learning | ✅ Outcome tracking | ❌ | ❌ | ✅ |
| Offline | ✅ | ❌ | ❌ | ❌ |
| A/B testing | ✅ Built-in | ❌ | ❌ | ❌ |
| Context-aware | ✅ | ❌ | ❌ | ❌ |
| Fallback chains | ✅ | ✅ | ✅ | ❌ |
| Zero dependencies | ✅ | ❌ | ❌ | ❌ |
| Infrastructure | None | Cloud | Cloud | GPU/API |

**Honest assessment:** OpenRouter and LiteLLM are API proxies that do more (actual request forwarding, billing, rate limiting). Antaris Router is a classification library — it tells you which model to use, not how to call it. Different tools for different needs.

## Legacy API

The v1 keyword-based router is still available:

```python
from antaris_router import Router  # v1 API
router = Router("./config")
decision = router.route("What's 2+2?")
```

We recommend migrating to `AdaptiveRouter` for better classification accuracy.

## Part of the Antaris Analytics Suite

- **[antaris-memory](https://pypi.org/project/antaris-memory/)** — Persistent memory for AI agents
- **antaris-router** — Adaptive model routing (this package)
- **antaris-guard** — Security and prompt injection detection (coming soon)
- **antaris-context** — Context window optimization (coming soon)

## License

Licensed under the Apache License 2.0. See [LICENSE](LICENSE) for details.
