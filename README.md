# Antaris Router

**Adaptive model router for LLM cost optimization. Learns from outcomes. Zero dependencies.**

Routes prompts to the cheapest capable model using semantic classification (TF-IDF), not keyword matching. Tracks outcomes to learn which models actually perform well on which tasks. Enforces cost/latency SLAs. All state stored in plain JSON files. No API keys, no vector database, no infrastructure.

[![PyPI](https://img.shields.io/pypi/v/antaris-router)](https://pypi.org/project/antaris-router/)
[![Tests](https://github.com/Antaris-Analytics/antaris-router/actions/workflows/tests.yml/badge.svg)](https://github.com/Antaris-Analytics/antaris-router/actions/workflows/tests.yml)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-green.svg)](https://python.org)
[![License](https://img.shields.io/badge/license-Apache%202.0-orange.svg)](LICENSE)

## What's New in v3.0.0

- **SLA Monitor** — enforce cost budgets and latency targets per model/tier; `SLAConfig(max_latency_ms=..., budget_per_hour_usd=...)`, `get_sla_report()`, `check_budget_alert()`
- **Confidence Routing** — `RoutingDecision.confidence_basis` for cross-package tracing; `ConfidenceRouter` for score-weighted decisions
- **Suite integration** — router hints consumed by `antaris-context` via `set_router_hints()` for adaptive context budget allocation
- **Backward compatibility** — all Sprint 7 SLA params optional; safe defaults throughout; existing `AdaptiveRouter` code unchanged
- 194 tests (all passing)

See [CHANGELOG.md](CHANGELOG.md) for full version history.

## What It Does

- **Semantic classification** — TF-IDF vectors + cosine similarity, not keyword matching
- **Outcome learning** — tracks routing decisions and their results, builds per-model quality profiles
- **SLA enforcement** — cost budget alerts, latency targets, quality score tracking per model/tier
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
Architect a globally distributed database with CRDTs            expert     claude-opus
```

Note: "Implement a React component" routes to **moderate**, not trivial. The semantic classifier understands that implementation tasks require real capability, regardless of prompt length.

## Performance

```
Routing latency: median 0.05ms, p99 0.09ms, avg 0.05ms (cold start)
Classification: ~50 seed examples across 5 tiers, TF-IDF with cosine similarity
Memory: <5MB for typical workloads
Storage: plain JSON files
```

Measured on Apple M4, Python 3.14.

## Install

```bash
pip install antaris-router
```

## Quick Start — AdaptiveRouter (v2/v3 API)

```python
from antaris_router import AdaptiveRouter, ModelConfig

router = AdaptiveRouter("./routing_data", ab_test_rate=0.05)

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

# Save state
router.save()
```

## SLA Enforcement (v3.0)

```python
from antaris_router import Router, SLAConfig

sla = SLAConfig(
    max_latency_ms=200,
    budget_per_hour_usd=5.00,
    min_quality_score=0.7,
    auto_escalate_on_breach=True,
)

router = Router(
    sla=sla,
    fallback_chain=["claude-sonnet", "claude-haiku"],
)

result = router.route("Summarize this document", auto_scale=True)

# SLA reporting
report = router.get_sla_report(since_hours=1.0)
print(f"Budget used: ${report['budget_used']:.2f} / ${report['budget_limit']:.2f}")
print(f"Avg latency: {report['avg_latency_ms']:.1f}ms")

# Budget alerts
alert = router.check_budget_alert()
if alert['triggered']:
    print(f"⚠️ Budget alert: {alert['message']}")
```

## Outcome Learning

The router gets smarter over time. When a cheap model consistently fails on a task type, the router learns to skip it.

```python
# Initial routing
result = router.route("Write a regex to validate emails")
# → cheap model

# After reporting failures on cheap:
router.report_outcome(result.prompt_hash, quality_score=0.15, success=False)
# ... repeat a few times ...

# Router learns to route this task type to a better model automatically
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

## Fallback Chains

```python
result = router.route("Write unit tests for authentication")
print(result.model)           # → gpt-4o-mini
print(result.fallback_chain)  # → ['claude-sonnet', 'claude-opus']

# Escalate if primary model fails:
next_model = router.escalate(result.prompt_hash)
print(next_model)  # → claude-sonnet
```

## Teaching Corrections

```python
# Classifier thinks this is simple, but it's actually complex
router.teach(
    "Optimize our Kubernetes deployment for cost efficiency",
    "complex"
)
# Correction is learned permanently
```

## Routing Analytics

```python
analytics = router.routing_analytics()
print(f"Total decisions: {analytics['total_decisions']}")
print(f"Tier distribution: {analytics['tier_distribution']}")
print(f"Cost saved vs all-premium: ${analytics['cost_savings']:.2f}")
```

## Works With Local Models (Ollama)

Register Ollama models at `$0.00` cost. The router sends trivial/simple work to local models and reserves cloud APIs for tasks that need them.

```python
router = AdaptiveRouter("./routing_data")

router.register_model(ModelConfig(
    name="qwen3-8b-local",       # Ollama — $0/request
    tier_range=("trivial", "simple"),
    cost_per_1k_input=0.0,
    cost_per_1k_output=0.0,
))
router.register_model(ModelConfig(
    name="claude-sonnet-4",      # Cloud — moderate/complex
    tier_range=("simple", "complex"),
    cost_per_1k_input=0.003,
    cost_per_1k_output=0.015,
))
```

**40% of typical requests route to local models ($0.00).** At 1,000 requests/day, that's ~$10.80/day vs ~$18.00/day all-Sonnet.

The router doesn't call models — it tells you which one to use. Wire it to Ollama's API, LiteLLM, or any client you prefer.

## Semantic Classification

TF-IDF with cosine similarity across ~50 labeled examples, not keyword matching.

**Why not embeddings?** Embeddings would be more accurate but require either an API call (defeats offline goal) or a model file (~100MB+). TF-IDF gets 80% of the benefit with zero dependencies.

## Tiers

| Tier | Description | Examples |
|------|-------------|----------|
| trivial | One-line answers, lookups | "What is 2+2?", "Define photosynthesis" |
| simple | Short tasks, basic code | "Reverse a string", "Explain TCP vs UDP" |
| moderate | Multi-step implementation | "Build a REST API with auth", "Write a caching layer" |
| complex | Architecture, multi-system | "Design microservices for e-commerce", "Build a custom ORM" |
| expert | Full system design | "Architect a globally distributed database", "Design HFT platform" |

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
AdaptiveRouter (v2/v3)
├── SemanticClassifier
│   └── TFIDFVectorizer     — Term weighting + cosine similarity
├── QualityTracker
│   ├── RoutingDecision     — Decision + outcome records
│   └── ModelProfiles       — Per-model per-tier quality scores
├── ContextAdjuster         — Iteration, conversation, expertise signals
├── FallbackChain           — Ordered model escalation
└── ABTester                — Validation routing (configurable %)

Router (v1/v3 with SLA)
├── TaskClassifier          — Keyword-based + structural classification
├── ModelRegistry           — Model capabilities and cost data
├── CostTracker             — Usage records, savings analysis
├── SLAMonitor              — Budget alerts, latency enforcement
└── ConfidenceRouter        — Score-weighted routing decisions
```

## What It Doesn't Do

- **Not a proxy** — doesn't forward requests to models. It tells you *which* model to use.
- **Not semantic search** — no embeddings, no vector DB. Uses TF-IDF (bag-of-words with term weighting).
- **Not real-time market data** — doesn't track live model pricing or availability.
- **Classification is statistical, not perfect** — edge cases exist. Use `teach()` to correct them.
- **Quality tracking requires your feedback** — call `report_outcome()` after using the model, or the router can't learn.

## Legacy API

The v1 keyword-based router is still available and fully supported:

```python
from antaris_router import Router  # v1 API (now with v3 SLA features)
router = Router(config_path="./config")
decision = router.route("What's 2+2?")
```

We recommend `AdaptiveRouter` for new code. Migrate when you're ready — there's no rush.

## Running Tests

```bash
git clone https://github.com/Antaris-Analytics/antaris-router.git
cd antaris-router
pip install pytest
python -m pytest tests/ -v
```

All 194 tests pass with zero external dependencies (pytest is the only test runner needed).

## Why Not OpenRouter / LiteLLM?

OpenRouter and LiteLLM are API proxies — they forward requests, handle billing, and rate limit. Antaris Router is a classification library — it tells you which model to use, not how to call it. Zero dependencies, no API keys, fully offline. Use both together if you want: Antaris Router decides, LiteLLM sends.

| | Antaris Router | OpenRouter | LiteLLM | RouteLLM |
|---|---|---|---|---|
| Classification | TF-IDF semantic | API proxy | API proxy | Embeddings |
| Outcome learning | ✅ | ❌ | ❌ | ✅ |
| SLA enforcement | ✅ | ❌ | ❌ | ❌ |
| Offline | ✅ | ❌ | ❌ | ❌ |
| A/B testing | ✅ Built-in | ❌ | ❌ | ❌ |
| Context-aware | ✅ | ❌ | ❌ | ❌ |
| Fallback chains | ✅ | ✅ | ✅ | ❌ |
| Zero dependencies | ✅ | ❌ | ❌ | ❌ |
| Infrastructure | None | Cloud | Cloud | GPU/API |

## Part of the Antaris Analytics Suite

- **[antaris-memory](https://pypi.org/project/antaris-memory/)** — Persistent memory for AI agents
- **antaris-router** — Adaptive model routing with SLA enforcement (this package)
- **[antaris-guard](https://pypi.org/project/antaris-guard/)** — Security and prompt injection detection
- **[antaris-context](https://pypi.org/project/antaris-context/)** — Context window optimization

## License

Licensed under the Apache License 2.0. See [LICENSE](LICENSE) for details.
