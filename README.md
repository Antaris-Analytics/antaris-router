# antaris-router

**Adaptive model router for LLM cost optimization. Learns from outcomes. Zero dependencies.**

**v4.1.0** — SLA enforcement • Outcome-based learning • Cost tracking • 13 source files • 253 tests**

## What's New in v4.1.0

- **SLA Enforcement** — Set latency/cost targets per route. Router automatically escalates to faster models when SLA is at risk. Proven on multi-billion-token workloads.
- **Outcome Learning** — Routes improve over time. Track which models actually nail which task types (not just cost, but quality).
- **Provider Health Tracking** — TTL-based status for provider outages. Automatic fallback to backup chains.
- **Cost Tracking & Reporting** — Per-model cost attribution. A/B test routing strategies.
- **Zero Infrastructure** — All state in JSON files. No vector DB, no external dependencies, no API keys.

## Phase 4 Roadmap

**v4.2:** Discord context as routing signal (conversation history informs task classification)  
**v4.3:** Self-improving routing (learn from outcome quality, not just cost)  
**v4.4+:** Multi-provider optimization (manage cost/latency across OpenAI, Anthropic, Google, local models)

Routes prompts to the cheapest capable model using semantic classification (TF-IDF), not keyword matching. Tracks outcomes to learn which models actually perform well on which tasks. Enforces cost/latency SLAs. Provider health tracking with TTL-based status. A/B testing for routing strategy validation. All state stored in plain JSON files. No API keys, no vector database, no infrastructure.

[![PyPI](https://img.shields.io/pypi/v/antaris-router)](https://pypi.org/project/antaris-router/)
[![Tests](https://github.com/Antaris-Analytics/antaris-router/actions/workflows/tests.yml/badge.svg)](https://github.com/Antaris-Analytics/antaris-router/actions/workflows/tests.yml)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-green.svg)](https://python.org)
[![License](https://img.shields.io/badge/license-Apache%202.0-orange.svg)](LICENSE)

## What's New

- **Provider health tracking** — `record_provider_health(provider, status, latency_ms, ttl_seconds)` with TTL-based expiry; routing automatically avoids "down" providers and de-prioritises "degraded" ones
- **SLA 24h pruning** — `SLAMonitor._records` bounded to a 24-hour window; no unbounded memory growth in long-running agents
- **Outcome-quality routing** — router adapts model selection based on real outcome feedback; models below quality threshold are auto-skipped
- **Confidence-gated escalation** — routes to a stronger model when classification confidence drops below a configurable threshold
- **ProviderHealthTracker** — bounded deques (maxlen=10,000) track latency and error rates per provider in real time
- **A/B testing** — deterministic variant assignment for reproducible routing experiments
- **SLA enforcement** — cost budgets, latency targets, quality floors; `SLAConfig`, `get_sla_report()`, `check_budget_alert()`
- **Suite integration** — router hints consumed by `antaris-context` via `set_router_hints()` for adaptive context budget allocation
- 253 tests (all passing)

See [CHANGELOG.md](CHANGELOG.md) for full version history.

---

## Install

```bash
pip install antaris-router
```

---

## AdaptiveRouter

The recommended API. Semantic classification, quality tracking, fallback chains, A/B testing, and outcome learning — all in one class.

```python
from antaris_router import AdaptiveRouter, ModelConfig

router = AdaptiveRouter("./routing_data", ab_test_rate=0.05)

# Register models with their capability ranges
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

router.save()
```

---

## Provider Health Tracking

Track provider status with TTL-based expiry. The router consults health state during routing to avoid down providers and de-prioritise degraded ones.

```python
from antaris_router import Router

router = Router(config_path="./config")

# Record health status with TTL (default 300 seconds)
router.record_provider_health("claude-sonnet", status="ok", latency_ms=45.2, ttl_seconds=300)
router.record_provider_health("gpt-4o-mini", status="degraded", latency_ms=890.0, ttl_seconds=120)
router.record_provider_health("claude-opus", status="down", latency_ms=0.0, ttl_seconds=60)

# Query current health state (expires after TTL)
state = router.get_provider_health_state("claude-sonnet")
print(state)
# → {"provider": "claude-sonnet", "status": "ok", "latency_ms": 45.2,
#    "recorded_at": 1740100000.0, "expires_at": 1740100300.0}

# Expired or unknown providers return {"status": "unknown"}
state = router.get_provider_health_state("unknown-model")
# → {"status": "unknown", "provider": "unknown-model"}

# Health-aware routing: avoids "down", prefers "ok" over "degraded"
decision = router.route("Summarize this document", prefer_healthy=True)
```

Status values: `"ok"`, `"degraded"`, `"down"`. The router also accepts event-level tracking via `record_provider_event(model, event, details, latency_ms)` for fine-grained health signals.

---

## SLA Enforcement

Enforce cost budgets, latency targets, and quality floors. The SLA monitor auto-escalates or downgrades models to stay within bounds.

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
print(f"Compliance: {report['compliance_rate']:.0%}")

# Budget alerts
alert = router.check_budget_alert()
if alert['status'] != 'ok':
    print(f"Budget alert ({alert['status']}): {alert['recommendation']}")
```

---

## A/B Testing

Validate routing strategies with deterministic variant assignment. Run experiments to compare cost-optimised vs quality-first routing.

```python
from antaris_router import Router
from antaris_router.confidence import ABTest

router = Router(config_path="./config")

# Create an A/B test
test = router.create_ab_test(
    name="cost_vs_quality",
    strategy_a="cost_optimized",
    strategy_b="quality_first",
    split=0.5,
)

# Pass the test to route() — variant assignment is deterministic
decision = router.route("Write a complex algorithm", ab_test=test)
print(f"Variant: {decision.ab_variant}")  # → "a" or "b"
```

The `AdaptiveRouter` also supports A/B testing via `ab_test_rate` — a configurable percentage of requests are routed to premium models to validate that cheap routing is working:

```python
router = AdaptiveRouter("./routing_data", ab_test_rate=0.05)
result = router.route("Simple question")
print(result.ab_test)  # → True on ~5% of requests
```

---

## Confidence Gating

When classification confidence is low, the router can escalate, fall back to a safe default, or flag the request for clarification.

```python
from antaris_router import AdaptiveRouter, ModelConfig

router = AdaptiveRouter(
    "./routing_data",
    confidence_threshold=0.6,
    confidence_strategy="escalate",    # or "safe_default", "clarify"
    safe_default_model="claude-sonnet", # used with "safe_default" strategy
)

# route_with_confidence() returns a RouteDecision with strategy metadata
decision = router.route_with_confidence("Some ambiguous request")
print(decision.confidence)        # → 0.42
print(decision.strategy_applied)  # → "escalated" (confidence < 0.6)
print(decision.basis)             # → "semantic_classifier" or "composite"
```

The legacy `Router` also supports confidence-gated escalation:

```python
from antaris_router import Router

router = Router(
    low_confidence_threshold=0.5,
    escalation_model="claude-opus",
    escalation_strategy="always",  # or "log_only", "ask"
)

decision = router.route("Vague request")
if decision.escalated:
    print(f"Escalated: {decision.escalation_reason}")
    print(f"Original confidence: {decision.original_confidence:.2f}")
```

---

## Outcome Learning

The router gets smarter over time. Report outcomes to build per-model per-tier quality profiles.

```python
result = router.route("Implement retry logic with exponential backoff")

# After using the model, report the outcome
router.report_outcome(result.prompt_hash, quality_score=0.9, success=True)

# Report failures — router learns to skip this model for this task type
router.report_outcome(result.prompt_hash, quality_score=0.15, success=False)
```

Quality scores per model per tier:
```
score = 0.4 * success_rate + 0.4 * avg_quality + 0.2 * (1 - escalation_rate)
```

Models below the escalation threshold (default 0.30) are automatically skipped.

---

## Cost Tracking

Track actual token usage, generate cost reports, and forecast future spend.

```python
from antaris_router import Router

router = Router(config_path="./config", enable_cost_tracking=True)

decision = router.route("Explain quantum computing")

# Log actual usage after model call
actual_cost = router.log_usage(decision, input_tokens=150, output_tokens=500)

# Cost report
report = router.cost_report(period="week")

# Savings estimate vs always using a premium model
savings = router.savings_estimate(comparison_model="gpt-4o")

# Cost forecasting
forecast = router.forecast_cost(
    requests_per_hour=100,
    avg_input_tokens=200,
    avg_output_tokens=400,
)
print(f"Projected daily cost: ${forecast['daily_cost_usd']:.2f}")
print(f"Tip: {forecast['optimization_tip']}")

# Cost optimization suggestions
optimizations = router.get_cost_optimizations()
for opt in optimizations:
    print(f"{opt['suggestion']} — saves ${opt['estimated_savings_usd_per_day']:.2f}/day")
```

---

## Suite Integration — `set_router_hints()`

antaris-router publishes routing decisions that `antaris-context` consumes via `set_router_hints()` for adaptive context budget allocation. The router tells the context manager which model was selected, its tier, and cost profile — so context windows are sized appropriately.

```python
from antaris_router import AdaptiveRouter, ModelConfig
# antaris-context reads router hints for budget allocation
# from antaris_context import set_router_hints

router = AdaptiveRouter("./routing_data")
router.register_model(ModelConfig(
    name="claude-sonnet",
    tier_range=("simple", "complex"),
    cost_per_1k_input=0.003,
    cost_per_1k_output=0.015,
))

result = router.route("Build a REST API with authentication")

# Pass routing decision to antaris-context for budget-aware context sizing
# set_router_hints(model=result.model, tier=result.tier, confidence=result.confidence)
```

This pairing is wired automatically in **antaris-pipeline**.

---

## OpenClaw Integration

antaris-router is designed for OpenClaw agent workflows. Drop it into any pipeline to get intelligent model selection without modifying your agent logic.

```python
from antaris_router import Router

router = Router(config_path="router.json")
model = router.route(prompt)  # Returns the optimal model for this prompt
```

Pairs naturally with antaris-guard (pre-routing safety check) and antaris-context (token budget awareness).

---

## Context-Aware Routing

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

# High urgency — boost tier
result = router.route("Fix production outage", context={"urgency": "high"})
```

---

## Fallback Chains

```python
result = router.route("Write unit tests for authentication")
print(result.model)           # → gpt-4o-mini
print(result.fallback_chain)  # → ['claude-sonnet', 'claude-opus']

# escalate() distinguishes two outcomes:
#   KeyError → hash not in tracker (process restarted, tracker rotated)
#              re-route from scratch rather than escalating
#   None     → hash found, but all fallback tiers are exhausted
#   str      → next model to try
try:
    next_model = router.escalate(result.prompt_hash)
    if next_model is None:
        print("All fallbacks exhausted — surface error to user")
    else:
        print(next_model)  # → claude-sonnet
except KeyError:
    print("Decision not tracked — re-route from scratch")
```

---

## Teaching Corrections

```python
# Classifier thinks this is simple, but it's actually complex
router.teach(
    "Optimize our Kubernetes deployment for cost efficiency",
    "complex"
)
# Correction is learned permanently
```

---

## Works With Local Models (Ollama)

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

40% of typical requests route to local models ($0.00). At 1,000 requests/day, that's ~$10.80/day vs ~$18.00/day all-Sonnet.

The router doesn't call models — it tells you which one to use. Wire it to Ollama's API, LiteLLM, or any client you prefer.

---

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

---

## Tiers

| Tier | Description | Examples |
|------|-------------|----------|
| trivial | One-line answers, lookups | "What is 2+2?", "Define photosynthesis" |
| simple | Short tasks, basic code | "Reverse a string", "Explain TCP vs UDP" |
| moderate | Multi-step implementation | "Build a REST API with auth" |
| complex | Architecture, multi-system | "Design microservices for e-commerce" |
| expert | Full system design | "Architect a globally distributed database" |

---

## Storage Format

```
routing_data/
├── routing_examples.json    # Labeled examples (seed + learned)
├── routing_model.json       # TF-IDF model (IDF weights, vocab)
├── routing_decisions.json   # Decision history for outcome learning
├── model_profiles.json      # Per-model per-tier quality scores
└── router_config.json       # Model registry and settings
```

Plain JSON. Inspect or edit with any text editor.

---

## Architecture

```
AdaptiveRouter (recommended)
├── SemanticClassifier
│   └── TFIDFVectorizer     — Term weighting + cosine similarity
├── QualityTracker
│   ├── RoutingDecision     — Decision + outcome records
│   └── ModelProfiles       — Per-model per-tier quality scores
├── ContextAdjuster         — Iteration, conversation, expertise, urgency signals
├── FallbackChain           — Ordered model escalation
├── ConfidenceGating        — Escalate / safe_default / clarify strategies
└── ABTester                — Validation routing (configurable %)

Router (legacy keyword-based, with SLA + health)
├── TaskClassifier          — Keyword-based + structural classification
├── ModelRegistry           — Model capabilities and cost data
├── CostTracker             — Usage records, savings analysis, forecasting
├── SLAMonitor              — Budget alerts, latency enforcement, 24h pruning
├── ProviderHealthTracker   — Bounded deques, real-time error/latency tracking
├── ProviderHealthState     — TTL-based explicit status (ok/degraded/down)
├── ConfidenceRouter        — Score-weighted routing with escalation strategies
└── ABTest                  — Deterministic variant assignment
```

---

## Performance

```
Routing latency: median 0.05ms, p99 0.09ms, avg 0.05ms
Classification: ~50 seed examples, TF-IDF with cosine similarity
Memory: <5MB for typical workloads
```

Measured on Apple M4, Python 3.14.

---

## What It Doesn't Do

- **Not a proxy** — doesn't forward requests to models. It tells you *which* model to use.
- **Not semantic search** — uses TF-IDF (bag-of-words with term weighting), not embeddings.
- **Not real-time market data** — doesn't track live model pricing or availability.
- **Classification is statistical, not perfect** — edge cases exist. Use `teach()` to correct them.
- **Quality tracking requires your feedback** — call `report_outcome()` after using the model.

---

## Routing Analytics

```python
analytics = router.routing_analytics()
print(f"Total decisions: {analytics['total_decisions']}")
print(f"Tier distribution: {analytics['tier_distribution']}")
print(f"Most used model: {analytics['most_used_model']}")
print(f"Avg confidence: {analytics['avg_confidence']:.3f}")
```

---

## Legacy API

The v1 keyword-based router is still available and fully supported:

```python
from antaris_router import Router  # v1 API (with SLA + health features)
router = Router(config_path="./config")
decision = router.route("What's 2+2?")
```

We recommend `AdaptiveRouter` for new code.

---

## Running Tests

```bash
git clone https://github.com/Antaris-Analytics/antaris-router.git
cd antaris-router
pip install pytest
python -m pytest tests/ -v
```

All 253 tests pass with zero external dependencies.

---

## Part of the Antaris Analytics Suite — v3.0.0

- **[antaris-memory](https://pypi.org/project/antaris-memory/)** — Persistent memory for AI agents
- **antaris-router** — Adaptive model routing with SLA enforcement (this package)
- **[antaris-guard](https://pypi.org/project/antaris-guard/)** — Security and prompt injection detection
- **[antaris-context](https://pypi.org/project/antaris-context/)** — Context window optimization
- **[antaris-pipeline](https://pypi.org/project/antaris-pipeline/)** — Agent orchestration pipeline
- **[antaris-contracts](https://pypi.org/project/antaris-contracts/)** — Versioned schemas, failure semantics, and debug CLI

## License

Apache 2.0 — see [LICENSE](LICENSE) for details.

---

**Built with love by Antaris Analytics**
*Deterministic infrastructure for AI agents*
