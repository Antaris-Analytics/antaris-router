# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),

## [3.2.0] - 2026-02-20

### Added — Sprint 2.7: Provider Health Tracking

- **`Router.record_provider_health(provider, status, latency_ms, ttl_seconds=300)`** — record explicit health status with 5-minute TTL
  - `"ok"` → provider is healthy, prefer in routing
  - `"degraded"` → provider has issues, deprioritise in routing
  - `"down"` → provider is unavailable, skip entirely in routing
  - Mirrors into `ProviderHealthTracker` for consistent `get_provider_health()` responses
- **`Router.get_provider_health_state(provider)`** — return the TTL-based explicit state; returns `{"status": "unknown"}` if expired or not set
- **`Router._get_effective_health_status(model)`** — internal helper that checks TTL state first, falls back to event-based tracker
- **Health-aware routing enhanced** — `route(prefer_healthy=True)` now also excludes `"down"` models by explicit state, and sorts `"degraded"` below `"ok"` in the candidate list
- 229 existing tests continue to pass (no regressions)

## [3.1.0] - 2026-02-20

### Added — Sprint 2.3: Confidence-Gated Routing

- **`RouteDecision` dataclass** — `AdaptiveRouter.route_with_confidence()` now returns a
  `RouteDecision` with `confidence` (0.0–1.0), `basis` (semantic_classifier / quality_tracker /
  composite), `reason` (human-readable), and `strategy_applied` fields.
- **Low-confidence strategies** (configurable via `confidence_strategy` constructor arg):
  - `"escalate"` — bumps routing to the next tier when confidence < threshold.
  - `"safe_default"` — routes to a configured fallback model on low confidence.
  - `"clarify"` — keeps routing but sets `strategy_applied="clarify"` to signal ambiguity.
  - Default threshold: 0.6 (exposed as `DEFAULT_CONFIDENCE_THRESHOLD`).
- **`AdaptiveRouter.explain(request)`** — read-only method returning a structured dict with
  classification result, quality scores, cost estimate, candidate models, `why_selected` map,
  and a human-readable `summary`. Does not record a routing decision.
- **Backward compatible** — `route()` return type (`RoutingResult`) and all existing fields
  unchanged; callers using only `result.model` require zero migration.
- **35 new tests** in `tests/test_confidence.py` covering all strategies, `explain()`,
  `RouteDecision.to_dict()`, error handling, and backward compatibility (229 total, all passing).

## [3.0.0] - 2026-02-18

### Added — Antaris Suite 2.0 GA
- **SLA Monitor** — cost budget enforcement, quality score tracking per model/tier, `get_sla_report()`, `check_budget_alert()`
- **Confidence Routing** — `ConfidenceRouter`, scoring with `RoutingDecision.confidence_basis` for cross-package tracing
- **Sprint 7 backward compat** — `RoutingDecision.to_dict()` extended fields; `SLAConfig` optional; safe defaults throughout
- **194 unit tests** (all passing); all Sprint 7 SLA params verified backward-compatible
- Suite integration: router hints consumed by `antaris-context` `set_router_hints()` for adaptive budget allocation

## [2.0.0] - 2026-02-16

### Added (Complete Routing Rewrite)
- **Semantic Classification** (`SemanticClassifier`): TF-IDF vectorizer + cosine similarity replaces keyword matching
- **TF-IDF Vectorizer** (`TFIDFVectorizer`): Lightweight implementation using only stdlib — tokenization, IDF computation, cosine similarity
- **Quality Tracking** (`QualityTracker`): Records routing decisions and outcomes, builds per-model per-tier quality profiles
- **Adaptive Router** (`AdaptiveRouter`): Combines semantic classification, quality learning, fallback chains, A/B testing, and context-aware routing
- **Outcome Learning**: `report_outcome()` feeds quality data back to improve future routing
- **A/B Testing**: Configurable percentage of requests routed to premium for validation
- **Context-Aware Routing**: Iteration count, conversation length, user expertise affect tier selection
- **Fallback Chains**: Ordered model escalation with `escalate()` method
- **Multi-Objective Optimization**: Route by quality, cost, speed, or balanced
- **Teaching API**: `teach()` permanently corrects classification mistakes
- 50+ seed classification examples across 5 tiers
- 33 new tests for v2.0 features (67 total)

### Changed
- Default classification method is now semantic (TF-IDF) instead of keyword matching
- Version bumped to 2.0.0 (breaking: new primary API)

### Migration
- Legacy v1 API (`Router`, `TaskClassifier`) still available and unchanged
- New API uses `AdaptiveRouter` and `ModelConfig`
- `pip install antaris-router` gets both APIs
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-02-15

### Added
- Initial release of Antaris Router
- Task classification into 5 complexity tiers (trivial, simple, moderate, complex, expert)
- Model registry with cost tracking and capability filtering
- Deterministic routing based on keywords, length, and structure
- Cost tracking with usage logging and savings analysis
- Support for provider preferences and tier overrides
- Comprehensive test suite with 95%+ coverage
- JSON-based configuration system with sensible defaults
- Zero-dependency implementation using Python standard library
- Routing analytics and performance metrics
- Examples and quickstart guide

### Models Included
- OpenAI: GPT-4o, GPT-4o-mini
- Anthropic: Claude Opus, Sonnet, Haiku 3.5
- Google: Gemini Pro 1.5, Gemini Flash 1.5  
- Local: Llama 3.1 70B Instruct

### Features
- Offline routing decisions (no API calls required)
- Transparent JSON storage format
- Fallback routing to higher tiers when needed
- Model efficiency analysis
- Cost optimization with automatic cheapest-model selection
- Single-process safe operations (concurrent writer locking planned for future release)
- Configurable classification rules and model definitions

## [0.3.0] - 2026-02-15

### Changed
- **Major README rewrite** to match antaris-memory positioning style
- Cost savings front and center (real examples: $847→$251/month, 70.3% savings)
- Function-first approach with concrete code examples vs marketing descriptions  
- Added honest limitations section ("When this works" vs "Not a good fit")
- Infrastructure tooling tone — boring in the right way, no AI buzzwords
- Streamlined from 300+ to ~150 lines focused on practical value
- Added "Related Tools" section linking to antaris-memory

### Added
- Production cost impact examples with real dollar amounts
- Clear use case guidance (high-volume mixed complexity vs single-model apps)
- Explicit comparison to OpenRouter, LiteLLM, LangChain with feature matrix
- Development workflow examples (cheap iteration → expensive production)

## [Unreleased]

### Planned
- Integration examples for popular LLM clients (OpenAI, Anthropic, etc.)
- Web dashboard for cost monitoring and analytics
- Model performance benchmarking tools
- A/B testing framework for routing strategies
- Export/import utilities for configuration management