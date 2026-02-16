# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
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
- Thread-safe operations
- Configurable classification rules and model definitions

## [Unreleased]

### Planned
- Integration examples for popular LLM clients (OpenAI, Anthropic, etc.)
- Web dashboard for cost monitoring and analytics
- Model performance benchmarking tools
- A/B testing framework for routing strategies
- Export/import utilities for configuration management