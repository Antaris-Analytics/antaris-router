# Antaris Router

**Deterministic model routing for 50-70% LLM cost reduction. Zero dependencies.**

File-based prompt classification that routes to the cheapest capable model. Same input always produces the same routing decision. No API calls for classification, no vector databases, no infrastructure overhead.

[![Tests](https://img.shields.io/badge/tests-passing-brightgreen)](https://github.com/Antaris-Analytics/antaris-router)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-green.svg)](https://python.org)
[![License](https://img.shields.io/badge/license-Apache%202.0-orange.svg)](LICENSE)

## Cost Impact

**Live test results (5 diverse prompts):**

```
Prompt                          Tier     Model              Cost      vs GPT-4o
"Hi there!"                    trivial   gpt-4o-mini       $0.000016  $0.275000
"What is machine learning?"    trivial   gpt-4o-mini       $0.000016  $0.275000  
"Explain microservices..."     simple    llama-3-1-70b     $0.000000  $0.862500
"React component with TS..."   simple    llama-3-1-70b     $0.000000  $0.862500
"Design distributed system..." expert    gemini-pro-1-5    $0.004375  $8.750000

Total cost:           $0.0044      vs      $11.0250
Monthly (10k reqs):   $8.82       vs      $22,050.00
Savings:              $22,041.18 (99.96%)
```

**Key insight:** Most applications waste money using expensive models for routine tasks. Simple routing rules deliver massive savings.

## How It Works

1. **Classify** prompts using deterministic keyword matching + structural analysis
2. **Route** to cheapest model in each capability tier (trivial → simple → moderate → complex → expert)
3. **Track** actual usage costs and compare against premium-only baseline
4. **Optimize** spending while maintaining output quality

All routing decisions happen offline using plain text rules stored in JSON files.

## What It Does

- Prompt complexity classification (5 tiers: trivial → expert)
- Cost-optimized model selection within each tier  
- Usage tracking with savings estimates vs. premium models
- Provider preferences and capability-based routing
- Deterministic decisions — same prompt always routes the same way

## What It Doesn't Do

- **API proxy** — Returns routing decisions only, you make the actual calls
- **Semantic analysis** — Uses keyword matching, not embeddings or model inference  
- **Learning system** — Rules are static, doesn't adapt based on outcomes
- **Rate limiting** — Handles routing logic only, not request management
- **Quality assessment** — Assumes all models in a tier produce equivalent results

## Technical Approach

Same principles as [antaris-memory](https://github.com/Antaris-Analytics/antaris-memory):

| Principle | Implementation |
|-----------|----------------|
| **File-based** | JSON config files. No databases, no external services. |  
| **Deterministic** | Identical inputs produce identical routing decisions. |
| **Offline-first** | Classification runs locally using keyword matching. |
| **Zero dependencies** | Pure Python stdlib. No vendor lock-in. |
| **Transparent** | Inspect routing rules with any text editor. |

## Install

```bash
pip install antaris-router
```

## Usage

```python
from antaris_router import Router

# Initialize with default config  
router = Router()

# Route prompts to appropriate models
simple_q = router.route("What is Python?")
# → gpt-4o-mini ($0.15/MTok) instead of gpt-4o ($2.50/MTok)

architecture = router.route("""
Design a microservices architecture for handling 
100k concurrent users with Redis caching...
""")  
# → claude-sonnet ($3/MTok) instead of opus ($15/MTok)

# Log actual usage for cost tracking
router.log_usage(simple_q, input_tokens=12, output_tokens=150, actual_cost=0.0024)

# View savings report
savings = router.savings_estimate()
print(f"This month: ${savings['period_cost']:.2f}")
print(f"Without router: ${savings['baseline_cost']:.2f}")  
print(f"Saved: ${savings['total_savings']:.2f} ({savings['savings_percent']:.1f}%)")
```

## Live Routing Examples

**How classification works in practice:**

```python
from antaris_router import Router

router = Router()

# Trivial: Short, conversational
decision = router.route("Hi there!")
# → gpt-4o-mini ($0.15/MTok) 
# → Reasoning: "Very short prompt (9 chars), 1 trivial-tier keyword found"

# Simple: Factual questions  
decision = router.route("What is machine learning?")
# → gpt-4o-mini ($0.15/MTok)
# → Reasoning: "Short prompt, basic question pattern"

# Complex: Technical implementation
decision = router.route("Implement a React component with TypeScript")
# → llama-3-1-70b (free local model)
# → Reasoning: "1 complex-tier keyword found (implement), programming context"

# Expert: System architecture
decision = router.route("""
Design a distributed system architecture for 100k concurrent users.
Include database sharding, Redis caching, and auto-scaling.
""")
# → gemini-pro-1-5 ($1.25/MTok)
# → Reasoning: "Architecture keywords, high complexity, long prompt"
```

**Deterministic decisions:** Same prompt always routes to the same model. No randomness.

## Classification System

**5 tiers from cheapest to most expensive:**

| Tier | Cost Range | Use Cases |
|------|------------|-----------|
| **Trivial** | $0.10-0.20/MTok | Greetings, confirmations, simple Q&A |
| **Simple** | $0.15-0.50/MTok | Factual lookup, basic explanations |
| **Moderate** | $1.00-3.00/MTok | Analysis, summarization, structured data |
| **Complex** | $2.50-15.0/MTok | Code generation, technical design |  
| **Expert** | $15.0-75.0/MTok | Novel research, creative problem solving |

**Classification signals:**
- Presence of technical keywords (`API`, `algorithm`, `architecture`)
- Prompt length and structural complexity (code blocks, numbered lists)
- Explicit complexity markers (`explain in detail`, `comprehensive analysis`)

**Not semantic understanding** — Uses pattern matching, not AI classification.

## When This Works

**Good fit:**
- High-volume applications with mixed complexity (customer support, content generation)
- Budget-conscious teams that need predictable routing decisions  
- Workflows where 80% of prompts are routine, 20% need premium models
- Integration into existing codebases without infrastructure changes

**Not a good fit:**
- Single-model applications (no cost optimization opportunity)
- Highly specialized domains where complexity classification fails
- Real-time applications needing sub-10ms routing decisions
- Teams that prefer semantic similarity over keyword matching

## Limitations  

- **Pattern-based only** — Misclassifies prompts that don't match keyword patterns
- **No quality feedback** — Doesn't learn if cheaper models produce poor results
- **Static rules** — Classification logic doesn't adapt to your specific use case
- **English-optimized** — Keyword matching may not work well for other languages
- **No model performance tracking** — Assumes all models in a tier are equivalent

If you need semantic classification or quality-based routing, this tool isn't suitable.

## Configuration

The router uses JSON files for all configuration. Defaults work for most use cases.

**Customize model costs:**
```bash
# Edit config/models.json to add new models or update pricing
vim config/models.json
```

**Adjust classification rules:**  
```bash
# Modify config/classification.json to tune keyword matching
vim config/classification.json  
```

**Track usage:**
```python
# Cost tracking happens automatically
report = router.cost_report()
print(f"Monthly cost: ${report['total_cost']:.2f}")
print(f"Requests routed: {report['total_requests']:,}")
```

All configuration files use plain JSON — no proprietary formats or complex schemas.
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

Simple 4-component design:
- **TaskClassifier** — Prompt → complexity tier  
- **ModelRegistry** — Model definitions and costs
- **CostTracker** — Usage logging and savings calculation
- **Router** — Combines everything, returns routing decisions

Data flow: `prompt → classify → find cheapest model for tier → return decision`

## Related Tools

- **[antaris-memory](https://github.com/Antaris-Analytics/antaris-memory)** — File-based persistent memory for AI agents
- **OpenRouter, LiteLLM** — Full model proxies (require API keys, network calls)
- **LangChain** — Agent framework (uses model inference for routing)

## Development

```bash
# Run tests
python -m pytest tests/ -v

# Install development dependencies  
pip install -e .[dev]

# Type checking
mypy antaris_router/
```

## License

Apache License 2.0. See [LICENSE](LICENSE) for details.

---

**Part of Antaris Analytics** — File-based tools for deterministic AI applications.