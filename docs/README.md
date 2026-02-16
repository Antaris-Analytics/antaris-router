# Antaris Router Documentation

Documentation for the Antaris Router package.

## Getting Started

- [Quick Start Guide](../README.md#quick-start)
- [Examples](../examples/)
- [API Reference](#api-reference)

## Core Concepts

### Task Classification
Prompts are classified into five complexity tiers:
- **Trivial**: Greetings, simple yes/no questions
- **Simple**: Basic factual questions, short explanations  
- **Moderate**: Analysis, planning, multi-step tasks
- **Complex**: Code generation, architecture decisions
- **Expert**: Research synthesis, novel problem solving

### Model Registry
The registry manages model definitions including:
- Cost per 1K input/output tokens
- Supported complexity tiers
- Capabilities (vision, code, etc.)
- Maximum context length

### Routing Logic
1. Classify prompt complexity
2. Find models supporting that tier
3. Apply filters (provider, capability, etc.)
4. Select cheapest suitable model
5. Return routing decision

## API Reference

### Router Class

Main interface for routing decisions.

```python
Router(config_path=None, enable_cost_tracking=True)
```

#### Methods

- `route(prompt, context=None, prefer=None, min_tier=None, capability=None)` - Route a prompt
- `log_usage(decision, input_tokens, output_tokens)` - Log actual usage
- `cost_report(period="week")` - Generate cost report
- `savings_estimate(comparison_model="gpt-4o")` - Calculate savings

### RoutingDecision Class

Result of routing operation.

#### Attributes
- `model` - Selected model name
- `provider` - Model provider
- `tier` - Complexity tier used
- `confidence` - Classification confidence
- `reasoning` - List of decision reasons
- `estimated_cost` - Estimated cost for typical usage
- `fallback_models` - Alternative model options

## Configuration

See [Configuration Guide](configuration.md) for details on customizing models and classification rules.

## Examples

See the [examples directory](../examples/) for complete usage examples.