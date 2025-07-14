# ARC Core Benchmarking Guide

## Overview

ARC Core v1.1.0 introduces a comprehensive benchmarking system for evaluating AI models and measuring the impact of biological learning mechanisms. This guide shows you how to use the benchmarking features effectively.

## Quick Start

### 1. Installation

```bash
pip install metisos-arc-core==1.1.0
```

### 2. Basic CLI Usage

```bash
# Run benchmark evaluation
arc bench --suite examples/sample_benchmark.jsonl

# Compare base model vs ARC-enhanced
arc bench --suite examples/sample_benchmark.jsonl --model gpt2 --output results.json

# Generate markdown report
arc bench --suite tests.jsonl --format markdown --output report.md
```

### 3. Python API Usage

```python
import arc_core
from datetime import datetime

# Create benchmark suite
suite = arc_core.BenchmarkSuite(
    name="my-test-suite",
    description="Custom evaluation suite",
    test_cases=[
        {"prompt": "What is 2+2?", "expected": "4", "category": "math"},
        {"prompt": "Capital of Japan?", "expected": "Tokyo", "category": "geography"}
    ]
)

# Create metrics
metrics = arc_core.BenchmarkMetrics(
    model_name="my-model",
    num_samples=10,
    perplexity=15.2,
    avg_latency_ms=100.0,
    peak_memory_mb=512.0,
    coherence_score=0.85,
    toxicity_score=0.05,
    factual_accuracy=0.92,
    response_length_avg=45.0,
    timestamp=datetime.now().isoformat()
)

# Export results
results_json = metrics.to_json()
```

## Benchmark Suite Format

Benchmark suites are stored in JSONL format (one JSON object per line):

```jsonl
{"prompt": "What is 2+2?", "expected": "4", "category": "math"}
{"prompt": "Capital of France?", "expected": "Paris", "category": "geography"}
{"prompt": "Who wrote Hamlet?", "expected": "Shakespeare", "category": "literature"}
```

### Required Fields
- `prompt`: The input question or task
- `expected`: The expected/correct answer
- `category`: Category for grouping (math, science, logic, etc.)

### Optional Fields
- `difficulty`: Easy, Medium, Hard
- `source`: Where the question came from
- `tags`: Array of additional tags

## Metrics Explained

### Core Performance Metrics

| Metric | Description | Range | Better |
|--------|-------------|--------|--------|
| **Perplexity** | Language model uncertainty | 1+ | Lower |
| **Coherence Score** | Response logical consistency | 0-1 | Higher |
| **Factual Accuracy** | Correctness of factual claims | 0-1 | Higher |
| **Toxicity Score** | Harmful content detection | 0-1 | Lower |

### Performance Metrics

| Metric | Description | Units | Notes |
|--------|-------------|--------|-------|
| **Avg Latency** | Response generation time | ms | Per response |
| **Peak Memory** | Maximum memory usage | MB | During evaluation |
| **Response Length** | Average response length | chars | Text generation |

### Sample Metrics

| Metric | Description | Type | Notes |
|--------|-------------|------|-------|
| **Num Samples** | Total evaluated samples | int | Benchmark size |
| **Model Name** | Model identifier | string | For comparison |
| **Timestamp** | Evaluation time | ISO 8601 | Reproducibility |

## CLI Commands

### Basic Evaluation

```bash
# Evaluate with default settings
arc bench --suite my_tests.jsonl

# Limit number of samples
arc bench --suite large_suite.jsonl --max-samples 50

# Specify output file
arc bench --suite tests.jsonl --output results.json
```

### Model Comparison

```bash
# Compare base model vs ARC-enhanced
arc bench --suite tests.jsonl --model gpt2

# Use specific model path
arc bench --suite tests.jsonl --model ./my-model

# Custom config file
arc bench --suite tests.jsonl --config-file my_config.yaml
```

### Output Formats

```bash
# JSON output (default)
arc bench --suite tests.jsonl --format json

# Markdown report
arc bench --suite tests.jsonl --format markdown --output report.md
```

## Python API Reference

### BenchmarkSuite Class

```python
class BenchmarkSuite:
    def __init__(self, name: str, description: str, test_cases: List[Dict]):
        """Create a benchmark suite."""
        
    @classmethod
    def from_jsonl(cls, file_path: str) -> 'BenchmarkSuite':
        """Load suite from JSONL file."""
        
    @property
    def prompts(self) -> List[str]:
        """Get all prompts from test cases."""
        
    @property 
    def expected(self) -> List[str]:
        """Get all expected answers."""
        
    @property
    def categories(self) -> List[str]:
        """Get all categories from test cases."""
        
    def get_categories(self) -> Set[str]:
        """Get unique categories."""
```

### BenchmarkMetrics Class

```python
@dataclass
class BenchmarkMetrics:
    perplexity: float
    avg_latency_ms: float
    peak_memory_mb: float
    toxicity_score: float
    coherence_score: float
    factual_accuracy: float
    response_length_avg: float
    num_samples: int
    model_name: str
    timestamp: str
    
    def to_json(self) -> str:
        """Export metrics as JSON string."""
        
    def to_dict(self) -> Dict:
        """Export metrics as dictionary."""
```

### BenchmarkEvaluator Class

```python
class BenchmarkEvaluator:
    def evaluate(self, model_path_or_trainer, benchmark_suite: BenchmarkSuite, 
                max_samples: Optional[int] = None) -> BenchmarkMetrics:
        """Evaluate model against benchmark suite."""
```

## Best Practices

### 1. Benchmark Suite Design

- **Diverse Categories**: Include math, science, logic, language, factual knowledge
- **Balanced Difficulty**: Mix of easy, medium, and hard questions
- **Clear Expected Answers**: Unambiguous correct responses
- **Sufficient Size**: At least 50+ samples for reliable metrics

### 2. Evaluation Strategy

- **Consistent Environment**: Same hardware and software setup
- **Multiple Runs**: Average results across multiple evaluations
- **Baseline Comparison**: Always compare against established baselines
- **Documentation**: Record evaluation conditions and parameters

### 3. Result Interpretation

- **Statistical Significance**: Ensure improvements are meaningful
- **Trade-off Analysis**: Balance performance vs computational cost
- **Domain Specificity**: Consider performance in specific domains
- **Reproducibility**: Save all parameters and random seeds

## Example Workflows

### Research Evaluation

```python
# Load benchmark suite
suite = arc_core.BenchmarkSuite.from_jsonl("research_benchmark.jsonl")

# Evaluate base model
base_evaluator = arc_core.BenchmarkEvaluator()
base_metrics = base_evaluator.evaluate("gpt2", suite)

# Evaluate ARC-enhanced model
arc_trainer = arc_core.ARCTrainer("gpt2")
# ... training process ...
arc_metrics = base_evaluator.evaluate(arc_trainer, suite)

# Compare results
improvement = {
    "perplexity": (base_metrics.perplexity - arc_metrics.perplexity) / base_metrics.perplexity,
    "coherence": (arc_metrics.coherence_score - base_metrics.coherence_score) / base_metrics.coherence_score,
    "accuracy": (arc_metrics.factual_accuracy - base_metrics.factual_accuracy) / base_metrics.factual_accuracy
}

print(f"Perplexity improved by {improvement['perplexity']:.2%}")
print(f"Coherence improved by {improvement['coherence']:.2%}")
print(f"Accuracy improved by {improvement['accuracy']:.2%}")
```

### Production Monitoring

```bash
#!/bin/bash
# Daily model evaluation script

DATE=$(date +%Y%m%d)
RESULTS_FILE="daily_benchmark_${DATE}.json"

# Run benchmark
arc bench --suite production_tests.jsonl --output "$RESULTS_FILE"

# Archive results
mkdir -p benchmark_history
mv "$RESULTS_FILE" benchmark_history/

# Alert if performance degrades
python check_performance_regression.py benchmark_history/
```

## Troubleshooting

### Common Issues

1. **Memory Errors**: Reduce `--max-samples` or use smaller models
2. **Slow Evaluation**: Enable GPU acceleration or reduce batch size
3. **Import Errors**: Ensure `metisos-arc-core>=1.1.0` is installed
4. **JSONL Format**: Validate file format with JSON linter

### Performance Tips

- Use GPU acceleration for large model evaluation
- Batch multiple samples together when possible
- Cache model loading between evaluations
- Use streaming for very large benchmark suites

## Support

- **Documentation**: Complete API reference and examples
- **Examples**: See `examples/` directory for working code
- **Issues**: Report bugs on GitHub
- **Community**: Join discussions for best practices

---

**ARC Core v1.1.0** - Comprehensive benchmarking for biological learning research
