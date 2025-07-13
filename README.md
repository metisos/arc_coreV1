# ARC Core - Public Interface Repository

This repository contains the **public interface** for ARC Core (Adaptive Recursive Consciousness Engine). 

**WARNING: This is NOT the implementation repository - it only shows the public API.**

[![PyPI version](https://badge.fury.io/py/arc-core.svg)](https://badge.fury.io/py/arc-core)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

## Installation & Usage

To use ARC Core, install the package from PyPI:

```bash
pip install metisos-arc-core
```

## What is ARC Core?

ARC Core is a sophisticated AI learning system that implements biological learning mechanisms in language models, enabling true continual learning and adaptive consciousness. 

**Key Features:**
- **Biological Learning Mechanisms**: Contextual gating, cognitive inhibition, and sleep-like consolidation
- **Hierarchical Memory Systems**: Working, episodic, and semantic memory with temporal context  
- **Continual Learning**: Real weight updates without catastrophic forgetting
- **Safety-First Design**: Multi-layered cognitive inhibition and metacognitive monitoring
- **Teaching Pack System**: Modular training with specialized learning modules
- **Modular Teaching Packs**: Easy-to-use training modules for specific domains
- **CLI Interface**: Simple command-line tools for model management
- **Hugging Face Integration**: Seamless model loading and saving

## Quick Start

### Installation

```bash
pip install metisos-arc-core
```

For GPU support:
```bash
pip install metisos-arc-core[gpu]
```

For Apple Silicon:
```bash
pip install metisos-arc-core[apple]
```

### Basic Usage

#### 1. Initialize ARC with a base model

```bash
arc init --base-model cognitivecomputations/TinyDolphin-2.8-1.1b
```

#### 2. Teach the model using a training pack

```bash
arc teach sentiment-basic
```

#### 3. Test the model's performance

```bash
arc test sentiment-basic
```

#### 4. Interactive chat with your enhanced model

```bash
arc chat
```

#### 5. Save your trained model

```bash
arc save --out ./my-arc-model
```

### Python API Usage

```python
from arc_core import ARCTrainer, ARCConfig

# Initialize configuration
config = ARCConfig()
config.device = "cuda"  # or "cpu", "mps"

# Create trainer
trainer = ARCTrainer(config)

# Initialize with base model
trainer.initialize_model("cognitivecomputations/TinyDolphin-2.8-1.1b")

# Train on a teaching pack
result = trainer.train_on_pack("sentiment-basic")

# Generate responses
response = trainer.generate_response("I'm feeling great today!")
print(response)  # Should show positive, supportive response

# Save the enhanced model
trainer.save_model("./my-enhanced-model")
```

## 🧬 Architecture

ARC Core implements several biologically-inspired learning mechanisms:

### Memory Systems
- **Working Memory**: Short-term context and active processing
- **Episodic Memory**: Specific interaction memories with temporal context
- **Semantic Memory**: Extracted concepts and knowledge patterns

### Safety Mechanisms
- **Cognitive Inhibition**: Filters harmful or inappropriate responses
- **Contextual Gating**: Controls memory encoding and retrieval
- **Metacognitive Monitoring**: Self-assessment of response quality

### Learning Systems
- **LoRA Adapters**: Efficient parameter updates without full retraining
- **Elastic Weight Consolidation**: Prevents catastrophic forgetting
- **Continual Learning**: Accumulates knowledge across training sessions

## 📦 Teaching Packs

Teaching packs are modular training datasets that enable targeted learning:

### Built-in Packs
- **sentiment-basic**: Basic sentiment analysis and appropriate responses

### Creating Custom Packs

Create a directory with the following structure:
```
my-pack/
├── pack.yml          # Metadata and configuration
├── training.jsonl    # Training data
└── test_suite.jsonl  # Evaluation data
```

Example `pack.yml`:
```yaml
name: my-pack
version: 1.0.0
description: Custom training pack
author: Your Name

learning_objectives:
  - Objective 1
  - Objective 2

datasets:
  training: training.jsonl
  
test_suite: test_suite.jsonl
```

Example training data (`training.jsonl`):
```json
{"input": "User message", "output": "Model response"}
{"input": "Another message", "output": "Another response"}
```

## 🛠️ CLI Commands

| Command | Description |
|---------|-------------|
| `arc init` | Initialize ARC with a base model |
| `arc teach <pack>` | Train on a teaching pack |
| `arc test <pack>` | Test model performance |
| `arc chat` | Interactive chat session |
| `arc save` | Save trained model |
| `arc status` | Show system status |
| `arc check` | Health check and requirements |

### CLI Examples

```bash
# Initialize with specific settings
arc init --base-model cognitivecomputations/TinyDolphin-2.8-1.1b --lora-rank 32 --device cuda

# Train with custom data
arc teach my-pack --data-path ./custom-data.jsonl --max-steps 200

# Chat with learning enabled
arc chat --max-turns 20 --learning

# Save in specific format
arc save --out ./models/my-model --format safetensors
```

## 🔧 Configuration

ARC Core uses a flexible configuration system:

```python
from arc_core import ARCConfig

config = ARCConfig()

# Model settings
config.base_model = "cognitivecomputations/TinyDolphin-2.8-1.1b"
config.context_length = 1024
config.device = "auto"

# LoRA settings
config.lora.r = 16
config.lora.alpha = 32
config.lora.dropout = 0.1

# Training settings
config.training.learning_rate = 5e-4
config.training.max_steps = 100
config.training.ewc_lambda = 0.4

# Memory settings
config.memory.working_memory_size = 10
config.memory.episodic_memory_size = 1000

# Safety settings
config.safety.enable_cognitive_inhibition = True
config.safety.enable_contextual_gating = True
config.safety.enable_metacognitive_monitoring = True

# Save configuration
config.save("my-config.json")

# Load configuration
config = ARCConfig.load("my-config.json")
```

## 🧪 Examples

### Example 1: Customer Service Bot

```python
from arc_core import ARCTrainer, ARCConfig

# Setup for customer service
config = ARCConfig()
config.safety.politeness_threshold = 0.8
config.memory.episodic_memory_size = 2000  # Remember more interactions

trainer = ARCTrainer(config)
trainer.initialize_model("cognitivecomputations/TinyDolphin-2.8-1.1b")

# Train on customer service pack (custom)
trainer.train_on_pack("customer-service-basic")

# Use in production
response = trainer.generate_response("I'm having trouble with my order")
```

### Example 2: Educational Assistant

```python
# Setup for education
config = ARCConfig()
config.safety.enable_metacognitive_monitoring = True  # Self-correction
config.memory.semantic_memory_size = 5000  # Large knowledge base

trainer = ARCTrainer(config)
trainer.initialize_model("cognitivecomputations/TinyDolphin-2.8-1.1b")

# Sequential learning
trainer.train_on_pack("math-basics")
trainer.train_on_pack("science-basics")
trainer.train_on_pack("history-basics")

# The model retains knowledge from all domains
math_response = trainer.generate_response("What is calculus?")
science_response = trainer.generate_response("Explain photosynthesis")
```

## 🔬 Research and Development

ARC Core is designed for researchers and developers working on:

- **Continual Learning**: Avoiding catastrophic forgetting in neural networks
- **Cognitive Architectures**: Biologically-inspired AI systems
- **Memory Systems**: Hierarchical and associative memory models
- **AI Safety**: Cognitive safety mechanisms and alignment
- **Human-AI Interaction**: Natural and safe conversational AI

### Extending ARC Core

```python
from arc_core.memory import MemorySystem
from arc_core.safety import SafetySystem

# Custom memory implementation
class CustomMemorySystem(MemorySystem):
    def consolidate_memories(self):
        # Custom consolidation logic
        pass

# Custom safety mechanism  
class CustomSafetySystem(SafetySystem):
    def evaluate_response(self, response):
        # Custom safety evaluation
        return safety_score
```

## 📊 Performance

ARC Core is designed to be efficient:

- **Memory Usage**: ~2-4GB RAM for medium models (with optimizations)
- **Training Speed**: ~1-5 minutes per teaching pack (100 samples)
- **Inference Speed**: ~100-500ms per response (GPU)
- **Model Size**: Base model + ~10-50MB LoRA adapters

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

### Development Setup

```bash
git clone https://github.com/metisai/arc-core.git
cd arc-core
pip install -e .[dev]
pre-commit install
```

### Running Tests

```bash
pytest tests/
```

## 📜 License

Apache License 2.0 - see [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Inspired by cognitive science research on human learning and memory
- Built on the excellent work of Hugging Face Transformers and PEFT
- Special thanks to the continual learning research community

## 📞 Support

- **Documentation**: [https://arc-core.readthedocs.io/](https://arc-core.readthedocs.io/)
- **Issues**: [GitHub Issues](https://github.com/metisai/arc-core/issues)
- **Discussions**: [GitHub Discussions](https://github.com/metisai/arc-core/discussions)
- **Email**: research@metisai.dev

---

**ARC Core** - *Enabling truly adaptive and conscious-like learning in AI systems*
