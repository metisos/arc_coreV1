# Getting Started with ARC Core

This guide will walk you through setting up and using ARC Core for the first time.

## Prerequisites

- Python 3.8 or higher
- 4GB+ RAM (8GB+ recommended)
- Internet connection for downloading models

## Installation

### Option 1: Install from PyPI (Recommended)

```bash
pip install metisos-arc-core
```

### Option 2: Install with GPU Support

For NVIDIA GPUs:
```bash
pip install metisos-arc-core[gpu]
```

For Apple Silicon (M1/M2):
```bash
pip install metisos-arc-core[apple]
```

### Option 3: Development Installation

```bash
git clone https://github.com/metisos/arc_coreV1.git
cd arc_coreV1
pip install -e .[dev]
```

## Quick Start Tutorial

### Step 1: Verify Installation

```bash
arc check
```

You should see a health check showing all dependencies are installed correctly.

### Step 2: Initialize ARC with a Base Model

Choose a base model to enhance with ARC capabilities:

```bash
# Recommended for beginners - TinyDolphin (optimized for ARC)
arc init --base-model cognitivecomputations/TinyDolphin-2.8-1.1b

# Alternative options:
# Small, fast model
arc init --base-model distilgpt2

# Dialog-focused model  
arc init --base-model microsoft/DialoGPT-medium
```

**Note**: The first run will download the model from Hugging Face, which may take a few minutes.

### Step 3: Check System Status

```bash
arc status
```

This shows your current configuration and available teaching packs.

### Step 4: Train on a Teaching Pack

ARC Core includes built-in teaching packs for common tasks:

```bash
# Train on sentiment analysis
arc teach sentiment-basic
```

### Step 5: Test the Enhanced Model

```bash
# Test performance on the teaching pack
arc test sentiment-basic
```

### Step 6: Interactive Chat

```bash
# Chat with your ARC-enhanced model
arc chat
```

Try these example inputs:
- "I'm feeling really happy today!"
- "This situation is making me frustrated."
- "I'm worried about my exam tomorrow."

### Step 7: Save Your Enhanced Model

```bash
arc save --out ./my-arc-model
```

## Understanding ARC Core

### What Makes ARC Different?

ARC Core implements several biologically-inspired learning mechanisms:

1. **Continual Learning**: Models learn new tasks without forgetting old ones
2. **Memory Systems**: Hierarchical memory (working, episodic, semantic)
3. **Safety Mechanisms**: Cognitive inhibition and contextual gating
4. **Adaptive Responses**: Context-aware and emotionally appropriate responses

### Key Concepts

#### Teaching Packs
Teaching packs are modular training datasets that teach specific skills:
- **sentiment-basic**: Emotional understanding and appropriate responses
- Custom packs can be created for your specific needs

#### Memory Systems
- **Working Memory**: Short-term context (last few interactions)
- **Episodic Memory**: Specific conversation memories
- **Semantic Memory**: General knowledge and concepts learned

#### Safety Systems
- **Cognitive Inhibition**: Filters inappropriate responses
- **Contextual Gating**: Controls what gets remembered
- **Metacognitive Monitoring**: Self-correction and quality assessment

## Common Use Cases

### 1. Customer Service Bot

```bash
# Initialize with a conversational model
arc init --base-model microsoft/DialoGPT-medium

# Create custom training data for your domain
# Then train: arc teach customer-service-pack
```

### 2. Educational Assistant

```bash
# Use a model good at explanations
arc init --base-model microsoft/DialoGPT-medium

# Train on educational content
# Sequential learning: arc teach math-basics, science-basics, etc.
```

### 3. Emotional Support Chatbot

```bash
# Focus on empathy and emotional understanding
arc init --base-model microsoft/DialoGPT-medium

# Train on emotional intelligence
arc teach sentiment-basic
```

### 4. Domain-Specific Expert

```bash
# Initialize and train on your domain-specific data
arc teach my-domain-pack --data-path ./my-training-data.jsonl
```

## Creating Custom Teaching Packs

### 1. Create Pack Directory Structure

```
my-pack/
├── pack.yml          # Metadata
├── training.jsonl    # Training conversations
└── test_suite.jsonl  # Test cases
```

### 2. Create pack.yml

```yaml
name: my-pack
version: 1.0.0
description: My custom training pack
author: Your Name

learning_objectives:
  - Learn specific domain knowledge
  - Provide helpful responses

datasets:
  training: training.jsonl
  
test_suite: test_suite.jsonl
```

### 3. Create Training Data (training.jsonl)

```json
{"input": "User question or statement", "output": "Desired model response"}
{"input": "How do I use ARC Core?", "output": "ARC Core is a continual learning system. Start with 'arc init' to set up a base model."}
```

### 4. Create Test Cases (test_suite.jsonl)

```json
{"input": "Test question", "expected_category": "positive"}
{"input": "Another test", "expected_category": "negative"}
```

### 5. Train Your Model

```bash
arc teach my-pack
arc test my-pack
```

## Advanced Configuration

### Python API

```python
from arc_core import ARCTrainer, ARCConfig

# Custom configuration
config = ARCConfig()
config.device = "cuda"  # Use GPU
config.lora.r = 32      # Higher LoRA rank for more capacity
config.training.max_steps = 200  # More training steps

# Initialize trainer
trainer = ARCTrainer(config)
trainer.initialize_model("microsoft/DialoGPT-medium")

# Custom training
result = trainer.train_on_pack("my-pack")

# Generate responses
response = trainer.generate_response("Hello!")
```

### Configuration Options

Key configuration parameters you can adjust:

```python
# Model settings
config.context_length = 1024    # Context window size
config.device = "auto"          # Device selection

# LoRA settings (controls learning capacity)
config.lora.r = 16             # Rank (higher = more capacity)
config.lora.alpha = 32         # Scaling factor
config.lora.dropout = 0.1      # Regularization

# Training settings
config.training.learning_rate = 5e-4    # Learning speed
config.training.max_steps = 100         # Training duration
config.training.ewc_lambda = 0.4        # Forgetting prevention

# Memory settings
config.memory.working_memory_size = 10      # Short-term memory
config.memory.episodic_memory_size = 1000   # Conversation memory
config.memory.semantic_memory_size = 10000  # Knowledge memory

# Safety settings
config.safety.enable_cognitive_inhibition = True      # Response filtering
config.safety.enable_contextual_gating = True         # Memory control
config.safety.enable_metacognitive_monitoring = True  # Self-correction
```

## Troubleshooting

### Common Issues

#### "No module named 'arc_core'"
```bash
# Ensure you're in the right environment and ARC is installed
pip install metisos-arc-core
```

#### "CUDA out of memory"
```bash
# Use a smaller model or switch to CPU
arc init --base-model distilgpt2 --device cpu
```

#### "No model configured"
```bash
# Initialize ARC first
arc init --base-model distilgpt2
```

#### "Teaching pack not found"
```bash
# Check available packs
arc status

# Create the pack directory in the right location
# Or use --data-path to specify custom location
```

### Performance Tips

1. **Start Small**: Use `distilgpt2` for testing, upgrade to larger models later
2. **Use GPU**: Install with `[gpu]` extra for much faster training
3. **Batch Training**: Train on multiple packs sequentially for comprehensive learning
4. **Monitor Memory**: Use `arc status` to check memory usage and statistics

### Getting Help

- **Documentation**: Check the README.md and inline help (`arc --help`)
- **Examples**: Look at the `examples/` directory
- **Issues**: Report bugs on GitHub
- **Discussions**: Join community discussions for tips and use cases

## What's Next?

Now that you have ARC Core running:

1. **Experiment** with different base models and teaching packs
2. **Create** custom teaching packs for your specific needs
3. **Integrate** ARC Core into your applications using the Python API
4. **Monitor** your model's learning and memory systems
5. **Share** your custom teaching packs with the community

Welcome to the future of adaptive AI learning!
