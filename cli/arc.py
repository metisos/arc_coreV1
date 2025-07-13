#!/usr/bin/env python3
"""
ARC CLI - Command Line Interface for Adaptive Recursive Consciousness Engine
"""

import click
import os
import sys
import json
from pathlib import Path
from typing import Optional, Dict, Any

# Add arc_core to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from arc_core import ARCTrainer, ARCConfig, MemorySystem, SafetySystem, __version__
from arc_core.config import create_default_config, load_teaching_pack_config


@click.group()
@click.version_option(version=__version__)
def cli():
    """ARC - Adaptive Recursive Consciousness Engine
    
    A modular continual learning system for language models.
    """
    pass


@cli.command()
@click.option('--base-model', required=True, help='Base model from Hugging Face (e.g., microsoft/DialoGPT-medium)')
@click.option('--device', default='auto', help='Device to use (auto, cpu, cuda, mps)')
@click.option('--lora-rank', default=16, help='LoRA rank parameter')
@click.option('--context-length', default=None, type=int, help='Override context length')
def init(base_model: str, device: str, lora_rank: int, context_length: Optional[int]):
    """Initialize ARC with a base model."""
    
    click.echo(f"[INIT] Initializing ARC with {base_model}")
    
    try:
        # Create config
        config = create_default_config()
        config.device = device
        config.lora.r = lora_rank
        
        if context_length:
            config.context_length = context_length
        
        # Initialize trainer
        trainer = ARCTrainer(config)
        
        # Initialize model
        click.echo("Loading model and adding LoRA adapters...")
        stats = trainer.initialize_model(base_model)
        
        # Display results
        click.echo("[SUCCESS] ARC initialized successfully!")
        click.echo(f"[STATS] Model Statistics:")
        click.echo(f"   Base Model: {stats['base_model']}")
        click.echo(f"   Context Length: {stats['context_length']:,}")
        click.echo(f"   Total Parameters: {stats['total_parameters']:,}")
        click.echo(f"   Trainable Parameters: {stats['trainable_parameters']:,}")
        click.echo(f"   Learning Percentage: {stats['learning_percentage']:.2f}%")
        click.echo(f"   Device: {stats['device']}")
        click.echo(f"   LoRA Rank: {stats['lora_rank']}")
        
        # Save config
        config.save()
        click.echo(f"[SAVE] Configuration saved to {config.config_path}")
        
    except Exception as e:
        click.echo(f"[ERROR] Initialization failed: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.option('--max-turns', default=10, help='Maximum conversation turns')
@click.option('--learning/--no-learning', default=True, help='Enable/disable learning during chat')
def chat(max_turns: int, learning: bool):
    """Interactive chat with ARC model."""
    
    click.echo("[CHAT] Starting ARC Chat Session")
    
    try:
        # Load config
        config = ARCConfig()
        if not config.base_model:
            click.echo("[ERROR] No model configured. Run 'arc init' first.", err=True)
            sys.exit(1)
        
        # Initialize trainer
        trainer = ARCTrainer(config)
        trainer.initialize_model(config.base_model)
        
        click.echo(f"[MODEL] Loaded: {config.model_name}")
        click.echo(f"[LEARN] Learning: {'Enabled' if learning else 'Disabled'}")
        click.echo("Type 'quit' to exit, 'stats' for statistics")
        click.echo("-" * 50)
        
        for turn in range(max_turns):
            try:
                user_input = input(f"\n[{turn+1}] You: ").strip()
                
                if user_input.lower() in ['quit', 'exit', 'q']:
                    break
                elif user_input.lower() == 'stats':
                    _show_system_stats(trainer)
                    continue
                elif not user_input:
                    continue
                
                # Generate response
                response = trainer.generate_response(user_input, max_length=150)
                click.echo(f"[{turn+1}] ARC: {response}")
                
                # Optional learning from positive feedback
                if learning:
                    feedback = input("    [Rate response - good/bad/skip]: ").strip().lower()
                    if feedback == 'good':
                        # Create simple learning data
                        learning_data = [{"input": user_input, "output": response}]
                        _quick_learn(trainer, learning_data)
                        click.echo("    [OK] Learning applied")
                
            except KeyboardInterrupt:
                break
            except Exception as e:
                click.echo(f"[ERROR] Error: {e}", err=True)
        
        click.echo("\n[END] Chat session ended")
        
    except Exception as e:
        click.echo(f"[ERROR] Chat failed: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.argument('pack_name')
@click.option('--data-path', help='Path to custom training data (JSONL)')
@click.option('--max-steps', default=100, help='Maximum training steps')
def teach(pack_name: str, data_path: Optional[str], max_steps: int):
    """Teach ARC using a training pack or custom data."""
    
    click.echo(f"[TEACH] Teaching ARC with: {pack_name}")
    
    try:
        # Load config
        config = ARCConfig()
        if not config.base_model:
            click.echo("[ERROR] No model configured. Run 'arc init' first.", err=True)
            sys.exit(1)
        
        config.training.max_steps = max_steps
        
        # Initialize trainer
        trainer = ARCTrainer(config)
        trainer.initialize_model(config.base_model)
        
        # Determine data path
        if data_path:
            training_path = data_path
        else:
            # Look for built-in pack
            pack_path = Path(__file__).parent.parent / "packs" / pack_name
            if not pack_path.exists():
                click.echo(f"[ERROR] Teaching pack not found: {pack_name}", err=True)
                click.echo("Available packs: sentiment-basic")
                sys.exit(1)
            training_path = str(pack_path)
        
        # Train on pack
        click.echo(f"[LOAD] Loading training data from: {training_path}")
        result = trainer.train_on_pack(training_path)
        
        # Show results
        click.echo("[SUCCESS] Teaching completed!")
        click.echo(f"[RESULTS] Training Results:")
        click.echo(f"   Pack: {result['pack']}")
        click.echo(f"   Samples: {result['samples']}")
        click.echo(f"   Steps: {result['steps']}")
        click.echo(f"   Average Loss: {result['avg_loss']:.4f}")
        click.echo(f"   Learning Rate: {result['learning_rate']}")
        
        # Save updated model
        model_path = config.get_model_save_path()
        trainer.save_model(model_path)
        click.echo(f"[SAVE] Updated model saved to: {model_path}")
        
    except Exception as e:
        click.echo(f"[ERROR] Teaching failed: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.argument('pack_name')
def test(pack_name: str):
    """Test ARC performance on a teaching pack."""
    
    click.echo(f"[TEST] Testing ARC on: {pack_name}")
    
    try:
        # Load config
        config = ARCConfig()
        if not config.base_model:
            click.echo("[ERROR] No model configured. Run 'arc init' first.", err=True)
            sys.exit(1)
        
        # Initialize trainer
        trainer = ARCTrainer(config)
        trainer.initialize_model(config.base_model)
        
        # Find test pack
        pack_path = Path(__file__).parent.parent / "packs" / pack_name
        if not pack_path.exists():
            click.echo(f"[ERROR] Test pack not found: {pack_name}", err=True)
            sys.exit(1)
        
        # Load test suite
        test_file = pack_path / "test_suite.jsonl"
        if not test_file.exists():
            click.echo(f"[ERROR] No test suite found in pack: {pack_name}", err=True)
            sys.exit(1)
        
        # Run tests
        test_cases = []
        with open(test_file, 'r') as f:
            for line in f:
                test_cases.append(json.loads(line.strip()))
        
        click.echo(f"[RUN] Running {len(test_cases)} test cases...")
        
        correct = 0
        total = len(test_cases)
        
        for i, test_case in enumerate(test_cases):
            input_text = test_case['input']
            expected_category = test_case.get('expected_category', 'positive')
            
            response = trainer.generate_response(input_text, max_length=50)
            
            # Simple sentiment classification test
            is_positive = any(word in response.lower() for word in ['good', 'great', 'positive', 'happy', 'excellent'])
            is_negative = any(word in response.lower() for word in ['bad', 'negative', 'sad', 'poor', 'terrible'])
            
            predicted_category = 'positive' if is_positive else 'negative' if is_negative else 'neutral'
            
            if predicted_category == expected_category:
                correct += 1
                status = "[PASS]"
            else:
                status = "[FAIL]"
            
            click.echo(f"  {status} Test {i+1}: {input_text[:40]}... -> {predicted_category} (expected: {expected_category})")
        
        accuracy = correct / total if total > 0 else 0
        click.echo(f"\n[RESULTS] Test Results:")
        click.echo(f"   Accuracy: {accuracy:.1%} ({correct}/{total})")
        click.echo(f"   Pack: {pack_name}")
        
    except Exception as e:
        click.echo(f"[ERROR] Testing failed: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.option('--out', required=True, help='Output path for saved model')
@click.option('--format', type=click.Choice(['safetensors', 'pytorch']), default='safetensors', help='Save format')
def save(out: str, format: str):
    """Save ARC-enhanced model."""
    
    click.echo(f"[SAVE] Saving ARC model to: {out}")
    
    try:
        # Load config
        config = ARCConfig()
        if not config.base_model:
            click.echo("[ERROR] No model configured. Run 'arc init' first.", err=True)
            sys.exit(1)
        
        # Initialize trainer
        trainer = ARCTrainer(config)
        trainer.initialize_model(config.base_model)
        
        # Save model
        saved_path = trainer.save_model(out)
        
        click.echo("[SUCCESS] Model saved successfully!")
        click.echo(f"[PATH] Location: {saved_path}")
        click.echo(f"[INFO] Includes:")
        click.echo(f"   - LoRA adapters")
        click.echo(f"   - Tokenizer")
        click.echo(f"   - ARC configuration")
        
    except Exception as e:
        click.echo(f"[ERROR] Save failed: {e}", err=True)
        sys.exit(1)


@cli.command()
def status():
    """Show ARC system status."""
    
    click.echo("[STATUS] ARC System Status")
    click.echo("=" * 30)
    
    try:
        # Load config
        config = ARCConfig()
        
        if not config.base_model:
            click.echo("[ERROR] No model configured. Run 'arc init' first.")
            return
        
        # Show basic config
        click.echo(f"[MODEL] Model Configuration:")
        click.echo(f"   Base Model: {config.base_model}")
        click.echo(f"   Model Name: {config.model_name}")
        click.echo(f"   Context Length: {config.context_length:,}")
        click.echo(f"   Device: {config.device}")
        click.echo(f"   LoRA Rank: {config.lora.r}")
        
        click.echo(f"\n[LEARN] Learning Configuration:")
        click.echo(f"   Learning Rate: {config.training.learning_rate}")
        click.echo(f"   EWC Lambda: {config.training.ewc_lambda}")
        click.echo(f"   Memory Size: W:{config.memory.working_memory_size} E:{config.memory.episodic_memory_size}")
        
        click.echo(f"\n[SAFETY] Safety Configuration:")
        click.echo(f"   Cognitive Inhibition: {'ON' if config.safety.enable_cognitive_inhibition else 'OFF'}")
        click.echo(f"   Contextual Gating: {'ON' if config.safety.enable_contextual_gating else 'OFF'}")
        click.echo(f"   Metacognitive Monitoring: {'ON' if config.safety.enable_metacognitive_monitoring else 'OFF'}")
        
        # Check for training history
        history_path = config.get_history_path()
        if os.path.exists(history_path):
            with open(history_path, 'r') as f:
                history_lines = f.readlines()
            
            click.echo(f"\n[HISTORY] Training History:")
            click.echo(f"   Total Sessions: {len(history_lines)}")
            
            if history_lines:
                latest = json.loads(history_lines[-1])
                click.echo(f"   Latest Session: {latest['timestamp'][:19]}")
                click.echo(f"   Latest Pack: {Path(latest['pack']).name}")
        
        # Check available packs
        packs_dir = Path(__file__).parent.parent / "packs"
        if packs_dir.exists():
            available_packs = [p.name for p in packs_dir.iterdir() if p.is_dir()]
            click.echo(f"\n[PACKS] Available Teaching Packs:")
            for pack in available_packs:
                click.echo(f"   - {pack}")
        
    except Exception as e:
        click.echo(f"[ERROR] Status check failed: {e}", err=True)


@cli.command()
def check():
    """Check ARC system health and requirements."""
    
    click.echo("[CHECK] ARC System Health Check")
    click.echo("=" * 35)
    
    # Check Python version
    python_version = sys.version_info
    if python_version >= (3, 8):
        click.echo(f"[OK] Python Version: {python_version.major}.{python_version.minor}.{python_version.micro}")
    else:
        click.echo(f"[ERROR] Python Version: {python_version.major}.{python_version.minor}.{python_version.micro} (requires 3.8+)")
    
    # Check dependencies
    dependencies = [
        ('torch', 'PyTorch'),
        ('transformers', 'Transformers'),
        ('peft', 'PEFT'),
        ('numpy', 'NumPy'),
        ('click', 'Click')
    ]
    
    click.echo(f"\n[DEPS] Dependencies:")
    for module, name in dependencies:
        try:
            __import__(module)
            click.echo(f"   [OK] {name}")
        except ImportError:
            click.echo(f"   [MISSING] {name} - Not installed")
    
    # Check GPU availability
    try:
        import torch
        if torch.cuda.is_available():
            click.echo(f"   [GPU] CUDA GPU: {torch.cuda.get_device_name()}")
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            click.echo(f"   [GPU] Apple Silicon GPU: Available")
        else:
            click.echo(f"   [INFO] GPU: Not available (CPU only)")
    except:
        pass
    
    # Check ARC directories
    click.echo(f"\n[DIRS] ARC Directories:")
    arc_dir = Path.home() / ".arc"
    if arc_dir.exists():
        click.echo(f"   [OK] Config Directory: {arc_dir}")
        
        config_file = arc_dir / "arc_config.json"
        if config_file.exists():
            click.echo(f"   [OK] Configuration File: Found")
        else:
            click.echo(f"   [INFO] Configuration File: Not found (run 'arc init')")
    else:
        click.echo(f"   [INFO] Config Directory: Will be created on first init")
    
    click.echo(f"\n[STATUS] Overall Status: System ready for ARC operations")


def _show_system_stats(trainer: ARCTrainer):
    """Show detailed system statistics."""
    click.echo("\n[STATS] ARC System Statistics:")
    click.echo("-" * 30)
    
    # Memory stats
    memory_stats = trainer.memory_system.get_system_stats()
    click.echo(f"[MEMORY] Memory System:")
    click.echo(f"   Working Memory: {memory_stats['working_memory']['active_items']}/{memory_stats['working_memory']['capacity']}")
    click.echo(f"   Episodic Memory: {memory_stats['episodic_memory']['count']}")
    click.echo(f"   Semantic Concepts: {memory_stats['semantic_memory']['concepts']}")
    
    # Safety stats
    safety_stats = trainer.safety_system.get_safety_stats()
    click.echo(f"[SAFETY] Safety System:")
    if 'violations' in safety_stats:
        violations = safety_stats['violations']
        click.echo(f"   Total Violations: {violations['total_violations']}")
        click.echo(f"   Recent Violations: {violations['recent_violations']}")
    
    # Training stats
    click.echo(f"[TRAIN] Training:")
    click.echo(f"   Sessions: {len(trainer.training_history)}")


def _quick_learn(trainer: ARCTrainer, learning_data: list):
    """Quick learning from feedback."""
    import tempfile
    import json
    
    # Create temporary JSONL file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
        for item in learning_data:
            f.write(json.dumps(item) + '\n')
        temp_path = f.name
    
    try:
        # Quick training
        trainer.config.training.max_steps = 5  # Very short training
        trainer.train_on_pack(os.path.dirname(temp_path), update_ewc=False)
    finally:
        # Cleanup
        os.unlink(temp_path)


if __name__ == '__main__':
    cli()
