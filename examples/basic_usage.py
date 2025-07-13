#!/usr/bin/env python3
"""
ARC Core Basic Usage Example

This example demonstrates the basic functionality of ARC Core:
1. Initializing the system
2. Training on a teaching pack
3. Interactive conversation
4. Saving the enhanced model
"""

import os
import sys
from pathlib import Path

# Add arc_core to path if running directly
sys.path.insert(0, str(Path(__file__).parent.parent))

from arc_core import ARCTrainer, ARCConfig
from arc_core.config import create_default_config


def main():
    """Run basic ARC Core example."""
    print("[DEMO] ARC Core - Basic Usage Example")
    print("=" * 40)
    
    # 1. Create configuration
    print("\n1. Setting up ARC configuration...")
    config = create_default_config()
    config.device = "cpu"  # Use CPU for this example
    config.training.max_steps = 50  # Quick training
    
    print(f"   Device: {config.device}")
    print(f"   LoRA Rank: {config.lora.r}")
    print(f"   Training Steps: {config.training.max_steps}")
    
    # 2. Initialize trainer
    print("\n2. Initializing ARC trainer...")
    trainer = ARCTrainer(config)
    
    # Use TinyDolphin model (recommended for ARC)
    base_model = "cognitivecomputations/TinyDolphin-2.8-1.1b"
    print(f"   Loading base model: {base_model}")
    
    try:
        stats = trainer.initialize_model(base_model)
        
        print("   [OK] Model loaded successfully!")
        print(f"   Total Parameters: {stats['total_parameters']:,}")
        print(f"   Trainable Parameters: {stats['trainable_parameters']:,}")
        print(f"   Learning Percentage: {stats['learning_percentage']:.2f}%")
        
    except Exception as e:
        print(f"   [ERROR] Failed to load model: {e}")
        print("   Note: This example requires internet connection to download the model")
        return
    
    # 3. Generate initial response (before training)
    print("\n3. Testing initial response (before training)...")
    test_input = "I'm feeling really happy today!"
    
    try:
        initial_response = trainer.generate_response(test_input, max_length=50)
        print(f"   Input: {test_input}")
        print(f"   Response: {initial_response}")
    except Exception as e:
        print(f"   [ERROR] Generation failed: {e}")
        return
    
    # 4. Train on sentiment pack (if available)
    print("\n4. Training on sentiment-basic pack...")
    pack_path = Path(__file__).parent.parent / "packs" / "sentiment-basic"
    
    if pack_path.exists():
        try:
            result = trainer.train_on_pack(str(pack_path))
            
            print("   [OK] Training completed!")
            print(f"   Samples: {result['samples']}")
            print(f"   Steps: {result['steps']}")
            print(f"   Average Loss: {result['avg_loss']:.4f}")
            
        except Exception as e:
            print(f"   [ERROR] Training failed: {e}")
            return
    else:
        print("   [WARN] Sentiment pack not found, skipping training")
    
    # 5. Generate response after training
    print("\n5. Testing response after training...")
    try:
        trained_response = trainer.generate_response(test_input, max_length=50)
        print(f"   Input: {test_input}")
        print(f"   Response: {trained_response}")
        
        # Compare responses
        if initial_response != trained_response:
            print("   [CHANGED] Response changed after training!")
        else:
            print("   [SIMILAR] Response similar to initial (may need more training)")
            
    except Exception as e:
        print(f"   [ERROR] Generation failed: {e}")
        return
    
    # 6. Show memory and safety stats
    print("\n6. System statistics...")
    memory_stats = trainer.memory_system.get_system_stats()
    safety_stats = trainer.safety_system.get_safety_stats()
    
    print("   [MEMORY] Memory System:")
    print(f"      Working Memory: {memory_stats['working_memory']['active_items']}")
    print(f"      Episodic Memory: {memory_stats['episodic_memory']['count']}")
    print(f"      Semantic Concepts: {memory_stats['semantic_memory']['concepts']}")
    
    print("   [SAFETY] Safety System:")
    if 'violations' in safety_stats:
        print(f"      Total Violations: {safety_stats['violations']['total_violations']}")
    else:
        print("      No violations recorded")
    
    # 7. Interactive mode (optional)
    print("\n7. Interactive mode (optional)...")
    print("   Type messages to chat with ARC, or 'quit' to exit")
    
    while True:
        try:
            user_input = input("\n   You: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q', '']:
                break
            
            response = trainer.generate_response(user_input, max_length=80)
            print(f"   ARC: {response}")
            
            # Optional: Store the interaction
            trainer.memory_system.store_interaction({
                "input": user_input,
                "output": response,
                "context": {"mode": "interactive"}
            })
            
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"   [ERROR] Error: {e}")
    
    print("\n8. Example completed!")
    print("   [SUCCESS] You've successfully used ARC Core!")
    print("   Try the CLI commands: arc init, arc teach, arc chat")


if __name__ == "__main__":
    main()
