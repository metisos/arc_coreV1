#!/usr/bin/env python3
"""
ARC Core Interactive Demo - Try It Yourself!
============================================

This script lets you experience ARC's continual learning capabilities firsthand.
Perfect for testing ARC from a GitHub repository.

Usage:
    python try_arc.py

Requirements:
    pip install metisos-arc-core

What this demonstrates:
- Real-time continual learning
- Interactive chat with learning
- Memory persistence across conversations
- LoRA adapter training on the fly

GitHub: https://github.com/your-username/arc-core
PyPI: https://pypi.org/project/metisos-arc-core/
"""

import sys
import os
import time
import json
from pathlib import Path

def check_dependencies():
    """Check if required dependencies are installed."""
    try:
        import torch
        import transformers
        import peft
        from arc_core import ARCConfig, ARCTrainer
        return True, None
    except ImportError as e:
        return False, str(e)

def print_header():
    """Print welcome header."""
    print("=" * 70)
    print("  ARC CORE - Adaptive Recursive Consciousness")
    print("  Interactive Continual Learning Demo")
    print("=" * 70)
    print()
    print("Welcome! This demo shows ARC's ability to learn continuously")
    print("while maintaining its base knowledge. You'll chat with ARC,")
    print("teach it new information, and see it adapt in real-time.")
    print()

def print_section(title, char="="):
    """Print a formatted section header."""
    print(f"\n{char * 50}")
    print(f" {title}")
    print(f"{char * 50}")

def get_user_input(prompt="You: "):
    """Get user input with error handling."""
    try:
        return input(prompt).strip()
    except KeyboardInterrupt:
        print("\n\nGoodbye! Thanks for trying ARC!")
        sys.exit(0)
    except EOFError:
        return ""

def initialize_arc():
    """Initialize ARC system with error handling."""
    print_section("Initializing ARC System")
    print("Loading GPT-2 and setting up LoRA adapters...")
    print("(This may take a moment on first run)")
    
    try:
        config = ARCConfig()
        config.model_name = "gpt2"
        config.learning_rate = 1e-4
        config.biological_learning_rate = 0.03
        
        trainer = ARCTrainer(config)
        print("Downloading model files if needed...")
        success = trainer.initialize_model()
        
        if success:
            print("[SUCCESS] ARC system ready!")
            print(f"Base model: {config.model_name}")
            print("LoRA adapters: Active and ready to learn")
            return trainer
        else:
            print("[ERROR] Failed to initialize ARC system")
            return None
            
    except Exception as e:
        print(f"[ERROR] Initialization failed: {e}")
        print("\nTroubleshooting:")
        print("1. Make sure you have: pip install metisos-arc-core")
        print("2. Check your internet connection (for model download)")
        print("3. Ensure you have enough disk space (~600MB)")
        return None

def chat_with_arc(trainer):
    """Interactive chat session with learning."""
    print_section("Interactive Chat with Learning")
    print("Chat with ARC and watch it learn from your conversations!")
    print("Commands:")
    print("  'teach <question> | <answer>' - Explicitly teach ARC")
    print("  'save' - Save learned model")
    print("  'stats' - Show learning statistics")
    print("  'quit' - Exit")
    print("\nStart chatting:")
    
    conversation_count = 0
    learning_count = 0
    
    while True:
        user_input = get_user_input("\nYou: ")
        
        if not user_input:
            continue
            
        if user_input.lower() == 'quit':
            break
            
        elif user_input.lower() == 'save':
            save_learned_model(trainer)
            continue
            
        elif user_input.lower() == 'stats':
            show_statistics(trainer, conversation_count, learning_count)
            continue
            
        elif user_input.lower().startswith('teach '):
            # Explicit teaching mode
            teach_content = user_input[6:].strip()
            if '|' in teach_content:
                question, answer = teach_content.split('|', 1)
                question = question.strip()
                answer = answer.strip()
                
                print(f"ARC: Learning that '{question}' -> '{answer}'")
                success = trainer.transformer.learn_from_interaction(question, answer)
                
                if success:
                    learning_count += 1
                    print("[LEARNED] Teaching successful - ARC has updated its understanding")
                else:
                    print("[ERROR] Teaching failed")
            else:
                print("Teaching format: teach <question> | <answer>")
            continue
        
        # Regular conversation
        print("ARC: ", end="", flush=True)
        start_time = time.time()
        
        try:
            response = trainer.generate_response(
                user_input, 
                max_length=150, 
                temperature=0.8,
                do_sample=True
            )
            
            generation_time = time.time() - start_time
            print(response)
            print(f"[Generated in {generation_time:.2f}s]")
            
            # Optional: Learn from this interaction (experimental)
            # This makes ARC learn from the conversation flow
            if len(response) > 10 and "not initialized" not in response.lower():
                # Only learn from successful responses
                learn_success = trainer.transformer.learn_from_interaction(
                    user_input, 
                    response
                )
                if learn_success:
                    learning_count += 1
                    print("[Background learning applied]")
            
            conversation_count += 1
            
        except Exception as e:
            print(f"[ERROR] Response generation failed: {e}")

def save_learned_model(trainer):
    """Save the learned model."""
    print_section("Saving Learned Model", "-")
    
    save_path = Path("./arc_session_model")
    print(f"Saving to: {save_path}")
    
    try:
        success = trainer.save_model(str(save_path))
        if success:
            # Show what was saved
            if save_path.exists():
                total_size = sum(f.stat().st_size for f in save_path.rglob('*') if f.is_file())
                size_mb = total_size / (1024 * 1024)
                print(f"[SUCCESS] Model saved ({size_mb:.1f}MB)")
                print("Saved components:")
                print(f"  - LoRA adapters: {save_path}/lora_adapter/")
                print(f"  - Learning stats: {save_path}/lora_adapter/learning_stats.json")
                print(f"  - Training stats: {save_path}/training_stats.json")
            else:
                print("[SUCCESS] Model components saved")
        else:
            print("[ERROR] Failed to save model")
    except Exception as e:
        print(f"[ERROR] Save failed: {e}")

def show_statistics(trainer, conversation_count, learning_count):
    """Display learning and conversation statistics."""
    print_section("ARC Learning Statistics", "-")
    
    print(f"Session Statistics:")
    print(f"  Conversations: {conversation_count}")
    print(f"  Learning updates: {learning_count}")
    
    if hasattr(trainer.transformer, 'learning_stats'):
        stats = trainer.transformer.learning_stats
        print(f"\nModel Learning Stats:")
        for key, value in stats.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.4f}")
            else:
                print(f"  {key}: {value}")
    
    if hasattr(trainer.transformer, 'total_updates'):
        print(f"\nTotal learning updates: {trainer.transformer.total_updates}")
    
    # Memory usage info
    try:
        import psutil
        process = psutil.Process()
        memory_mb = process.memory_info().rss / 1024 / 1024
        print(f"Memory usage: {memory_mb:.1f}MB")
    except:
        pass

def run_quick_demo(trainer):
    """Run a quick automated demo before interactive mode."""
    print_section("Quick Demo - Watching ARC Learn")
    
    demo_interactions = [
        ("What is your name?", "I am ARC, an Adaptive Recursive Consciousness system."),
        ("What can you do?", "I can learn continuously while maintaining my knowledge base."),
        ("How do you learn?", "I use LoRA adapters to update my understanding without changing my base model.")
    ]
    
    print("Teaching ARC some facts about itself...")
    
    for i, (question, answer) in enumerate(demo_interactions, 1):
        print(f"\n[Demo {i}/3] Teaching: {question}")
        print(f"          Answer: {answer}")
        
        success = trainer.transformer.learn_from_interaction(question, answer)
        if success:
            print("[SUCCESS] Learning applied")
        else:
            print("[ERROR] Learning failed")
        
        time.sleep(0.5)  # Brief pause for readability
    
    print("\nNow let's test what ARC learned:")
    test_response = trainer.generate_response("What is your name?", max_length=50)
    print(f"ARC: {test_response}")

def main():
    """Main demo function."""
    print_header()
    
    # Check dependencies
    deps_ok, error = check_dependencies()
    if not deps_ok:
        print("[ERROR] Missing dependencies!")
        print(f"Error: {error}")
        print("\nPlease install ARC Core:")
        print("  pip install metisos-arc-core")
        print("\nOr install with all dependencies:")
        print("  pip install torch transformers peft metisos-arc-core")
        sys.exit(1)
    
    # Initialize ARC
    trainer = initialize_arc()
    if not trainer:
        print("Failed to initialize. Please check the troubleshooting steps above.")
        sys.exit(1)
    
    # Run quick demo
    try:
        run_quick_demo(trainer)
    except Exception as e:
        print(f"Demo error: {e}")
        print("Continuing to interactive mode...")
    
    # Interactive chat
    try:
        chat_with_arc(trainer)
    except Exception as e:
        print(f"\nChat session error: {e}")
    
    # Final stats
    print_section("Session Complete")
    print("Thank you for trying ARC!")
    print("\nTo learn more:")
    print("- GitHub: https://github.com/your-username/arc-core")
    print("- PyPI: https://pypi.org/project/metisos-arc-core/")
    print("- Documentation: Check the README.md")
    
    # Clean up info
    if Path("./arc_session_model").exists():
        print(f"\nYour learned model was saved in: ./arc_session_model/")
        print("You can load this model later to continue from where you left off.")

if __name__ == "__main__":
    main()
