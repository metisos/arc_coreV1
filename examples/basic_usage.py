"""
ARC Core Basic Usage Example

This example demonstrates how to use the ARC Core package for interactive chat
and learning capabilities, including loading saved model states.

Features:
- Interactive chat with memory and learning
- Model state management (save/load)
- Basic commands for controlling the interaction
- Error handling and user feedback

Installation:
    pip install metisos-arc-core[gpu]  # For GPU support
    # or
    pip install metisos-arc-core[apple]  # For Apple Silicon

Usage:
    python examples/basic_usage.py
"""

import os
import torch
from arc_core import LearningARCConsciousness

def print_help():
    """Print available commands and usage information."""
    print("\nAvailable Commands:")
    print("  exit, quit, q - Exit the program")
    print("  help, ?      - Show this help message")
    print("  load         - Load saved model state")
    print("  save         - Save current model state")
    print("  stats        - Show model statistics")
    print("  clear        - Clear the screen")
    print("  model        - Show model information")
    print("  device       - Show current computation device")
    print()

def clear_screen():
    """Clear the terminal screen."""
    os.system('cls' if os.name == 'nt' else 'clear')

def main():
    print("ARC Core - Basic Usage Example")
    print("=" * 40)
    print("Initializing... (this may take a moment)")
    
    try:
        # Initialize the ARC model with basic configuration
        model = LearningARCConsciousness(
            model_name="gpt2",  # Start with a small model by default
            device_map="auto" if torch.cuda.is_available() else None,
            learning_rate=1e-4,
            continue_learning=True
        )
        
        print("\n[SUCCESS] Model initialized successfully!")
        print(f"Device: {next(model.transformer.model.parameters()).device}")
        print(f"Model: {model.transformer.model.name_or_path}")
        
        # Try to load a saved model state
        print("\nLooking for saved model state...")
        if model.load_learning_state():
            print("[SUCCESS] Model state loaded!")
        else:
            print("No saved state found. Starting fresh.")
        
        # Interactive chat loop
        print("\n" + "=" * 40)
        print("Type your message or a command. Type 'help' for available commands.")
        print("=" * 40 + "\n")
        
        while True:
            try:
                user_input = input("You: ").strip()
                
                # Handle commands
                if not user_input:
                    continue
                    
                if user_input.lower() in ('exit', 'quit', 'q'):
                    print("\nExiting...")
                    break
                    
                elif user_input.lower() in ('help', '?'):
                    print_help()
                    continue
                    
                elif user_input.lower() == 'clear':
                    clear_screen()
                    continue
                    
                elif user_input.lower() == 'load':
                    print("\nLoading saved model state...")
                    if model.load_learning_state():
                        print("[SUCCESS] Model state loaded!")
                    else:
                        print("[WARNING] No saved state found or failed to load")
                    continue
                    
                elif user_input.lower() == 'save':
                    print("\nSaving model state...")
                    if model.save_learning_state():
                        print("[SUCCESS] Model state saved!")
                    else:
                        print("[ERROR] Failed to save model state")
                    continue
                    
                elif user_input.lower() == 'stats':
                    print("\nModel Statistics:")
                    model.display_learning_stats()
                    continue
                    
                elif user_input.lower() == 'model':
                    print("\nModel Information:")
                    print(f"Name: {model.transformer.model.name_or_path}")
                    print(f"Class: {model.transformer.model.__class__.__name__}")
                    print(f"Trainable params: {sum(p.numel() for p in model.transformer.model.parameters() if p.requires_grad):,}")
                    print(f"Total params: {sum(p.numel() for p in model.transformer.model.parameters()):,}")
                    continue
                    
                elif user_input.lower() == 'device':
                    device = next(model.transformer.model.parameters()).device
                    print(f"\nCurrent device: {device}")
                    if torch.cuda.is_available():
                        print(f"GPU: {torch.cuda.get_device_name(0)}")
                        print(f"Memory: {torch.cuda.memory_allocated(0)/1e9:.2f}GB / {torch.cuda.get_device_properties(0).total_memory/1e9:.2f}GB")
                    continue
                
                # Process user input and get response
                print("\nARC is thinking...")
                response, reflection = model.process_user_interaction(
                    user_input,
                    max_new_tokens=150,
                    temperature=0.7,
                    do_sample=True
                )
                
                # Display the response
                if isinstance(response, dict) and 'thought' in response:
                    print(f"\nARC: {response['thought']}\n")
                    # Optionally show reflection
                    # print(f"Internal reflection: {reflection['thought']}\n")
                else:
                    print(f"\nARC: {response}\n")
                
            except KeyboardInterrupt:
                print("\n[INFO] Type 'exit' to quit or 'help' for commands")
                
            except Exception as e:
                print(f"\n[ERROR] {str(e)}")
                continue
        
        # Save the model state before exiting
        print("\nSaving model state before exit...")
        save_path = "my_arc_model_state"
        try:
            if model.save_learning_state(save_path):
                print(f"[SUCCESS] Model state saved to: {os.path.abspath(save_path)}")
            else:
                print("[WARNING] Failed to save model state")
        except Exception as e:
            print(f"[WARNING] Could not save model state: {e}")
        
    except Exception as e:
        print(f"\n[CRITICAL] Fatal error: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print("\nThank you for using ARC Core!")
    print("For more examples and documentation:")
    print("https://github.com/metisos/arc_coreV1")

if __name__ == "__main__":
    main()
