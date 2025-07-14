"""
ARC Core Basic Usage Example

This example demonstrates how to use the ARC Core package for interactive chat
and learning capabilities, including loading saved model states.

Installation:
    pip install metisos-arc-core

Usage:
    python examples/basic_usage.py
"""

import os
from arc_core import LearningARCConsciousness

def main():
    print("ARC Core - Basic Usage Example")
    print("=" * 40)
    print()
    
    # Initialize the ARC model
    print("Initializing ARC model...")
    try:
        model = LearningARCConsciousness()
        print("[SUCCESS] Model initialized successfully!")
    except Exception as e:
        print(f"[ERROR] Failed to initialize model: {e}")
        return
    
    # Load a saved model state
    print("\nLoading saved model state...")
    if model.load_learning_state():
        print("[SUCCESS] Model state loaded successfully!")
    else:
        print("[ERROR] Failed to load model state")
    
    # Interactive chat loop
    print("\nStarting interactive chat. Type 'exit' to quit.")
    print("Commands: 'exit', 'load', 'save', 'stats'")
    print("You can ask questions or give instructions to the model.\n")
    
    while True:
        try:
            user_input = input("You: ")
            if user_input.lower() in ('exit', 'quit', 'q'):
                break
            elif user_input.lower() == 'load':
                print("\nLoading saved model state...")
                if model.load_learning_state():
                    print("[SUCCESS] Model state loaded successfully!")
                else:
                    print("[ERROR] Failed to load model state")
            elif user_input.lower() == 'save':
                print("\nSaving model state...")
                if model.save_learning_state():
                    print("[SUCCESS] Model state saved successfully!")
                else:
                    print("[ERROR] Failed to save model state")
            elif user_input.lower() == 'stats':
                print("\nDisplaying model statistics:")
                model.display_learning_stats()
                
            # Process the input and get response
            response = model.process_user_interaction(user_input)
            
            # Extract and display the response
            if isinstance(response, dict) and 'thought' in response:
                print(f"\nARC: {response['thought']}\n")
            else:
                print(f"\nARC: {response}\n")
                
        except KeyboardInterrupt:
            print("\n[INFO] Exiting chat...")
            break
        except Exception as e:
            print(f"\n[ERROR] {str(e)}")
            continue
    
    # Save the model state
    save_path = "my_arc_model_state.json"
    try:
        model.save_learning_state(save_path)
        print(f"\n[SUCCESS] Model state saved to {os.path.abspath(save_path)}")
    except Exception as e:
        print(f"\n[WARNING] Could not save model state: {e}")
    
    print("\nExample completed successfully!")
    print("For more examples and documentation:")
    print("https://github.com/metisos/arc_coreV1")

if __name__ == "__main__":
    main()
