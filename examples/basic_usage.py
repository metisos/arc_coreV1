"""
ARC Core Basic Usage Example

This example demonstrates how to use the ARC Core package for interactive chat
and learning capabilities.

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
    
    # Interactive chat loop
    print("\nStarting interactive chat. Type 'exit' to quit.")
    print("You can ask questions or give instructions to the model.\n")
    
    while True:
        try:
            user_input = input("You: ")
            if user_input.lower() in ('exit', 'quit', 'q'):
                break
                
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
