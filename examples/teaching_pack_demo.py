"""
ARC Core Teaching Pack Demo

This example demonstrates how to use teaching packs with ARC Core
for specialized training and evaluation.

Key Features Demonstrated:
- Listing available teaching packs
- Installing and using teaching packs
- Training the model with a teaching pack
- Testing the model's learning

Prerequisites:
    pip install metisos-arc-core
"""

import os
import sys
from arc_core import LearningARCConsciousness

def print_header(title):
    """Print a formatted header."""
    print("\n" + "=" * 80)
    print(f"{title:^80}")
    print("=" * 80 + "\n")

def run_cli_command(command):
    """Run a CLI command and print the output."""
    print(f"$ arc {command}")
    os.system(f"arc {command}")

def main():
    print_header("ARC Core Teaching Pack Demo")
    
    # Show available teaching packs
    print_header("1. Available Teaching Packs")
    run_cli_command("pack list")
    
    # Install a teaching pack if not already installed
    pack_name = "sentiment-basic"
    print_header(f"2. Installing Teaching Pack: {pack_name}")
    run_cli_command(f"pack install {pack_name}")
    
    # Initialize the model
    print_header("3. Initializing ARC Model")
    try:
        model = LearningARCConsciousness()
        print("[SUCCESS] Model initialized successfully!")
    except Exception as e:
        print(f"[ERROR] Failed to initialize model: {e}")
        return
    
    # Train using the teaching pack via CLI
    print_header(f"4. Training with {pack_name} (CLI)")
    run_cli_command(f"teach {pack_name}")
    
    # Test the model's learning
    print_header("5. Testing the Model")
    test_phrases = [
        "I love this product!",
        "This is terrible, I hate it.",
        "It's okay, not great but not bad either."
    ]
    
    for phrase in test_phrases:
        print(f"\nTesting phrase: {phrase}")
        response = model.process_user_interaction(phrase)
        if isinstance(response, dict) and 'thought' in response:
            print(f"ARC: {response['thought']}")
        else:
            print(f"ARC: {response}")
    
    # Save the trained model
    save_path = "trained_arc_model.json"
    try:
        model.save_learning_state(save_path)
        print(f"\n[SUCCESS] Trained model saved to {os.path.abspath(save_path)}")
    except Exception as e:
        print(f"\n[WARNING] Could not save model: {e}")
    
    print("\n" + "=" * 80)
    print("Teaching Pack Demo Completed!")
    print("=" * 80)
    print("\nNext steps:")
    print(f"1. Try the interactive chat: arc chat")
    print(f"2. Check model status: arc status")
    print(f"3. Explore more teaching packs: arc pack list")

if __name__ == "__main__":
    main()
