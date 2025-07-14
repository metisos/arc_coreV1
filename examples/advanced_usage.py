"""
ARC Core Advanced Usage Example

This example demonstrates advanced usage of the ARC Core package,
including custom configuration, training, and model management.

Key Features Demonstrated:
- Custom model initialization
- Advanced configuration
- Training and fine-tuning
- Model state management (save/load)
- Direct access to biological learning mechanisms
- Learning statistics and monitoring
"""

import os
import json
from datetime import datetime
from arc_core import LearningARCConsciousness

def print_section(title):
    """Print a formatted section header."""
    print("\n" + "=" * 80)
    print(f" {title.upper()} ".center(80, "#"))
    print("=" * 80)

def main():
    print_section("ARC Core Advanced Usage Example")
    
    # 1. Custom Configuration
    print_section("1. Custom Configuration")
    
    # Model configuration
    model_config = {
        "model_name": "gpt2",  # or "cognitivecomputations/TinyDolphin-2.8-1.1b"
        "learning_rate": 0.0001,
        "continue_learning": True
    }
    
    print("Initializing ARC with configuration:")
    print(json.dumps(model_config, indent=2))
    
    try:
        model = LearningARCConsciousness(
            model_name=model_config["model_name"],
            learning_rate=model_config["learning_rate"],
            continue_learning=model_config["continue_learning"]
        )
        print("\n[SUCCESS] Model initialized with custom configuration")
    except Exception as e:
        print(f"\n[ERROR] Failed to initialize model: {e}")
        return  # This return is now properly inside the function

    # 2. Model State Management
    print_section("2. Model State Management")
    
    # Try to load existing state
    print("\nAttempting to load existing model state...")
    if model.load_learning_state():
        print("[SUCCESS] Existing state loaded successfully!")
    else:
        print("No existing state found - starting fresh")
    
    # Save initial state
    print("\nSaving initial model state...")
    if model.save_learning_state():
        print("[SUCCESS] Initial state saved successfully!")
    else:
        print("[ERROR] Failed to save initial state")
        
    # 2. Basic Interaction
    print_section("2. Basic Interaction")
    test_phrases = [
        "Hello, how are you?",
        "What can you tell me about machine learning?",
        "Explain quantum computing in simple terms."
    ]
    
    for phrase in test_phrases:
        print(f"\nInput: {phrase}")
        response = model.process_user_interaction(phrase)
        
        if isinstance(response, dict):
            print("\nResponse:")
            print(f"- Thought: {response.get('thought', 'N/A')}")
            print(f"- Confidence: {response.get('confidence', 'N/A'):.2f}")
            
            if 'novel_concepts' in response and response['novel_concepts']:
                print(f"- Novel Concepts: {', '.join(response['novel_concepts'])}")
    
    # 3. Training and Learning
    print_section("3. Training and Learning")
    
    training_data = [
        ("What is the capital of France?", "The capital of France is Paris."),
        ("Who wrote Romeo and Juliet?", "William Shakespeare wrote Romeo and Juliet."),
        ("What is the square root of 64?", "The square root of 64 is 8.")
    ]
    
    print("Training the model with custom data...")
    for question, answer in training_data:
        print(f"\nTeaching: Q: {question} | A: {answer}")
        response = model.process_user_interaction(f"{question} {answer}")
        print(f"Learning status: {'Success' if response else 'Failed'}")
    
    # 4. Model Persistence
    print_section("4. Model Persistence")
    
    # Create a timestamped directory for saving
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = f"arc_model_{timestamp}"
    os.makedirs(save_dir, exist_ok=True)
    
    # Save model state
    model_path = os.path.join(save_dir, "model_state.json")
    config_path = os.path.join(save_dir, "config.json")
    
    try:
        # Save model state
        model.save_learning_state(model_path)
        
        # Save configuration
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"\n[SUCCESS] Model saved to: {os.path.abspath(save_dir)}")
        print(f"- Model state: {model_path}")
        print(f"- Configuration: {config_path}")
        
    except Exception as e:
        print(f"\n[ERROR] Failed to save model: {e}")
    
    # 5. Loading a Saved Model
    print_section("5. Loading a Saved Model")
    
    try:
        print(f"Loading model from: {model_path}")
        new_model = LearningARCConsciousness()
        new_model.load_learning_state(model_path)
        
        # Test the loaded model
        test_question = "What is the capital of France?"
        print(f"\nTesting loaded model with: {test_question}")
        response = new_model.process_user_interaction(test_question)
        
        if isinstance(response, dict) and 'thought' in response:
            print(f"Response: {response['thought']}")
        else:
            print(f"Response: {response}")
            
        print("\n[SUCCESS] Model loaded and tested successfully!")
        
    except Exception as e:
        print(f"\n[ERROR] Failed to load model: {e}")
    
    print_section("Example Completed")
    print("\nNext steps:")
    print("1. Explore the saved model directory")
    print("2. Try the interactive chat with your trained model")
    print("3. Check out the teaching pack examples")

if __name__ == "__main__":
    main()