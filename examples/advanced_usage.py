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
import torch
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
    
    # Model configuration with advanced options
    model_config = {
        "model_name": "gpt2",  # or "deepseek-ai/deepseek-llm-7b" for larger models
        "learning_rate": 1e-4,
        "continue_learning": True,
        "device_map": "auto" if torch.cuda.is_available() else None,
        "lora_config": {
            'r': 16,  # Rank
            'lora_alpha': 32,
            'lora_dropout': 0.1,
            'task_type': 'CAUSAL_LM',
            'target_modules': None  # Auto-detect
        },
        "max_memory": {0: '20GB'} if torch.cuda.is_available() else None
    }
    
    print("Initializing ARC with configuration:")
    print(json.dumps(model_config, indent=2, default=str))
    
    try:
        model = LearningARCConsciousness(
            model_name=model_config["model_name"],
            learning_rate=model_config["learning_rate"],
            continue_learning=model_config["continue_learning"],
            lora_config=model_config["lora_config"],
            device_map=model_config["device_map"],
            max_memory=model_config["max_memory"]
        )
        print("\n[SUCCESS] Model initialized with custom configuration")
        
        # Print model info
        print("\nModel Information:")
        print(f"Device: {next(model.transformer.model.parameters()).device}")
        print(f"Model class: {model.transformer.model.__class__.__name__}")
        print(f"Trainable parameters: {sum(p.numel() for p in model.transformer.model.parameters() if p.requires_grad):,}")
        print(f"Total parameters: {sum(p.numel() for p in model.transformer.model.parameters()):,}")
        
    except Exception as e:
        print(f"\n[ERROR] Failed to initialize model: {e}")
        import traceback
        traceback.print_exc()
        return

    # 2. Model State Management
    print_section("2. Model State Management")
    
    # Try to load existing state
    print("\nAttempting to load existing model state...")
    try:
        if model.load_learning_state():
            print("[SUCCESS] Existing state loaded successfully!")
        else:
            print("No existing state found - starting fresh")
            
        # Save a test state
        test_save_path = "test_model_state"
        print(f"\nSaving test state to: {test_save_path}")
        model.save_learning_state(test_save_path)
        print("[SUCCESS] Test state saved successfully!")
        
    except Exception as e:
        print(f"[ERROR] State management operation failed: {e}")
        return
    
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
        response, reflection = model.process_user_interaction(phrase)
        
        if isinstance(response, dict):
            print("\nResponse:")
            print(f"- Thought: {response.get('thought', 'N/A')}")
            print(f"- Confidence: {response.get('confidence', 'N/A'):.2f}")
            
            if 'novel_concepts' in response and response['novel_concepts']:
                print(f"- Novel Concepts: {', '.join(response['novel_concepts'])}")
                
            # Optionally show reflection data
            # print("\nInternal Reflection:")
            # print(f"- Thought: {reflection.get('thought', 'N/A')}")
    
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
        response, reflection = model.process_user_interaction(f"{question} {answer}")
        print(f"Learning status: {'Success' if response and 'learned' in response and response['learned'] else 'Failed'}")
    
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
        response, reflection = new_model.process_user_interaction(test_question)
        
        if isinstance(response, dict) and 'thought' in response:
            print(f"Response: {response['thought']}")
            # Optionally display reflection
            # print(f"Internal reflection: {reflection['thought']}")
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