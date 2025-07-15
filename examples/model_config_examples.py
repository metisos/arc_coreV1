"""
Example usage of ARC Core with different model architectures.

This script demonstrates how to initialize and use the ARC system with various model types,
including automatic configuration for different architectures.
"""

from arc_core import LearningARCConsciousness
import torch

def run_example(model_name, custom_lora_config=None):
    print(f"\n{'='*80}")
    print(f"RUNNING EXAMPLE WITH MODEL: {model_name}")
    print(f"{'='*80}\n")
    
    try:
        # Initialize the model with custom LoRA config if provided
        arc = LearningARCConsciousness(
            model_name=model_name,
            learning_rate=1e-4,
            lora_config=custom_lora_config,
            device_map="auto" if torch.cuda.is_available() else None
        )
        
        # Test the model with a simple interaction
        response = arc.process_user_interaction(
            "Hello, how are you?",
            max_new_tokens=100,
            temperature=0.7
        )
        
        print(f"\nResponse from {model_name}:")
        print(f"{response['thought']}\n")
        
        # Show model info and configuration
        print("\nModel Information:")
        print(f"Device: {next(arc.transformer.model.parameters()).device}")
        print(f"Model class: {arc.transformer.model.__class__.__name__}")
        print(f"Trainable parameters: {sum(p.numel() for p in arc.transformer.model.parameters() if p.requires_grad):,}")
        print(f"Total parameters: {sum(p.numel() for p in arc.transformer.model.parameters()):,}")
        
        # Show LoRA configuration if available
        if hasattr(arc.transformer.model, 'peft_config'):
            print("\nLoRA Configuration:")
            for name, config in arc.transformer.model.peft_config.items():
                print(f"- {name}: {config}")
        
        return True
        
    except Exception as e:
        print(f"\nError with {model_name}: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def main():
    # Example 1: GPT-2 (small model for testing)
    print("\n=== Example 1: GPT-2 ===")
    run_example("gpt2")
    
    # Example 2: LLaMA model (uncomment if you have access)
    # print("\n=== Example 2: LLaMA ===")
    # run_example("meta-llama/Llama-2-7b-hf")
    
    # Example 3: DeepSeek model
    print("\n=== Example 3: DeepSeek ===")
    run_example("deepseek-ai/deepseek-llm-7b")
    
    # Example 4: Custom model with explicit LoRA config
    print("\n=== Example 4: Custom Configuration ===")
    custom_config = {
        'r': 8,  # Lower rank for faster training
        'lora_alpha': 16,
        'lora_dropout': 0.05,
        'target_modules': ['q_proj', 'v_proj'],  # Explicitly set for known architecture
        'task_type': 'CAUSAL_LM'
    }
    run_example("cognitivecomputations/TinyDolphin-2.8-1.1b", custom_lora_config=custom_config)

if __name__ == "__main__":
    main()
