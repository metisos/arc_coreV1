"""
ARC Core Basic Usage Example

This example shows how to use ARC Core after installation.

Installation:
    pip install metisos-arc-core

Usage:
    python examples/basic_usage.py
"""

def main():
    print("ARC Core - Basic Usage Example")
    print("=" * 40)
    print()
    print("To run this example, install ARC Core:")
    print("  pip install metisos-arc-core")
    print()
    print("Example usage after installation:")
    print()
    
    example_code = '''
# Import ARC Core components
from arc_core import ARCConfig, ARCTrainer, create_default_config

# Create configuration
config = create_default_config()

# Initialize trainer  
trainer = ARCTrainer(config)

# Initialize with TinyDolphin model (recommended)
trainer.initialize_model("cognitivecomputations/TinyDolphin-2.8-1.1b")

# Train on a teaching pack
trainer.train_on_pack("sentiment-basic")

# Generate enhanced responses
response = trainer.generate_response("How are you feeling today?")
print(response)

# Save the enhanced model
trainer.save_model("./my_arc_model")
'''
    
    print(example_code)
    print()
    print("For more examples and documentation:")
    print("https://github.com/metisos/arc_coreV1")

if __name__ == "__main__":
    main()
