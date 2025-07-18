"""
ARC Core Minimal Usage Example

This example shows the most basic way to use the ARC Core package
for simple interaction with proper handling of return values.
"""

from arc_core import LearningARCConsciousness

def main():
    # Initialize with default model (GPT-2)
    model = LearningARCConsciousness()
    
    # Process a user interaction
    print("Sending a message to ARC...")
    
    # Note: process_user_interaction returns a tuple (response_data, reflection_data)
    response, reflection = model.process_user_interaction("Hello, how can you help me?")
    
    # Print the AI's response
    print(f"\nAI Response: {response['thought']}")
    
    # You can also access the internal reflection
    print(f"\nInternal Reflection: {reflection['thought']}")
    
    # Additional data is available in both response and reflection dictionaries
    print("\nNovel concepts discovered:", response['novel_concepts'])
    print("Learning occurred:", response['learned'])

if __name__ == "__main__":
    main()
