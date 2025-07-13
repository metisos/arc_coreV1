"""
ARC Core CLI Interface

This is a stub file showing the CLI interface.
The actual CLI implementation is provided via the PyPI package installation.

Install: pip install metisos-arc-core

Available commands after installation:
- arc init --base-model MODEL_NAME    # Initialize ARC with a base model
- arc chat [--learning]               # Start interactive chat session  
- arc teach PACK_NAME                 # Train using a teaching pack
- arc test [--pack PACK_NAME]         # Test model performance
- arc save --path PATH                # Save model to specified path
- arc status                          # Show current model status
- arc check                           # Health check and system info

For full documentation: https://github.com/metisos/arc_coreV1
"""

def main():
    """
    Main CLI entry point.
    
    This is a stub - install the package to use CLI:
    pip install metisos-arc-core
    """
    print("ARC Core CLI")
    print("=" * 50)
    print("This is the public interface repository.")
    print("To use ARC Core, install the package:")
    print()
    print("  pip install metisos-arc-core")
    print()
    print("Then use the 'arc' command:")
    print("  arc check                    # System health check")
    print("  arc init --base-model MODEL  # Initialize with model")
    print()
    print("For documentation: https://github.com/metisos/arc_coreV1")

if __name__ == "__main__":
    main()
