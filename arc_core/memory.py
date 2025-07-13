"""
ARC Core Memory Systems Interface

This module provides the public API for ARC memory functionality.
The actual implementation is provided via the PyPI package installation.

Install: pip install metisos-arc-core
"""

class MemorySystem:
    """
    Hierarchical memory system for ARC Core.
    
    Features:
    - Working Memory: Short-term context processing
    - Episodic Memory: Interaction history with temporal context
    - Semantic Memory: Extracted knowledge patterns
    - Contextual Gating: Hippocampus-like memory encoding
    - Sleep-like Consolidation: Memory strengthening
    
    Install the package to access full functionality:
    pip install metisos-arc-core
    """
    def __init__(self, config):
        raise ImportError(
            "ARC Core implementation not found. Please install the package:\n"
            "pip install metisos-arc-core"
        )
    
    def add_interaction(self, input_text, output_text, context=None):
        """Add interaction to episodic memory."""
        pass
    
    def get_relevant_memories(self, query, memory_type="all"):
        """Retrieve relevant memories for context."""
        pass
    
    def consolidate_memories(self):
        """Perform sleep-like memory consolidation."""
        pass
