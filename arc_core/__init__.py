"""
ARC Core - Adaptive Recursive Consciousness Engine
A modular continual learning system for language models.
"""

__version__ = "0.9.0"
__author__ = "Metis AI"
__description__ = "Adaptive Recursive Consciousness Engine"

from .train import ARCTrainer
from .memory import MemorySystem
from .safety import SafetySystem
from .config import ARCConfig

__all__ = [
    "ARCTrainer",
    "MemorySystem", 
    "SafetySystem",
    "ARCConfig",
    "__version__"
]
