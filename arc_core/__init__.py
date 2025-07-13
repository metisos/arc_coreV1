"""
ARC Core - Adaptive Recursive Consciousness Engine

A modular continual learning framework for language models with biological learning mechanisms.

Public API for metisos-arc-core package.
Installation: pip install metisos-arc-core

For complete documentation visit: https://github.com/metisos/arc_coreV1
"""

__version__ = "0.1.0"

# Public API exports
from .config import ARCConfig, create_default_config
from .train import ARCTrainer
from .memory import MemorySystem
from .safety import SafetySystem

__all__ = [
    "ARCConfig",
    "create_default_config", 
    "ARCTrainer",
    "MemorySystem",
    "SafetySystem"
]
