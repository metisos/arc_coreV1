"""
ARC Configuration Management
Handles YAML-based configuration for ARC models and training.
"""

import os
import json
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict


@dataclass
class LoRAConfig:
    """LoRA adapter configuration."""
    r: int = 16
    alpha: int = 32
    dropout: float = 0.1
    target_modules: list = None
    
    def __post_init__(self):
        if self.target_modules is None:
            # Default target modules for common architectures
            self.target_modules = ["q_proj", "v_proj", "k_proj", "o_proj"]


@dataclass
class TrainingConfig:
    """Training configuration."""
    learning_rate: float = 1e-4
    batch_size: int = 4
    gradient_accumulation_steps: int = 1
    max_steps: int = 1000
    warmup_steps: int = 100
    weight_decay: float = 0.01
    ewc_lambda: float = 0.4  # Elastic Weight Consolidation


@dataclass
class MemoryConfig:
    """Memory system configuration."""
    working_memory_size: int = 7
    episodic_memory_size: int = 1000
    consolidation_interval: int = 50
    attention_threshold: float = 0.7


@dataclass
class SafetyConfig:
    """Safety system configuration."""
    enable_cognitive_inhibition: bool = True
    enable_contextual_gating: bool = True
    enable_metacognitive_monitoring: bool = True
    response_quality_threshold: float = 0.7


class ARCConfig:
    """Main ARC configuration manager."""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path or self._get_default_config_path()
        self.base_model: str = ""
        self.model_name: str = ""
        self.context_length: int = 2048
        self.device: str = "auto"
        
        # Component configurations
        self.lora = LoRAConfig()
        self.training = TrainingConfig()
        self.memory = MemoryConfig()
        self.safety = SafetyConfig()
        
        # Load existing config if available
        if os.path.exists(self.config_path):
            self.load()
    
    def _get_default_config_path(self) -> str:
        """Get default configuration path."""
        arc_dir = Path.home() / ".arc"
        arc_dir.mkdir(exist_ok=True)
        return str(arc_dir / "arc_config.json")
    
    def load(self) -> None:
        """Load configuration from file."""
        try:
            with open(self.config_path, 'r') as f:
                data = json.load(f)
            
            # Load basic settings
            self.base_model = data.get("base_model", "")
            self.model_name = data.get("model_name", "")
            self.context_length = data.get("context_length", 2048)
            self.device = data.get("device", "auto")
            
            # Load component configs
            if "lora" in data:
                self.lora = LoRAConfig(**data["lora"])
            if "training" in data:
                self.training = TrainingConfig(**data["training"])
            if "memory" in data:
                self.memory = MemoryConfig(**data["memory"])
            if "safety" in data:
                self.safety = SafetyConfig(**data["safety"])
                
        except Exception as e:
            print(f"Warning: Could not load config from {self.config_path}: {e}")
    
    def save(self) -> None:
        """Save configuration to file."""
        config_data = {
            "base_model": self.base_model,
            "model_name": self.model_name,
            "context_length": self.context_length,
            "device": self.device,
            "lora": asdict(self.lora),
            "training": asdict(self.training),
            "memory": asdict(self.memory),
            "safety": asdict(self.safety)
        }
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
        
        with open(self.config_path, 'w') as f:
            json.dump(config_data, f, indent=2)
    
    def update_model_info(self, base_model: str, model_name: str, context_length: int):
        """Update model information."""
        self.base_model = base_model
        self.model_name = model_name
        self.context_length = context_length
        self.save()
    
    def get_model_save_path(self) -> str:
        """Get path for saving model artifacts."""
        arc_dir = Path(self.config_path).parent
        models_dir = arc_dir / "models"
        models_dir.mkdir(exist_ok=True)
        return str(models_dir)
    
    def get_history_path(self) -> str:
        """Get path for training history log."""
        arc_dir = Path(self.config_path).parent
        return str(arc_dir / "history.log")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            "base_model": self.base_model,
            "model_name": self.model_name,
            "context_length": self.context_length,
            "device": self.device,
            "lora": asdict(self.lora),
            "training": asdict(self.training),
            "memory": asdict(self.memory),
            "safety": asdict(self.safety)
        }
    
    def __str__(self) -> str:
        """String representation of configuration."""
        return f"""ARC Configuration:
  Base Model: {self.base_model}
  Model Name: {self.model_name}
  Context Length: {self.context_length}
  Device: {self.device}
  LoRA Rank: {self.lora.r}
  Learning Rate: {self.training.learning_rate}
  Memory Size: W:{self.memory.working_memory_size} E:{self.memory.episodic_memory_size}
  Safety: {'Enabled' if self.safety.enable_cognitive_inhibition else 'Disabled'}"""


def load_teaching_pack_config(pack_path: str) -> Dict[str, Any]:
    """Load teaching pack configuration from pack.yml."""
    pack_yml = Path(pack_path) / "pack.yml"
    
    if not pack_yml.exists():
        raise FileNotFoundError(f"Teaching pack config not found: {pack_yml}")
    
    with open(pack_yml, 'r') as f:
        return yaml.safe_load(f)


def create_default_config() -> ARCConfig:
    """Create a default ARC configuration."""
    config = ARCConfig()
    config.base_model = "microsoft/DialoGPT-medium"
    config.model_name = "arc-dialogpt"
    config.context_length = 1024
    return config
