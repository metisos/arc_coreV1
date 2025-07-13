"""Tests for ARC configuration system."""

import pytest
import tempfile
import os
from pathlib import Path

from arc_core.config import ARCConfig, LoRAConfig, TrainingConfig, MemoryConfig, SafetyConfig


class TestARCConfig:
    """Test ARC configuration functionality."""

    def test_default_config(self):
        """Test default configuration creation."""
        config = ARCConfig()
        
        assert config.base_model == ""
        assert config.model_name == "arc-model"
        assert config.context_length == 1024
        assert config.device == "auto"
        
        # Test nested configs
        assert isinstance(config.lora, LoRAConfig)
        assert isinstance(config.training, TrainingConfig)
        assert isinstance(config.memory, MemoryConfig)
        assert isinstance(config.safety, SafetyConfig)

    def test_config_serialization(self):
        """Test config save/load functionality."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            config_path = os.path.join(tmp_dir, "test_config.json")
            
            # Create and modify config
            config = ARCConfig()
            config.base_model = "gpt2"
            config.model_name = "test-model"
            config.lora.r = 32
            config.training.learning_rate = 1e-4
            
            # Save config
            config.config_path = config_path
            config.save()
            
            # Load config
            loaded_config = ARCConfig.load(config_path)
            
            assert loaded_config.base_model == "gpt2"
            assert loaded_config.model_name == "test-model"
            assert loaded_config.lora.r == 32
            assert loaded_config.training.learning_rate == 1e-4

    def test_lora_config(self):
        """Test LoRA configuration."""
        lora_config = LoRAConfig()
        
        assert lora_config.r == 16
        assert lora_config.alpha == 32
        assert lora_config.dropout == 0.1
        assert "q_proj" in lora_config.target_modules

    def test_training_config(self):
        """Test training configuration."""
        training_config = TrainingConfig()
        
        assert training_config.learning_rate == 5e-4
        assert training_config.max_steps == 100
        assert training_config.ewc_lambda == 0.4
        assert training_config.batch_size == 1

    def test_memory_config(self):
        """Test memory configuration."""
        memory_config = MemoryConfig()
        
        assert memory_config.working_memory_size == 10
        assert memory_config.episodic_memory_size == 1000
        assert memory_config.semantic_memory_size == 10000
        assert memory_config.consolidation_threshold == 0.7

    def test_safety_config(self):
        """Test safety configuration."""
        safety_config = SafetyConfig()
        
        assert safety_config.enable_cognitive_inhibition is True
        assert safety_config.enable_contextual_gating is True
        assert safety_config.enable_metacognitive_monitoring is True
        assert safety_config.inhibition_threshold == 0.8

    def test_config_paths(self):
        """Test configuration path methods."""
        config = ARCConfig()
        config.model_name = "test-model"
        
        # Test path generation
        model_path = config.get_model_save_path()
        assert "test-model" in str(model_path)
        
        history_path = config.get_history_path()
        assert "training_history.jsonl" in str(history_path)

    def test_config_validation(self):
        """Test configuration validation."""
        config = ARCConfig()
        
        # Valid config should not raise
        config.context_length = 1024
        config.lora.r = 16
        
        # Invalid values should be caught
        with pytest.raises(ValueError):
            config.context_length = -1
        
        with pytest.raises(ValueError):
            config.lora.r = 0
