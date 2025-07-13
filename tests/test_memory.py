"""Tests for ARC memory systems."""

import pytest
from datetime import datetime, timedelta

from arc_core.memory import (
    MemoryItem, WorkingMemory, EpisodicMemory, 
    SemanticMemory, MemorySystem
)
from arc_core.config import MemoryConfig


class TestMemoryItem:
    """Test memory item functionality."""

    def test_memory_item_creation(self):
        """Test memory item creation and properties."""
        content = "Test memory content"
        item = MemoryItem(content=content, memory_type="episodic")
        
        assert item.content == content
        assert item.memory_type == "episodic"
        assert item.strength == 1.0
        assert isinstance(item.timestamp, datetime)
        assert item.context == {}

    def test_memory_item_decay(self):
        """Test memory strength decay."""
        item = MemoryItem(content="test", memory_type="working")
        original_strength = item.strength
        
        # Simulate decay
        item.decay(0.1)
        assert item.strength < original_strength
        assert item.strength >= 0.0

    def test_memory_item_reinforcement(self):
        """Test memory reinforcement."""
        item = MemoryItem(content="test", memory_type="episodic")
        item.strength = 0.5
        
        item.reinforce(0.3)
        assert item.strength == 0.8
        
        # Should cap at 1.0
        item.reinforce(0.5)
        assert item.strength == 1.0


class TestWorkingMemory:
    """Test working memory functionality."""

    def test_working_memory_creation(self):
        """Test working memory initialization."""
        config = MemoryConfig()
        working_memory = WorkingMemory(config)
        
        assert working_memory.capacity == config.working_memory_size
        assert len(working_memory.items) == 0

    def test_working_memory_add(self):
        """Test adding items to working memory."""
        config = MemoryConfig()
        config.working_memory_size = 3
        working_memory = WorkingMemory(config)
        
        # Add items
        working_memory.add("Item 1")
        working_memory.add("Item 2")
        working_memory.add("Item 3")
        
        assert len(working_memory.items) == 3
        
        # Adding beyond capacity should remove oldest
        working_memory.add("Item 4")
        assert len(working_memory.items) == 3
        assert "Item 1" not in [item.content for item in working_memory.items]
        assert "Item 4" in [item.content for item in working_memory.items]

    def test_working_memory_decay(self):
        """Test working memory decay."""
        config = MemoryConfig()
        working_memory = WorkingMemory(config)
        
        working_memory.add("Test item")
        original_strength = working_memory.items[0].strength
        
        working_memory.decay_step()
        assert working_memory.items[0].strength < original_strength


class TestEpisodicMemory:
    """Test episodic memory functionality."""

    def test_episodic_memory_creation(self):
        """Test episodic memory initialization."""
        config = MemoryConfig()
        episodic_memory = EpisodicMemory(config)
        
        assert episodic_memory.capacity == config.episodic_memory_size
        assert len(episodic_memory.episodes) == 0

    def test_episodic_memory_add(self):
        """Test adding episodes."""
        config = MemoryConfig()
        episodic_memory = EpisodicMemory(config)
        
        interaction = {
            "input": "Hello",
            "output": "Hi there!",
            "context": {"mood": "friendly"}
        }
        
        episodic_memory.add_interaction(interaction)
        assert len(episodic_memory.episodes) == 1
        
        episode = episodic_memory.episodes[0]
        assert episode.content["input"] == "Hello"
        assert episode.content["output"] == "Hi there!"

    def test_episodic_memory_search(self):
        """Test episodic memory search."""
        config = MemoryConfig()
        episodic_memory = EpisodicMemory(config)
        
        # Add test episodes
        episodic_memory.add_interaction({
            "input": "What is Python?",
            "output": "Python is a programming language"
        })
        episodic_memory.add_interaction({
            "input": "Hello there",
            "output": "Hello! How can I help?"
        })
        
        # Search for programming-related content
        results = episodic_memory.search("Python programming", top_k=1)
        assert len(results) == 1
        assert "Python" in results[0].content["input"]


class TestSemanticMemory:
    """Test semantic memory functionality."""

    def test_semantic_memory_creation(self):
        """Test semantic memory initialization."""
        config = MemoryConfig()
        semantic_memory = SemanticMemory(config)
        
        assert semantic_memory.capacity == config.semantic_memory_size
        assert len(semantic_memory.concepts) == 0

    def test_semantic_memory_learn(self):
        """Test concept learning."""
        config = MemoryConfig()
        semantic_memory = SemanticMemory(config)
        
        # Learn concept
        concept = "machine learning"
        description = "A type of artificial intelligence"
        
        semantic_memory.learn_concept(concept, description)
        assert len(semantic_memory.concepts) == 1
        assert concept in semantic_memory.concepts
        
        # Learning same concept should reinforce
        semantic_memory.learn_concept(concept, description)
        assert len(semantic_memory.concepts) == 1  # Still just one concept
        assert semantic_memory.concepts[concept].strength > 1.0  # But stronger

    def test_semantic_memory_retrieve(self):
        """Test concept retrieval."""
        config = MemoryConfig()
        semantic_memory = SemanticMemory(config)
        
        # Add concepts
        semantic_memory.learn_concept("AI", "Artificial Intelligence")
        semantic_memory.learn_concept("ML", "Machine Learning")
        
        # Retrieve concept
        concept = semantic_memory.retrieve_concept("AI")
        assert concept is not None
        assert concept.content == "Artificial Intelligence"
        
        # Non-existent concept
        concept = semantic_memory.retrieve_concept("XYZ")
        assert concept is None


class TestMemorySystem:
    """Test integrated memory system."""

    def test_memory_system_creation(self):
        """Test memory system initialization."""
        config = MemoryConfig()
        memory_system = MemorySystem(config)
        
        assert memory_system.working_memory is not None
        assert memory_system.episodic_memory is not None
        assert memory_system.semantic_memory is not None

    def test_memory_system_store_interaction(self):
        """Test storing complete interactions."""
        config = MemoryConfig()
        memory_system = MemorySystem(config)
        
        interaction = {
            "input": "Explain neural networks",
            "output": "Neural networks are computing systems inspired by biological neural networks",
            "context": {"domain": "AI"}
        }
        
        memory_system.store_interaction(interaction)
        
        # Should be in working memory
        assert len(memory_system.working_memory.items) == 1
        
        # Should be in episodic memory
        assert len(memory_system.episodic_memory.episodes) == 1

    def test_memory_system_consolidation(self):
        """Test memory consolidation process."""
        config = MemoryConfig()
        config.consolidation_threshold = 0.5
        memory_system = MemorySystem(config)
        
        # Add strong episodic memory
        interaction = {
            "input": "What is deep learning?",
            "output": "Deep learning is a subset of machine learning",
            "context": {"importance": "high"}
        }
        
        memory_system.store_interaction(interaction)
        
        # Manually strengthen the memory
        episode = memory_system.episodic_memory.episodes[0]
        episode.reinforce(0.6)  # Make it strong enough for consolidation
        
        # Run consolidation
        memory_system.consolidate_memories()
        
        # Should have extracted concepts
        assert len(memory_system.semantic_memory.concepts) > 0

    def test_memory_system_stats(self):
        """Test memory system statistics."""
        config = MemoryConfig()
        memory_system = MemorySystem(config)
        
        # Add some memories
        memory_system.store_interaction({
            "input": "Test 1",
            "output": "Response 1"
        })
        memory_system.store_interaction({
            "input": "Test 2", 
            "output": "Response 2"
        })
        
        stats = memory_system.get_system_stats()
        
        assert "working_memory" in stats
        assert "episodic_memory" in stats
        assert "semantic_memory" in stats
        
        assert stats["working_memory"]["active_items"] == 2
        assert stats["episodic_memory"]["count"] == 2
