"""
ARC Memory System
Episodic and semantic memory stores with consolidation.
"""

import json
import time
from typing import Dict, List, Any, Optional, Tuple
from collections import deque, defaultdict
from datetime import datetime, timedelta
import numpy as np
from dataclasses import dataclass, asdict
import hashlib

from .config import MemoryConfig


@dataclass
class MemoryItem:
    """Individual memory item."""
    content: str
    context: Dict[str, Any]
    timestamp: float
    importance: float = 0.5
    access_count: int = 0
    last_accessed: float = None
    memory_type: str = "episodic"
    
    def __post_init__(self):
        if self.last_accessed is None:
            self.last_accessed = self.timestamp
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MemoryItem':
        return cls(**data)


class WorkingMemory:
    """Working memory with limited capacity."""
    
    def __init__(self, capacity: int = 7):
        self.capacity = capacity
        self.items = deque(maxlen=capacity)
        self.attention_weights = deque(maxlen=capacity)
    
    def add(self, item: MemoryItem, attention: float = 1.0):
        """Add item to working memory."""
        self.items.append(item)
        self.attention_weights.append(attention)
    
    def get_active_items(self, threshold: float = 0.3) -> List[MemoryItem]:
        """Get items above attention threshold."""
        active = []
        for item, weight in zip(self.items, self.attention_weights):
            if weight >= threshold:
                active.append(item)
        return active
    
    def clear(self):
        """Clear working memory."""
        self.items.clear()
        self.attention_weights.clear()
    
    def get_context_summary(self) -> str:
        """Get summary of current working memory context."""
        if not self.items:
            return "Empty working memory"
        
        contexts = [item.context.get('type', 'unknown') for item in self.items]
        content_summary = "; ".join([item.content[:50] + "..." if len(item.content) > 50 
                                   else item.content for item in list(self.items)[-3:]])
        
        return f"Recent context ({len(self.items)} items): {content_summary}"


class EpisodicMemory:
    """Episodic memory for storing experiences."""
    
    def __init__(self, capacity: int = 1000):
        self.capacity = capacity
        self.memories: List[MemoryItem] = []
        self.index_by_time = {}
        self.index_by_type = defaultdict(list)
    
    def store(self, item: MemoryItem):
        """Store episodic memory item."""
        # Remove oldest if at capacity
        if len(self.memories) >= self.capacity:
            removed = self.memories.pop(0)
            self._remove_from_indices(removed)
        
        # Add new memory
        self.memories.append(item)
        self._add_to_indices(item)
    
    def _add_to_indices(self, item: MemoryItem):
        """Add item to search indices."""
        time_key = int(item.timestamp // 3600)  # Hour buckets
        if time_key not in self.index_by_time:
            self.index_by_time[time_key] = []
        self.index_by_time[time_key].append(item)
        
        context_type = item.context.get('type', 'unknown')
        self.index_by_type[context_type].append(item)
    
    def _remove_from_indices(self, item: MemoryItem):
        """Remove item from search indices."""
        time_key = int(item.timestamp // 3600)
        if time_key in self.index_by_time:
            if item in self.index_by_time[time_key]:
                self.index_by_time[time_key].remove(item)
        
        context_type = item.context.get('type', 'unknown')
        if item in self.index_by_type[context_type]:
            self.index_by_type[context_type].remove(item)
    
    def retrieve_by_recency(self, limit: int = 10) -> List[MemoryItem]:
        """Retrieve most recent memories."""
        return self.memories[-limit:] if self.memories else []
    
    def retrieve_by_type(self, memory_type: str, limit: int = 10) -> List[MemoryItem]:
        """Retrieve memories by context type."""
        return self.index_by_type[memory_type][-limit:] if memory_type in self.index_by_type else []
    
    def retrieve_by_time_range(self, start_time: float, end_time: float) -> List[MemoryItem]:
        """Retrieve memories within time range."""
        results = []
        for memory in self.memories:
            if start_time <= memory.timestamp <= end_time:
                results.append(memory)
        return results
    
    def search_content(self, query: str, limit: int = 5) -> List[MemoryItem]:
        """Simple content-based search."""
        query_lower = query.lower()
        matches = []
        
        for memory in self.memories:
            if query_lower in memory.content.lower():
                matches.append((memory, memory.importance))
        
        # Sort by importance and recency
        matches.sort(key=lambda x: (x[1], x[0].timestamp), reverse=True)
        return [match[0] for match in matches[:limit]]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get memory statistics."""
        if not self.memories:
            return {"count": 0, "types": {}, "time_span": 0}
        
        type_counts = defaultdict(int)
        for memory in self.memories:
            context_type = memory.context.get('type', 'unknown')
            type_counts[context_type] += 1
        
        time_span = self.memories[-1].timestamp - self.memories[0].timestamp if len(self.memories) > 1 else 0
        
        return {
            "count": len(self.memories),
            "capacity": self.capacity,
            "types": dict(type_counts),
            "time_span_hours": time_span / 3600,
            "oldest": datetime.fromtimestamp(self.memories[0].timestamp).isoformat() if self.memories else None,
            "newest": datetime.fromtimestamp(self.memories[-1].timestamp).isoformat() if self.memories else None
        }


class SemanticMemory:
    """Semantic memory for storing learned patterns and concepts."""
    
    def __init__(self):
        self.concepts: Dict[str, Dict[str, Any]] = {}
        self.patterns: Dict[str, List[str]] = defaultdict(list)
        self.associations: Dict[str, List[str]] = defaultdict(list)
    
    def learn_concept(self, concept: str, properties: Dict[str, Any]):
        """Learn or update a concept."""
        if concept in self.concepts:
            # Update existing concept
            self.concepts[concept].update(properties)
        else:
            # New concept
            self.concepts[concept] = properties.copy()
        
        self.concepts[concept]['last_updated'] = time.time()
        self.concepts[concept]['access_count'] = self.concepts[concept].get('access_count', 0) + 1
    
    def learn_pattern(self, pattern_type: str, pattern: str):
        """Learn a new pattern."""
        if pattern not in self.patterns[pattern_type]:
            self.patterns[pattern_type].append(pattern)
    
    def add_association(self, concept_a: str, concept_b: str):
        """Add association between concepts."""
        if concept_b not in self.associations[concept_a]:
            self.associations[concept_a].append(concept_b)
        if concept_a not in self.associations[concept_b]:
            self.associations[concept_b].append(concept_a)
    
    def get_concept(self, concept: str) -> Optional[Dict[str, Any]]:
        """Get concept information."""
        if concept in self.concepts:
            self.concepts[concept]['access_count'] = self.concepts[concept].get('access_count', 0) + 1
            return self.concepts[concept]
        return None
    
    def get_associations(self, concept: str) -> List[str]:
        """Get associated concepts."""
        return self.associations.get(concept, [])
    
    def get_patterns(self, pattern_type: str) -> List[str]:
        """Get patterns of a specific type."""
        return self.patterns.get(pattern_type, [])
    
    def consolidate_from_episodic(self, episodic_memories: List[MemoryItem]):
        """Consolidate patterns from episodic memories."""
        # Simple pattern extraction
        for memory in episodic_memories:
            if memory.importance > 0.7:  # Only consolidate important memories
                # Extract simple patterns
                content_type = memory.context.get('type', 'general')
                self.learn_pattern(content_type, memory.content[:100])
                
                # Extract key concepts (simple word-based approach)
                words = memory.content.lower().split()
                important_words = [w for w in words if len(w) > 3 and w.isalpha()]
                
                for word in important_words[:3]:  # Top 3 words
                    self.learn_concept(word, {
                        'context': content_type,
                        'frequency': self.concepts.get(word, {}).get('frequency', 0) + 1
                    })
    
    def get_stats(self) -> Dict[str, Any]:
        """Get semantic memory statistics."""
        return {
            "concepts": len(self.concepts),
            "patterns": {ptype: len(patterns) for ptype, patterns in self.patterns.items()},
            "associations": len(self.associations),
            "top_concepts": sorted(self.concepts.items(), 
                                 key=lambda x: x[1].get('access_count', 0), 
                                 reverse=True)[:5]
        }


class MemorySystem:
    """Integrated memory system with working, episodic, and semantic memory."""
    
    def __init__(self, config: MemoryConfig):
        self.config = config
        
        # Initialize memory components
        self.working_memory = WorkingMemory(config.working_memory_size)
        self.episodic_memory = EpisodicMemory(config.episodic_memory_size)
        self.semantic_memory = SemanticMemory()
        
        # Consolidation tracking
        self.consolidation_counter = 0
        self.last_consolidation = time.time()
    
    def store_interaction(self, input_text: str, output_text: str, context: Dict[str, Any]):
        """Store a complete interaction."""
        timestamp = time.time()
        
        # Create memory items
        input_item = MemoryItem(
            content=input_text,
            context={**context, 'role': 'input'},
            timestamp=timestamp,
            importance=self._calculate_importance(input_text, context)
        )
        
        output_item = MemoryItem(
            content=output_text,
            context={**context, 'role': 'output'},
            timestamp=timestamp + 0.1,  # Slight offset
            importance=self._calculate_importance(output_text, context)
        )
        
        # Add to working memory
        self.working_memory.add(input_item, attention=1.0)
        self.working_memory.add(output_item, attention=0.8)
        
        # Add to episodic memory
        self.episodic_memory.store(input_item)
        self.episodic_memory.store(output_item)
        
        # Check for consolidation
        self.consolidation_counter += 1
        if self.consolidation_counter >= self.config.consolidation_interval:
            self._consolidate_memories()
    
    def _calculate_importance(self, content: str, context: Dict[str, Any]) -> float:
        """Calculate importance score for content."""
        base_importance = 0.5
        
        # Length factor
        length_factor = min(len(content) / 100, 1.0) * 0.2
        
        # Context type factor
        context_factors = {
            'error': 0.8,
            'learning': 0.9,
            'question': 0.7,
            'conversation': 0.5,
            'instruction': 0.8
        }
        
        context_type = context.get('type', 'conversation')
        context_factor = context_factors.get(context_type, 0.5)
        
        # Keyword factor (simple approach)
        important_keywords = ['learn', 'remember', 'important', 'error', 'problem', 'help']
        keyword_factor = sum(0.1 for kw in important_keywords if kw in content.lower())
        
        return min(base_importance + length_factor + context_factor + keyword_factor, 1.0)
    
    def _consolidate_memories(self):
        """Consolidate episodic memories into semantic memory."""
        # Get recent important episodic memories
        recent_memories = self.episodic_memory.retrieve_by_recency(50)
        important_memories = [m for m in recent_memories if m.importance > self.config.attention_threshold]
        
        # Consolidate into semantic memory
        self.semantic_memory.consolidate_from_episodic(important_memories)
        
        # Reset consolidation counter
        self.consolidation_counter = 0
        self.last_consolidation = time.time()
    
    def retrieve_context(self, query: str, max_items: int = 5) -> List[MemoryItem]:
        """Retrieve relevant context for a query."""
        results = []
        
        # Search episodic memory
        episodic_results = self.episodic_memory.search_content(query, max_items // 2)
        results.extend(episodic_results)
        
        # Add recent working memory items
        working_items = self.working_memory.get_active_items()
        results.extend(working_items[-2:])  # Last 2 active items
        
        # Sort by importance and recency
        results.sort(key=lambda x: (x.importance, x.timestamp), reverse=True)
        
        return results[:max_items]
    
    def get_conversation_context(self) -> str:
        """Get formatted conversation context."""
        return self.working_memory.get_context_summary()
    
    def get_system_stats(self) -> Dict[str, Any]:
        """Get comprehensive memory system statistics."""
        return {
            "working_memory": {
                "active_items": len(self.working_memory.items),
                "capacity": self.working_memory.capacity
            },
            "episodic_memory": self.episodic_memory.get_stats(),
            "semantic_memory": self.semantic_memory.get_stats(),
            "consolidation": {
                "counter": self.consolidation_counter,
                "interval": self.config.consolidation_interval,
                "last_consolidation": datetime.fromtimestamp(self.last_consolidation).isoformat(),
                "next_consolidation_in": self.config.consolidation_interval - self.consolidation_counter
            }
        }
    
    def save_memories(self, filepath: str):
        """Save memories to file."""
        data = {
            "episodic": [m.to_dict() for m in self.episodic_memory.memories],
            "semantic": {
                "concepts": self.semantic_memory.concepts,
                "patterns": dict(self.semantic_memory.patterns),
                "associations": dict(self.semantic_memory.associations)
            },
            "metadata": {
                "saved_at": time.time(),
                "consolidation_counter": self.consolidation_counter
            }
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
    
    def load_memories(self, filepath: str):
        """Load memories from file."""
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            # Load episodic memories
            for item_data in data.get("episodic", []):
                item = MemoryItem.from_dict(item_data)
                self.episodic_memory.store(item)
            
            # Load semantic memories
            semantic_data = data.get("semantic", {})
            self.semantic_memory.concepts = semantic_data.get("concepts", {})
            self.semantic_memory.patterns = defaultdict(list, semantic_data.get("patterns", {}))
            self.semantic_memory.associations = defaultdict(list, semantic_data.get("associations", {}))
            
            # Load metadata
            metadata = data.get("metadata", {})
            self.consolidation_counter = metadata.get("consolidation_counter", 0)
            
        except Exception as e:
            print(f"Warning: Could not load memories from {filepath}: {e}")
