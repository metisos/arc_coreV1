"""
ARC Safety System
Gating and inhibition modules for safe AI behavior.
"""

import re
import json
import time
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass
from datetime import datetime, timedelta
import hashlib

from .config import SafetyConfig


@dataclass
class SafetyViolation:
    """Safety violation record."""
    violation_type: str
    content: str
    severity: float
    timestamp: float
    context: Dict[str, Any]
    action_taken: str


class CognitiveInhibition:
    """Cognitive inhibition system to filter inappropriate responses."""
    
    def __init__(self):
        # Define harmful content patterns
        self.harmful_patterns = [
            r'\b(kill|murder|suicide|harm|hurt|violence)\b',
            r'\b(illegal|criminal|fraud|scam)\b',
            r'\b(racist|sexist|discriminat|hate)\b',
            r'\b(private information|personal data|password|credit card)\b'
        ]
        
        # Define low-quality response patterns
        self.quality_patterns = [
            r'^(I don\'t know|I can\'t|Sorry, I cannot)',
            r'^(As an AI|I am an AI|I\'m an AI)',
            r'^\s*$',  # Empty responses
            r'^(.{1,10})$'  # Very short responses
        ]
        
        # Positive indicators
        self.positive_patterns = [
            r'\b(help|assist|support|guide|explain)\b',
            r'\b(learn|understand|knowledge|information)\b',
            r'\b(please|thank|welcome|glad)\b'
        ]
        
        self.violation_history: List[SafetyViolation] = []
    
    def inhibit_inappropriate_response(self, response: str, context: Dict[str, Any]) -> str:
        """Apply cognitive inhibition to filter inappropriate responses."""
        
        # Check for harmful content
        harm_score = self._assess_harm_risk(response)
        if harm_score > 0.7:
            violation = SafetyViolation(
                violation_type="harmful_content",
                content=response[:100],
                severity=harm_score,
                timestamp=time.time(),
                context=context,
                action_taken="response_blocked"
            )
            self.violation_history.append(violation)
            return self._generate_safe_alternative(context)
        
        # Check response quality
        quality_score = self._assess_response_quality(response)
        if quality_score < 0.3:
            # Don't block but note low quality
            violation = SafetyViolation(
                violation_type="low_quality",
                content=response[:100],
                severity=1.0 - quality_score,
                timestamp=time.time(),
                context=context,
                action_taken="quality_warning"
            )
            self.violation_history.append(violation)
        
        return response
    
    def _assess_harm_risk(self, text: str) -> float:
        """Assess potential harm risk in text."""
        text_lower = text.lower()
        risk_score = 0.0
        
        # Check harmful patterns
        for pattern in self.harmful_patterns:
            matches = len(re.findall(pattern, text_lower))
            risk_score += matches * 0.3
        
        # Check for explicit instructions to do harmful things
        if re.search(r'\b(how to|instructions|steps).+(harm|illegal|dangerous)', text_lower):
            risk_score += 0.5
        
        # Check for personal information requests
        if re.search(r'\b(give me|tell me|what is).+(password|ssn|credit card|personal)', text_lower):
            risk_score += 0.4
        
        return min(risk_score, 1.0)
    
    def _assess_response_quality(self, text: str) -> float:
        """Assess response quality."""
        if not text or not text.strip():
            return 0.0
        
        quality_score = 0.5  # Base score
        
        # Length factor
        if len(text) > 20:
            quality_score += 0.2
        if len(text) > 50:
            quality_score += 0.1
        
        # Check for low-quality patterns (negative)
        for pattern in self.quality_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                quality_score -= 0.3
        
        # Check for positive indicators
        for pattern in self.positive_patterns:
            matches = len(re.findall(pattern, text.lower()))
            quality_score += matches * 0.1
        
        return max(0.0, min(quality_score, 1.0))
    
    def _generate_safe_alternative(self, context: Dict[str, Any]) -> str:
        """Generate a safe alternative response."""
        alternatives = [
            "I understand you're looking for information, but I can't provide content that might be harmful. Is there another way I can help?",
            "I'd be happy to help with something constructive instead. What other questions do you have?",
            "Let me help you with something positive and useful. What would you like to learn about?",
            "I'm designed to be helpful and safe. Is there a different topic I can assist you with?"
        ]
        
        # Simple selection based on context
        context_type = context.get('type', 'general')
        if context_type == 'question':
            return alternatives[0]
        elif context_type == 'conversation':
            return alternatives[1]
        else:
            return alternatives[2]
    
    def get_violation_stats(self) -> Dict[str, Any]:
        """Get safety violation statistics."""
        if not self.violation_history:
            return {"total_violations": 0, "types": {}, "recent_violations": 0}
        
        # Count by type
        type_counts = {}
        recent_count = 0
        recent_threshold = time.time() - 3600  # Last hour
        
        for violation in self.violation_history:
            type_counts[violation.violation_type] = type_counts.get(violation.violation_type, 0) + 1
            if violation.timestamp > recent_threshold:
                recent_count += 1
        
        return {
            "total_violations": len(self.violation_history),
            "types": type_counts,
            "recent_violations": recent_count,
            "last_violation": self.violation_history[-1].timestamp if self.violation_history else None
        }


class ContextualGating:
    """Contextual gating system for memory encoding and retrieval."""
    
    def __init__(self):
        self.encoding_gates = {
            'importance_gate': 0.5,
            'novelty_gate': 0.3,
            'relevance_gate': 0.7,
            'emotional_gate': 0.4
        }
        
        self.retrieval_gates = {
            'recency_gate': 0.6,
            'frequency_gate': 0.4,
            'similarity_gate': 0.8
        }
        
        self.context_history: List[Dict[str, Any]] = []
    
    def should_encode_memory(self, content: str, context: Dict[str, Any]) -> Tuple[bool, float]:
        """Determine if content should be encoded to memory."""
        
        # Calculate gating scores
        importance_score = self._assess_importance(content, context)
        novelty_score = self._assess_novelty(content, context)
        relevance_score = self._assess_relevance(content, context)
        emotional_score = self._assess_emotional_content(content)
        
        # Apply gates
        passes_importance = importance_score >= self.encoding_gates['importance_gate']
        passes_novelty = novelty_score >= self.encoding_gates['novelty_gate']
        passes_relevance = relevance_score >= self.encoding_gates['relevance_gate']
        passes_emotional = emotional_score >= self.encoding_gates['emotional_gate']
        
        # At least 2 out of 4 gates must pass
        gate_passes = sum([passes_importance, passes_novelty, passes_relevance, passes_emotional])
        should_encode = gate_passes >= 2
        
        # Overall encoding strength
        encoding_strength = (importance_score + novelty_score + relevance_score + emotional_score) / 4
        
        # Store context for future novelty assessment
        context_summary = {
            'content_hash': hashlib.md5(content.encode()).hexdigest(),
            'context_type': context.get('type', 'unknown'),
            'timestamp': time.time(),
            'encoding_strength': encoding_strength
        }
        self.context_history.append(context_summary)
        
        # Keep only recent context history
        if len(self.context_history) > 1000:
            self.context_history = self.context_history[-500:]
        
        return should_encode, encoding_strength
    
    def _assess_importance(self, content: str, context: Dict[str, Any]) -> float:
        """Assess content importance."""
        importance_indicators = [
            'important', 'critical', 'urgent', 'remember', 'key', 'essential',
            'warning', 'error', 'problem', 'issue', 'learn', 'understand'
        ]
        
        content_lower = content.lower()
        importance_score = 0.3  # Base score
        
        for indicator in importance_indicators:
            if indicator in content_lower:
                importance_score += 0.1
        
        # Context type importance
        context_importance = {
            'error': 0.8,
            'learning': 0.9,
            'instruction': 0.7,
            'question': 0.6,
            'conversation': 0.4
        }
        
        context_type = context.get('type', 'conversation')
        importance_score += context_importance.get(context_type, 0.4)
        
        return min(importance_score, 1.0)
    
    def _assess_novelty(self, content: str, context: Dict[str, Any]) -> float:
        """Assess content novelty."""
        content_hash = hashlib.md5(content.encode()).hexdigest()
        
        # Check if similar content was seen recently
        recent_threshold = time.time() - 3600  # Last hour
        similar_count = 0
        
        for ctx in self.context_history:
            if ctx['timestamp'] > recent_threshold:
                # Simple similarity check using hash comparison
                if ctx['content_hash'] == content_hash:
                    similar_count += 1
        
        # Higher novelty if content hasn't been seen recently
        novelty_score = max(0.1, 1.0 - (similar_count * 0.3))
        
        return novelty_score
    
    def _assess_relevance(self, content: str, context: Dict[str, Any]) -> float:
        """Assess content relevance to current context."""
        # Simple relevance based on context continuity
        relevance_score = 0.5  # Base relevance
        
        # If this is part of an ongoing conversation
        if context.get('conversation_id'):
            relevance_score += 0.3
        
        # If it's a direct response to a question
        if context.get('type') == 'response' and context.get('in_reply_to'):
            relevance_score += 0.4
        
        return min(relevance_score, 1.0)
    
    def _assess_emotional_content(self, content: str) -> float:
        """Assess emotional content strength."""
        emotional_words = [
            'happy', 'sad', 'angry', 'excited', 'worried', 'surprised',
            'love', 'hate', 'fear', 'joy', 'disappointment', 'frustration',
            'amazing', 'terrible', 'wonderful', 'awful', 'fantastic', 'horrible'
        ]
        
        content_lower = content.lower()
        emotional_score = 0.2  # Base score
        
        for word in emotional_words:
            if word in content_lower:
                emotional_score += 0.1
        
        # Check for emotional punctuation
        if '!' in content or '?' in content:
            emotional_score += 0.1
        
        return min(emotional_score, 1.0)
    
    def should_retrieve_memory(self, query: str, memory_item: Dict[str, Any]) -> Tuple[bool, float]:
        """Determine if a memory item should be retrieved for a query."""
        
        recency_score = self._assess_recency(memory_item)
        frequency_score = self._assess_frequency(memory_item)
        similarity_score = self._assess_similarity(query, memory_item)
        
        # Apply retrieval gates
        passes_recency = recency_score >= self.retrieval_gates['recency_gate']
        passes_frequency = frequency_score >= self.retrieval_gates['frequency_gate']
        passes_similarity = similarity_score >= self.retrieval_gates['similarity_gate']
        
        # At least 1 gate must pass strongly, or 2 gates moderately
        strong_passes = sum([
            recency_score > 0.8,
            frequency_score > 0.8,
            similarity_score > 0.8
        ])
        
        moderate_passes = sum([passes_recency, passes_frequency, passes_similarity])
        
        should_retrieve = strong_passes >= 1 or moderate_passes >= 2
        
        # Overall retrieval strength
        retrieval_strength = (recency_score + frequency_score + similarity_score) / 3
        
        return should_retrieve, retrieval_strength
    
    def _assess_recency(self, memory_item: Dict[str, Any]) -> float:
        """Assess memory item recency."""
        timestamp = memory_item.get('timestamp', 0)
        time_diff = time.time() - timestamp
        
        # Exponential decay over time
        if time_diff < 3600:  # Less than 1 hour
            return 1.0
        elif time_diff < 86400:  # Less than 1 day
            return 0.8
        elif time_diff < 604800:  # Less than 1 week
            return 0.5
        else:
            return 0.2
    
    def _assess_frequency(self, memory_item: Dict[str, Any]) -> float:
        """Assess memory item access frequency."""
        access_count = memory_item.get('access_count', 0)
        
        if access_count == 0:
            return 0.3
        elif access_count < 3:
            return 0.5
        elif access_count < 10:
            return 0.7
        else:
            return 1.0
    
    def _assess_similarity(self, query: str, memory_item: Dict[str, Any]) -> float:
        """Assess similarity between query and memory item."""
        content = memory_item.get('content', '')
        
        # Simple word overlap similarity
        query_words = set(query.lower().split())
        content_words = set(content.lower().split())
        
        if not query_words or not content_words:
            return 0.0
        
        overlap = len(query_words.intersection(content_words))
        total_unique = len(query_words.union(content_words))
        
        similarity = overlap / total_unique if total_unique > 0 else 0.0
        return similarity


class MetacognitiveMonitoring:
    """Metacognitive monitoring system for self-correction and quality control."""
    
    def __init__(self):
        self.confidence_threshold = 0.7
        self.consistency_threshold = 0.8
        self.monitoring_history: List[Dict[str, Any]] = []
    
    def monitor_response_quality(self, response: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Monitor and assess response quality."""
        
        # Assess various quality dimensions
        confidence_score = self._assess_confidence(response, context)
        consistency_score = self._assess_consistency(response, context)
        completeness_score = self._assess_completeness(response, context)
        clarity_score = self._assess_clarity(response)
        
        # Overall quality score
        quality_score = (confidence_score + consistency_score + completeness_score + clarity_score) / 4
        
        # Generate recommendations
        recommendations = self._generate_recommendations(
            confidence_score, consistency_score, completeness_score, clarity_score
        )
        
        # Store monitoring result
        monitoring_result = {
            'timestamp': time.time(),
            'response_length': len(response),
            'context_type': context.get('type', 'unknown'),
            'quality_scores': {
                'confidence': confidence_score,
                'consistency': consistency_score,
                'completeness': completeness_score,
                'clarity': clarity_score,
                'overall': quality_score
            },
            'recommendations': recommendations,
            'needs_revision': quality_score < self.confidence_threshold
        }
        
        self.monitoring_history.append(monitoring_result)
        
        # Keep history manageable
        if len(self.monitoring_history) > 500:
            self.monitoring_history = self.monitoring_history[-250:]
        
        return monitoring_result
    
    def _assess_confidence(self, response: str, context: Dict[str, Any]) -> float:
        """Assess confidence in response."""
        # Indicators of low confidence
        low_confidence_phrases = [
            "i'm not sure", "i think", "maybe", "possibly", "perhaps",
            "i believe", "it seems", "it might be", "i guess"
        ]
        
        # Indicators of high confidence
        high_confidence_phrases = [
            "definitely", "certainly", "absolutely", "clearly", "obviously",
            "without doubt", "for sure", "indeed"
        ]
        
        response_lower = response.lower()
        confidence_score = 0.5  # Base confidence
        
        # Check for confidence indicators
        for phrase in low_confidence_phrases:
            if phrase in response_lower:
                confidence_score -= 0.1
        
        for phrase in high_confidence_phrases:
            if phrase in response_lower:
                confidence_score += 0.1
        
        # Response length factor (very short responses often indicate uncertainty)
        if len(response) < 20:
            confidence_score -= 0.2
        elif len(response) > 100:
            confidence_score += 0.1
        
        return max(0.0, min(confidence_score, 1.0))
    
    def _assess_consistency(self, response: str, context: Dict[str, Any]) -> float:
        """Assess response consistency with context and history."""
        # Simple consistency check based on context type
        context_type = context.get('type', 'conversation')
        
        consistency_score = 0.7  # Base consistency
        
        # Check if response matches expected context type
        if context_type == 'question' and '?' not in response and not any(
            word in response.lower() for word in ['answer', 'is', 'are', 'yes', 'no']
        ):
            consistency_score -= 0.2
        
        if context_type == 'instruction' and not any(
            word in response.lower() for word in ['step', 'first', 'then', 'next', 'follow']
        ):
            consistency_score -= 0.1
        
        return max(0.0, min(consistency_score, 1.0))
    
    def _assess_completeness(self, response: str, context: Dict[str, Any]) -> float:
        """Assess response completeness."""
        completeness_score = 0.5  # Base score
        
        # Length factor
        if len(response) > 50:
            completeness_score += 0.2
        if len(response) > 150:
            completeness_score += 0.1
        
        # Check for complete sentence structures
        if response.endswith('.') or response.endswith('!') or response.endswith('?'):
            completeness_score += 0.1
        
        # Check for explanation or reasoning
        if any(word in response.lower() for word in ['because', 'since', 'therefore', 'so', 'thus']):
            completeness_score += 0.1
        
        return max(0.0, min(completeness_score, 1.0))
    
    def _assess_clarity(self, response: str) -> float:
        """Assess response clarity."""
        clarity_score = 0.5  # Base score
        
        # Sentence structure
        sentences = response.split('.')
        avg_sentence_length = sum(len(s.split()) for s in sentences) / len(sentences) if sentences else 0
        
        # Optimal sentence length is around 15-20 words
        if 10 <= avg_sentence_length <= 25:
            clarity_score += 0.2
        elif avg_sentence_length > 40:  # Very long sentences reduce clarity
            clarity_score -= 0.2
        
        # Check for clear structure words
        structure_words = ['first', 'second', 'then', 'next', 'finally', 'however', 'therefore']
        if any(word in response.lower() for word in structure_words):
            clarity_score += 0.1
        
        # Avoid excessive repetition
        words = response.lower().split()
        if len(set(words)) / len(words) < 0.7:  # High repetition
            clarity_score -= 0.1
        
        return max(0.0, min(clarity_score, 1.0))
    
    def _generate_recommendations(self, confidence: float, consistency: float, 
                                completeness: float, clarity: float) -> List[str]:
        """Generate improvement recommendations."""
        recommendations = []
        
        if confidence < 0.6:
            recommendations.append("Consider providing more definitive information or acknowledging uncertainty explicitly")
        
        if consistency < 0.6:
            recommendations.append("Ensure response matches the context and question type")
        
        if completeness < 0.6:
            recommendations.append("Provide more detailed explanation or examples")
        
        if clarity < 0.6:
            recommendations.append("Use clearer sentence structure and avoid excessive complexity")
        
        return recommendations
    
    def get_monitoring_stats(self) -> Dict[str, Any]:
        """Get metacognitive monitoring statistics."""
        if not self.monitoring_history:
            return {"total_monitored": 0, "avg_quality": 0, "needs_revision_rate": 0}
        
        total_quality = sum(item['quality_scores']['overall'] for item in self.monitoring_history)
        avg_quality = total_quality / len(self.monitoring_history)
        
        needs_revision_count = sum(1 for item in self.monitoring_history if item['needs_revision'])
        needs_revision_rate = needs_revision_count / len(self.monitoring_history)
        
        return {
            "total_monitored": len(self.monitoring_history),
            "avg_quality": avg_quality,
            "needs_revision_rate": needs_revision_rate,
            "recent_avg_quality": sum(item['quality_scores']['overall'] 
                                    for item in self.monitoring_history[-10:]) / min(10, len(self.monitoring_history))
        }


class SafetySystem:
    """Integrated safety system combining all safety components."""
    
    def __init__(self, config: SafetyConfig):
        self.config = config
        
        # Initialize safety components based on config
        self.cognitive_inhibition = CognitiveInhibition() if config.enable_cognitive_inhibition else None
        self.contextual_gating = ContextualGating() if config.enable_contextual_gating else None
        self.metacognitive_monitoring = MetacognitiveMonitoring() if config.enable_metacognitive_monitoring else None
    
    def filter_response(self, response: str, context: Dict[str, Any]) -> str:
        """Apply complete safety filtering to response."""
        filtered_response = response
        
        # Apply cognitive inhibition
        if self.cognitive_inhibition:
            filtered_response = self.cognitive_inhibition.inhibit_inappropriate_response(
                filtered_response, context
            )
        
        return filtered_response
    
    def should_encode_memory(self, content: str, context: Dict[str, Any]) -> Tuple[bool, float]:
        """Determine if content should be encoded to memory."""
        if self.contextual_gating:
            return self.contextual_gating.should_encode_memory(content, context)
        else:
            return True, 0.5  # Default: encode with medium strength
    
    def should_retrieve_memory(self, query: str, memory_item: Dict[str, Any]) -> Tuple[bool, float]:
        """Determine if memory should be retrieved."""
        if self.contextual_gating:
            return self.contextual_gating.should_retrieve_memory(query, memory_item)
        else:
            return True, 0.5  # Default: retrieve with medium strength
    
    def monitor_response_quality(self, response: str, context: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Monitor response quality."""
        if self.metacognitive_monitoring:
            return self.metacognitive_monitoring.monitor_response_quality(response, context)
        else:
            return None
    
    def get_safety_stats(self) -> Dict[str, Any]:
        """Get comprehensive safety statistics."""
        stats = {
            "config": {
                "cognitive_inhibition_enabled": self.config.enable_cognitive_inhibition,
                "contextual_gating_enabled": self.config.enable_contextual_gating,
                "metacognitive_monitoring_enabled": self.config.enable_metacognitive_monitoring
            }
        }
        
        if self.cognitive_inhibition:
            stats["violations"] = self.cognitive_inhibition.get_violation_stats()
        
        if self.metacognitive_monitoring:
            stats["quality_monitoring"] = self.metacognitive_monitoring.get_monitoring_stats()
        
        return stats
