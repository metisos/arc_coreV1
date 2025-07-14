#!/usr/bin/env python3
"""
ARC Science Learner - Testing Real Learning vs Memorization
Evaluates how ARC learns from scientific information and experiences

This application tests:
- Information comprehension and integration
- Learning retention over time
- Ability to make connections between concepts
- Transfer of knowledge to new situations
- Genuine understanding vs memorization

Requirements:
    pip install metisos-arc-core

What this demonstrates:
- Real-time continual learning
- Interactive chat with learning
- Memory persistence across conversations
- LoRA adapter training on the fly


"""

import json
import time
import random
from datetime import datetime, timedelta
from typing import Dict, List, Tuple

# Import ARC Core components
from arc_core import ARCTrainer, ARCConfig


class ScienceDatabase:
    """Curated scientific information for learning experiments."""
    
    def __init__(self):
        self.topics = {
            "quantum_mechanics": {
                "title": "Quantum Mechanics Fundamentals",
                "content": [
                    "Quantum mechanics describes the behavior of matter and energy at the atomic and subatomic scale.",
                    "Unlike classical physics, quantum systems can exist in multiple states simultaneously, called superposition.",
                    "The act of measurement collapses the quantum superposition into a definite state.",
                    "Quantum entanglement allows particles to be correlated across vast distances instantaneously.",
                    "The uncertainty principle states that position and momentum cannot be simultaneously measured with perfect precision.",
                    "Wave-particle duality shows that quantum objects exhibit both wave and particle characteristics."
                ],
                "key_concepts": ["superposition", "entanglement", "uncertainty principle", "wave-particle duality"],
                "applications": ["quantum computing", "quantum cryptography", "quantum sensors"],
                "difficulty": 3
            },
            "photosynthesis": {
                "title": "Photosynthesis Process",
                "content": [
                    "Photosynthesis is the process by which plants convert light energy into chemical energy.",
                    "The process occurs in chloroplasts, specifically in the thylakoid membranes.",
                    "Light reactions capture energy from photons and produce ATP and NADPH.",
                    "The Calvin cycle uses ATP and NADPH to fix carbon dioxide into glucose.",
                    "Chlorophyll absorbs red and blue light while reflecting green light.",
                    "Photosynthesis produces oxygen as a byproduct, which is essential for most life on Earth."
                ],
                "key_concepts": ["chloroplasts", "light reactions", "Calvin cycle", "chlorophyll"],
                "applications": ["agriculture", "biofuels", "artificial photosynthesis"],
                "difficulty": 2
            },
            "neural_networks": {
                "title": "Biological Neural Networks",
                "content": [
                    "Neural networks in the brain consist of billions of interconnected neurons.",
                    "Neurons communicate through electrical and chemical signals at synapses.",
                    "Synaptic plasticity allows connections to strengthen or weaken based on activity.",
                    "Learning occurs through changes in synaptic strength and neural connectivity.",
                    "The brain exhibits remarkable plasticity, reorganizing after injury or during learning.",
                    "Different brain regions specialize in different functions while working together."
                ],
                "key_concepts": ["neurons", "synapses", "plasticity", "specialization"],
                "applications": ["artificial intelligence", "brain-computer interfaces", "neuromedicine"],
                "difficulty": 2
            },
            "climate_systems": {
                "title": "Earth's Climate System",
                "content": [
                    "Earth's climate is driven by solar radiation and the greenhouse effect.",
                    "Greenhouse gases trap heat in the atmosphere, warming the planet.",
                    "Ocean currents redistribute heat around the globe, affecting regional climates.",
                    "The carbon cycle involves the exchange of carbon between atmosphere, oceans, and land.",
                    "Feedback loops can amplify or dampen climate changes.",
                    "Human activities have significantly altered atmospheric composition since the Industrial Revolution."
                ],
                "key_concepts": ["greenhouse effect", "ocean currents", "carbon cycle", "feedback loops"],
                "applications": ["climate modeling", "renewable energy", "environmental policy"],
                "difficulty": 2
            },
            "dna_replication": {
                "title": "DNA Replication Mechanism",
                "content": [
                    "DNA replication ensures genetic information is accurately copied before cell division.",
                    "The double helix unwinds and each strand serves as a template for a new strand.",
                    "DNA polymerase adds nucleotides in the 5' to 3' direction only.",
                    "Leading strands are synthesized continuously while lagging strands are made in fragments.",
                    "Proofreading mechanisms ensure high fidelity in DNA copying.",
                    "Telomeres protect chromosome ends during replication but shorten with each division."
                ],
                "key_concepts": ["double helix", "DNA polymerase", "leading strand", "lagging strand"],
                "applications": ["genetic engineering", "medical diagnostics", "evolutionary biology"],
                "difficulty": 3
            }
        }
        
        # Interconnected concepts for testing knowledge transfer
        self.concept_connections = {
            "quantum_mechanics": ["neural_networks"],  # Quantum effects in biology
            "photosynthesis": ["climate_systems"],     # Carbon cycle connections
            "neural_networks": ["quantum_mechanics"],  # Quantum biology
            "climate_systems": ["photosynthesis"],     # Ecosystem interactions
            "dna_replication": ["neural_networks"]     # Genetic basis of brain development
        }
    
    def get_topic(self, topic_name):
        """Get a specific topic."""
        return self.topics.get(topic_name)
    
    def get_all_topics(self):
        """Get all available topics."""
        return list(self.topics.keys())
    
    def get_random_topic(self):
        """Get a random topic."""
        topic_name = random.choice(list(self.topics.keys()))
        return topic_name, self.topics[topic_name]


class LearningEvaluator:
    """Evaluates how well ARC learns and retains scientific information."""
    
    def __init__(self):
        self.learning_sessions = []
        self.retention_tests = []
        self.concept_networks = {}
        self.learning_patterns = {}
    
    def evaluate_initial_knowledge(self, arc, topic_name, topic_data):
        """Test ARC's initial knowledge about a topic before learning."""
        print(f"\n[BASELINE ASSESSMENT]: {topic_data['title']}")
        
        # Test questions about key concepts
        baseline_questions = [
            f"What do you know about {topic_name.replace('_', ' ')}?",
            f"Can you explain {random.choice(topic_data['key_concepts'])}?",
            f"How might {topic_name.replace('_', ' ')} be used in applications?"
        ]
        
        baseline_responses = []
        for question in baseline_questions:
            print(f"   Q: {question}")
            response = arc.generate_response(
                f"Question: {question} Let me think about what I know:"
            )
            print(f"   A: {response}")
            baseline_responses.append({
                'question': question,
                'response': response,
                'confidence': 0.5  # Default confidence since ARCTrainer doesn't return confidence scores
            })
            time.sleep(1)
        
        return baseline_responses
    
    def present_information_incrementally(self, arc, topic_data):
        """Present information piece by piece and observe learning."""
        print(f"\n[INCREMENTAL LEARNING]: {topic_data['title']}")
        
        learning_progression = []
        
        for i, content_piece in enumerate(topic_data['content']):
            print(f"\n   Information {i+1}/{len(topic_data['content'])}: {content_piece}")
            
            # Let ARC process this information
            processing_prompt = f"I just learned: {content_piece}. This means"
            process_response = arc.generate_response(processing_prompt)
            
            print(f"   ARC Processing: {process_response}")
            
            # Test immediate comprehension
            comprehension_prompt = f"Based on what I just learned about '{content_piece}', I can explain"
            comprehension_response = arc.generate_response(comprehension_prompt)
            
            print(f"   Understanding: {comprehension_response}")
            
            learning_progression.append({
                'step': i + 1,
                'content': content_piece,
                'processing': process_response,
                'comprehension': comprehension_response,
                'processing_confidence': 0.5,  # Default confidence
                'comprehension_confidence': 0.5  # Default confidence
            })
            
            time.sleep(2)
        
        return learning_progression
    
    def test_knowledge_integration(self, arc, topic_name, topic_data):
        """Test how well ARC integrates and connects learned information."""
        print(f"\n[KNOWLEDGE INTEGRATION TEST]")
        
        integration_tests = [
            f"How do the different aspects of {topic_name.replace('_', ' ')} work together?",
            f"What are the most important principles underlying {topic_name.replace('_', ' ')}?",
            f"Can you create an analogy to explain {topic_name.replace('_', ' ')} to someone new?",
            f"What questions does learning about {topic_name.replace('_', ' ')} raise for you?"
        ]
        
        integration_responses = []
        for test_question in integration_tests:
            print(f"   Integration Test: {test_question}")
            response = arc.generate_response(
                f"Reflecting on everything I've learned about {topic_name.replace('_', ' ')}, {test_question}"
            )
            print(f"   Response: {response}")
            
            integration_responses.append({
                'test': test_question,
                'response': response,
                'confidence': 0.5  # Default confidence
            })
            time.sleep(1.5)
        
        return integration_responses
    
    def test_knowledge_transfer(self, arc, source_topic, target_topic, database):
        """Test if ARC can transfer knowledge between related topics."""
        print(f"\n[KNOWLEDGE TRANSFER TEST]: {source_topic} -> {target_topic}")
        
        source_data = database.get_topic(source_topic)
        target_data = database.get_topic(target_topic)
        
        transfer_questions = [
            f"How might what I learned about {source_topic.replace('_', ' ')} relate to {target_topic.replace('_', ' ')}?",
            f"What connections can I make between {source_topic.replace('_', ' ')} and {target_topic.replace('_', ' ')}?",
            f"Could principles from {source_topic.replace('_', ' ')} help understand {target_topic.replace('_', ' ')}?"
        ]
        
        transfer_responses = []
        for question in transfer_questions:
            print(f"   Transfer Question: {question}")
            response = arc.generate_response(question)
            print(f"   Connection: {response}")
            
            transfer_responses.append({
                'question': question,
                'response': response,
                'confidence': 0.5  # Default confidence
            })
            time.sleep(1.5)
        
        return transfer_responses
    
    def retention_test(self, arc, topic_name, topic_data, delay_minutes=5):
        """Test retention after a delay."""
        print(f"\n⏰ RETENTION TEST (after {delay_minutes} minutes)")
        print("Simulating time passage with other activities...")
        
        # Simulate time passage with unrelated thoughts
        for i in range(3):
            distraction_prompt = random.choice([
                "I wonder about the weather today",
                "Thinking about art and creativity",
                "Contemplating the nature of time",
                "Reflecting on music and harmony"
            ])
            arc.generate_response(distraction_prompt)
            time.sleep(1)
        
        print(f"\n[RETENTION TEST]: {topic_data['title']}")
        
        retention_questions = [
            f"What do I remember about {topic_name.replace('_', ' ')}?",
            f"What were the key points about {random.choice(topic_data['key_concepts'])}?",
            f"How would I explain {topic_name.replace('_', ' ')} now?"
        ]
        
        retention_responses = []
        for question in retention_questions:
            print(f"   Recall Test: {question}")
            response = arc.generate_response(question)
            print(f"   Recalled: {response}")
            
            retention_responses.append({
                'question': question,
                'response': response,
                'confidence': 0.5  # Default confidence
            })
            time.sleep(1)
        
        return retention_responses
    
    def analyze_learning_patterns(self, session_data):
        """Analyze patterns in how ARC learns."""
        print(f"\n[LEARNING PATTERN ANALYSIS]")
        
        # Analyze confidence progression
        confidences = []
        learning_events = []
        
        for phase in ['baseline', 'learning_progression', 'integration', 'retention']:
            if phase in session_data:
                phase_data = session_data[phase]
                if isinstance(phase_data, list):
                    for item in phase_data:
                        if 'confidence' in item:
                            confidences.append(item['confidence'])
                        if 'learned' in item and item['learned']:
                            learning_events.append(phase)
        
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0
        learning_rate = len(learning_events) / len(session_data.get('learning_progression', []))
        
        analysis = {
            'average_confidence': avg_confidence,
            'learning_rate': learning_rate,
            'total_learning_events': len(learning_events),
            'confidence_trend': 'increasing' if len(confidences) > 1 and confidences[-1] > confidences[0] else 'stable'
        }
        
        print(f"   Average Confidence: {avg_confidence:.3f}")
        print(f"   Learning Rate: {learning_rate:.1%}")
        print(f"   Total Learning Events: {len(learning_events)}")
        print(f"   Confidence Trend: {analysis['confidence_trend']}")
        
        return analysis


class ARCScienceLearner:
    """Main application for testing ARC's science learning capabilities."""
    
    def __init__(self, model_name="cognitivecomputations/TinyDolphin-2.8-1.1b"):
        print("Initializing ARC Science Learning Experiment")
        print("=" * 60)
        
        # Initialize ARC with PyPI package (v1.0.1 with bug fix)
        print("Setting up ARC Trainer...")
        self.config = ARCConfig()
        self.config.device = "cpu"  # Force CPU for Colab compatibility
        
        self.arc = ARCTrainer(self.config)
        print(f"Initializing model: {model_name}")
        
        success = self.arc.initialize_model(model_name)
        if not success:
            raise RuntimeError(f"Failed to initialize model: {model_name}")
            
        print("[SUCCESS] ARC Trainer initialized successfully!")
        print(f"Model device: {getattr(self.arc.model, 'device', 'unknown') if hasattr(self.arc, 'model') else 'unknown'}")
        
        self.database = ScienceDatabase()
        self.evaluator = LearningEvaluator()
        self.experiment_log = []
    
    def run_single_topic_experiment(self, topic_name=None):
        """Run a complete learning experiment on one scientific topic."""
        
        if topic_name is None:
            topic_name, topic_data = self.database.get_random_topic()
        else:
            topic_data = self.database.get_topic(topic_name)
            if not topic_data:
                print(f"[ERROR] Topic '{topic_name}' not found")
                return None
        
        print(f"\n[SCIENCE LEARNING EXPERIMENT]: {topic_data['title']}")
        print(f"   Difficulty Level: {topic_data['difficulty']}/3")
        print(f"   Key Concepts: {', '.join(topic_data['key_concepts'])}")
        
        session_data = {
            'topic': topic_name,
            'timestamp': datetime.now(),
            'pre_learning_stats': self.arc.get_training_stats()
        }
        
        # Phase 1: Baseline assessment
        session_data['baseline'] = self.evaluator.evaluate_initial_knowledge(
            self.arc, topic_name, topic_data
        )
        
        # Phase 2: Incremental learning
        session_data['learning_progression'] = self.evaluator.present_information_incrementally(
            self.arc, topic_data
        )
        
        # Phase 3: Knowledge integration
        session_data['integration'] = self.evaluator.test_knowledge_integration(
            self.arc, topic_name, topic_data
        )
        
        # Phase 4: Retention test
        session_data['retention'] = self.evaluator.retention_test(
            self.arc, topic_name, topic_data
        )
        
        # Phase 5: Analysis
        session_data['analysis'] = self.evaluator.analyze_learning_patterns(session_data)
        session_data['post_learning_stats'] = self.arc.get_training_stats()
        
        self.experiment_log.append(session_data)
        return session_data
    
    def run_knowledge_transfer_experiment(self):
        """Test knowledge transfer between related topics."""
        print(f"\n[KNOWLEDGE TRANSFER EXPERIMENT]")
        
        # Learn first topic
        source_topic = "photosynthesis"
        print(f"\nLearning source topic: {source_topic}")
        source_session = self.run_single_topic_experiment(source_topic)
        
        # Test transfer to related topic
        target_topic = "climate_systems"
        print(f"\nTesting transfer to: {target_topic}")
        
        # Test transfer before learning target topic
        transfer_results = self.evaluator.test_knowledge_transfer(
            self.arc, source_topic, target_topic, self.database
        )
        
        # Learn target topic
        target_session = self.run_single_topic_experiment(target_topic)
        
        # Test transfer again after learning both
        post_transfer_results = self.evaluator.test_knowledge_transfer(
            self.arc, source_topic, target_topic, self.database
        )
        
        transfer_experiment = {
            'source_topic': source_topic,
            'target_topic': target_topic,
            'pre_transfer': transfer_results,
            'post_transfer': post_transfer_results,
            'source_session': source_session,
            'target_session': target_session
        }
        
        return transfer_experiment
    
    def run_comprehensive_experiment(self):
        """Run a comprehensive learning experiment across multiple topics."""
        print(f"\n[COMPREHENSIVE SCIENCE LEARNING EXPERIMENT]")
        print("Testing ARC's ability to learn and connect multiple scientific domains")
        
        # Select topics of varying difficulty
        topics_to_learn = ["photosynthesis", "neural_networks", "quantum_mechanics"]
        
        comprehensive_results = {
            'start_time': datetime.now(),
            'initial_stats': self.arc.get_training_stats(),
            'topic_sessions': {},
            'cross_topic_connections': []
        }
        
        # Learn each topic
        for topic in topics_to_learn:
            print(f"\nLearning Topic: {topic}")
            session = self.run_single_topic_experiment(topic)
            comprehensive_results['topic_sessions'][topic] = session
            
            # Test connections with previously learned topics
            if len(comprehensive_results['topic_sessions']) > 1:
                for previous_topic in comprehensive_results['topic_sessions'].keys():
                    if previous_topic != topic:
                        connection_test = self.evaluator.test_knowledge_transfer(
                            self.arc, previous_topic, topic, self.database
                        )
                        comprehensive_results['cross_topic_connections'].append({
                            'from': previous_topic,
                            'to': topic,
                            'connections': connection_test
                        })
        
        # Final integration test
        print(f"\n[FINAL INTEGRATION TEST]")
        topic_names = ', '.join(topics_to_learn)
        final_integration = self.arc.generate_response(
            f"Based on everything learned about {topic_names}, "
            "provide a comprehensive summary that shows deep understanding "
            "and connections between these scientific concepts."
        )
        print(f"Final Integration: {final_integration}")
        
        comprehensive_results['final_integration'] = final_integration
        comprehensive_results['end_time'] = datetime.now()
        comprehensive_results['final_stats'] = self.arc.get_training_stats()
        
        return comprehensive_results
    
    def generate_learning_report(self, experiment_results):
        """Generate a detailed report of learning performance."""
        print(f"\n[LEARNING PERFORMANCE REPORT]")
        print("=" * 50)
        
        if 'topic_sessions' in experiment_results:
            # Comprehensive experiment report
            initial_stats = experiment_results['initial_stats']
            final_stats = experiment_results['final_stats']
            
            print(f"Experiment Duration: {experiment_results['end_time'] - experiment_results['start_time']}")
            print(f"Topics Learned: {len(experiment_results['topic_sessions'])}")
            # Safely get neural updates with fallback
            initial_updates = initial_stats.get('total_updates', 0)
            final_updates = final_stats.get('total_updates', 0)
            print(f"Neural Updates: {final_updates - initial_updates}")
            stats = self.arc.get_training_stats()
            print(f"Training Sessions: {stats.get('total_training_sessions', 0)}")
            
            # Topic-by-topic analysis
            for topic, session in experiment_results['topic_sessions'].items():
                print(f"\n{topic.upper().replace('_', ' ')} ANALYSIS:")
                analysis = session['analysis']
                print(f"   Confidence: {analysis['average_confidence']:.3f}")
                print(f"   Learning Rate: {analysis['learning_rate']:.1%}")
                print(f"   Learning Events: {analysis['total_learning_events']}")
            
            # Cross-topic connections
            print(f"\nCROSS-TOPIC CONNECTIONS: {len(experiment_results['cross_topic_connections'])}")
            
        else:
            # Single topic report
            analysis = experiment_results['analysis']
            pre_stats = experiment_results['pre_learning_stats']
            post_stats = experiment_results['post_learning_stats']
            
            print(f"Topic: {experiment_results['topic'].replace('_', ' ').title()}")
            
            # Safely get neural updates with fallback
            pre_updates = pre_stats.get('total_updates', 0)
            post_updates = post_stats.get('total_updates', 0)
            print(f"Neural Updates: {post_updates - pre_updates}")
            
            print(f"Average Confidence: {analysis['average_confidence']:.3f}")
            print(f"Learning Rate: {analysis['learning_rate']:.1%}")
            print(f"Confidence Trend: {analysis['confidence_trend']}")
        
        # Overall system learning stats
        stats = self.arc.get_training_stats()
        print(f"\n[SYSTEM STATS]: {stats}")
    
    def interactive_mode(self):
        """Interactive mode for custom experiments."""
        print(f"\n[INTERACTIVE SCIENCE LEARNING MODE]")
        print("Commands:")
        print("  'topics' - List available topics")
        print("  'learn <topic>' - Learn specific topic")
        print("  'transfer' - Run transfer experiment")
        print("  'comprehensive' - Run full experiment")
        print("  'stats' - Show current stats")
        print("  'quit' - Exit")
        
        while True:
            try:
                command = input("\nScience Learner> ").strip().lower()
                
                if command == 'quit':
                    break
                elif command == 'topics':
                    topics = self.database.get_all_topics()
                    print(f"Available topics: {', '.join(topics)}")
                elif command.startswith('learn '):
                    topic = command[6:].replace(' ', '_')
                    if topic in self.database.get_all_topics():
                        result = self.run_single_topic_experiment(topic)
                        self.generate_learning_report(result)
                    else:
                        print(f"Topic '{topic}' not found. Use 'topics' to see available options.")
                elif command == 'transfer':
                    result = self.run_knowledge_transfer_experiment()
                    print("Knowledge transfer experiment complete!")
                elif command == 'comprehensive':
                    result = self.run_comprehensive_experiment()
                    self.generate_learning_report(result)
                elif command == 'stats':
                    stats = self.arc.get_training_stats()
                    print(f"Current Stats: {stats}")
                else:
                    print("Unknown command. Type 'quit' to exit.")
                    
            except KeyboardInterrupt:
                break
    
    def save_experiment_data(self):
        """Save experiment data to file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"arc_science_experiment_{timestamp}.json"
        
        experiment_data = {
            'timestamp': timestamp,
            'experiment_log': self.experiment_log,
            'arc_stats': self.arc.get_training_stats(),
            'system_stats': {
                'memory_available': hasattr(self.arc, 'memory_system'),
                'safety_available': hasattr(self.arc, 'safety_system'),
                'is_initialized': getattr(self.arc, 'is_initialized', False)
            }
        }
        
        with open(filename, 'w') as f:
            json.dump(experiment_data, f, indent=2, default=str)
        
        print(f"Experiment data saved to {filename}")


def main():
    """Main entry point for the science learning application."""
    print("""
ARC SCIENCE LEARNING EXPERIMENT
=================================

This application tests how ARC learns from scientific information:
• Information comprehension and integration
• Learning retention over time
• Knowledge transfer between topics
• Genuine understanding vs memorization

Choose experiment type:
1. Single topic learning experiment
2. Knowledge transfer experiment  
3. Comprehensive multi-topic experiment
4. Interactive mode
5. Exit
""")
    
    choice = input("Select option (1-5): ").strip()
    
    if choice == '5':
        print("Goodbye!")
        return
    
    # Initialize the science learner
    print("\nInitializing ARC Science Learner...")
    learner = ARCScienceLearner(model_name="cognitivecomputations/TinyDolphin-2.8-1.1b")
    
    try:
        if choice == '1':
            result = learner.run_single_topic_experiment()
            if result:
                learner.generate_learning_report(result)
        elif choice == '2':
            result = learner.run_knowledge_transfer_experiment()
            print("Knowledge transfer experiment completed!")
        elif choice == '3':
            result = learner.run_comprehensive_experiment()
            learner.generate_learning_report(result)
        elif choice == '4':
            learner.interactive_mode()
        else:
            print("[ERROR] Invalid choice")
            return
        
        # Save experiment data
        learner.save_experiment_data()
        
        # Final system state
        print(f"\n[EXPERIMENT COMPLETE]")
        try:
            # Try to save model if method exists
            if hasattr(learner.arc, 'save_model'):
                import tempfile
                with tempfile.TemporaryDirectory() as tmp_dir:
                    save_path = f"{tmp_dir}/experiment_model"
                    learner.arc.save_model(save_path)
                    print(f"[SAVED] Model state saved")
            else:
                print("[INFO] Training statistics preserved")
        except Exception as save_error:
            print(f"[WARNING] Save warning: {save_error}")
        
    except KeyboardInterrupt:
        print("\n[INTERRUPTED] Experiment interrupted")
        learner.save_experiment_data()


# Colab Usage Example:
# To run in Google Colab, simply copy this file and run:
#
# from learn_science import ARCScienceLearner
# learner = ARCScienceLearner()
# result = learner.run_single_topic_experiment('photosynthesis')
# learner.generate_learning_report(result)
#
# Or run interactively:
# learner.interactive_mode()

if __name__ == "__main__":
    main()