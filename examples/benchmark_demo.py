#!/usr/bin/env python3
"""
ARC Core Benchmarking Demo

This example demonstrates how to use the ARC Core v1.1.0 benchmarking system
to evaluate and compare AI models. Perfect for researchers who want to measure
the impact of ARC's biological learning mechanisms.

Run this demo:
    python examples/benchmark_demo.py
"""

import arc_core
from datetime import datetime
import json
import os


def create_sample_benchmark_suite():
    """Create a sample benchmark suite for demonstration."""
    
    print("📋 Creating Sample Benchmark Suite")
    print("-" * 50)
    
    # Create diverse test cases covering different capabilities
    test_cases = [
        # Mathematical reasoning
        {"prompt": "What is 15 × 8?", "expected": "120", "category": "math"},
        {"prompt": "If a triangle has angles of 60° and 70°, what is the third angle?", "expected": "50°", "category": "math"},
        
        # Factual knowledge
        {"prompt": "What is the capital of Japan?", "expected": "Tokyo", "category": "geography"},
        {"prompt": "Who wrote 'Romeo and Juliet'?", "expected": "William Shakespeare", "category": "literature"},
        
        # Scientific knowledge
        {"prompt": "What is the chemical symbol for gold?", "expected": "Au", "category": "science"},
        {"prompt": "How many planets are in our solar system?", "expected": "8", "category": "science"},
        
        # Language understanding
        {"prompt": "Complete the phrase: 'A bird in the hand is worth...'", "expected": "two in the bush", "category": "language"},
        {"prompt": "What is the opposite of 'empty'?", "expected": "full", "category": "language"},
        
        # Logical reasoning
        {"prompt": "If all roses are flowers, and all flowers are plants, then all roses are:", "expected": "plants", "category": "logic"},
        {"prompt": "Complete the pattern: 2, 4, 6, 8, ?", "expected": "10", "category": "logic"}
    ]
    
    # Create benchmark suite
    suite = arc_core.BenchmarkSuite(
        name="arc-demo-suite",
        description="Comprehensive demo benchmark suite for ARC Core evaluation",
        test_cases=test_cases
    )
    
    print(f"✅ Created benchmark suite: {suite.name}")
    print(f"   - Total test cases: {len(suite.test_cases)}")
    print(f"   - Categories: {sorted(list(suite.get_categories()))}")
    print(f"   - Sample prompts: {suite.prompts[:3]}")
    
    return suite


def demonstrate_metrics_creation():
    """Demonstrate creating and working with benchmark metrics."""
    
    print("\n📊 Demonstrating Benchmark Metrics")
    print("-" * 50)
    
    # Simulate results from a base model
    base_model_metrics = arc_core.BenchmarkMetrics(
        model_name="gpt2-base",
        num_samples=10,
        perplexity=25.8,
        avg_latency_ms=95.2,
        peak_memory_mb=384.0,
        coherence_score=0.72,
        toxicity_score=0.12,
        factual_accuracy=0.65,
        response_length_avg=38.5,
        timestamp=datetime.now().isoformat()
    )
    
    # Simulate results from ARC-enhanced model
    arc_enhanced_metrics = arc_core.BenchmarkMetrics(
        model_name="gpt2-arc-enhanced",
        num_samples=10,
        perplexity=18.3,  # Lower is better
        avg_latency_ms=105.8,  # Slightly higher due to processing
        peak_memory_mb=456.0,  # Higher due to additional mechanisms
        coherence_score=0.89,  # Much better
        toxicity_score=0.03,  # Much lower (better)
        factual_accuracy=0.94,  # Much higher
        response_length_avg=42.1,
        timestamp=datetime.now().isoformat()
    )
    
    print("📈 Performance Comparison:")
    print(f"                    Base Model  │  ARC Enhanced  │  Improvement")
    print(f"   Perplexity:      {base_model_metrics.perplexity:8.1f}  │      {arc_enhanced_metrics.perplexity:8.1f}  │    {((base_model_metrics.perplexity - arc_enhanced_metrics.perplexity) / base_model_metrics.perplexity * 100):+5.1f}%")
    print(f"   Coherence:       {base_model_metrics.coherence_score:8.3f}  │      {arc_enhanced_metrics.coherence_score:8.3f}  │    {((arc_enhanced_metrics.coherence_score - base_model_metrics.coherence_score) / base_model_metrics.coherence_score * 100):+5.1f}%")
    print(f"   Factual Acc.:    {base_model_metrics.factual_accuracy:8.3f}  │      {arc_enhanced_metrics.factual_accuracy:8.3f}  │    {((arc_enhanced_metrics.factual_accuracy - base_model_metrics.factual_accuracy) / base_model_metrics.factual_accuracy * 100):+5.1f}%")
    print(f"   Toxicity:        {base_model_metrics.toxicity_score:8.3f}  │      {arc_enhanced_metrics.toxicity_score:8.3f}  │    {((base_model_metrics.toxicity_score - arc_enhanced_metrics.toxicity_score) / base_model_metrics.toxicity_score * 100):+5.1f}%")
    
    return base_model_metrics, arc_enhanced_metrics


def save_benchmark_results(suite, base_metrics, arc_metrics):
    """Save benchmark results to JSON file."""
    
    print("\n💾 Saving Benchmark Results")
    print("-" * 50)
    
    results = {
        "benchmark_suite": {
            "name": suite.name,
            "description": suite.description,
            "test_cases": len(suite.test_cases),
            "categories": sorted(list(suite.get_categories()))
        },
        "evaluation_results": {
            "base_model": base_metrics.to_dict(),
            "arc_enhanced_model": arc_metrics.to_dict()
        },
        "performance_improvements": {
            "perplexity_reduction": f"{((base_metrics.perplexity - arc_metrics.perplexity) / base_metrics.perplexity * 100):+.1f}%",
            "coherence_improvement": f"{((arc_metrics.coherence_score - base_metrics.coherence_score) / base_metrics.coherence_score * 100):+.1f}%",
            "factual_accuracy_improvement": f"{((arc_metrics.factual_accuracy - base_metrics.factual_accuracy) / base_metrics.factual_accuracy * 100):+.1f}%",
            "toxicity_reduction": f"{((base_metrics.toxicity_score - arc_metrics.toxicity_score) / base_metrics.toxicity_score * 100):+.1f}%"
        },
        "generated_at": datetime.now().isoformat(),
        "arc_core_version": getattr(arc_core, '__version__', '1.1.0')
    }
    
    # Save to file
    results_file = "benchmark_comparison_results.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"✅ Results saved to: {results_file}")
    print(f"   File size: {os.path.getsize(results_file)} bytes")
    
    return results_file


def demonstrate_cli_usage():
    """Show CLI usage examples."""
    
    print("\n🚀 CLI Usage Examples")
    print("-" * 50)
    
    print("The ARC Core v1.1.0 benchmarking system includes powerful CLI commands:")
    print()
    print("📋 Basic benchmark evaluation:")
    print("   arc bench --suite my_benchmark.jsonl")
    print()
    print("🆚 Compare base model vs ARC-enhanced:")
    print("   arc bench --suite tests.jsonl --model gpt2 --output comparison.json")
    print()
    print("📊 Generate markdown report:")
    print("   arc bench --suite evaluation.jsonl --format markdown --output report.md")
    print()
    print("🎯 Limit evaluation samples:")
    print("   arc bench --suite large_suite.jsonl --max-samples 50")
    print()
    print("💡 Get help on all options:")
    print("   arc bench --help")


def main():
    """Main demo function."""
    
    print("🧪 ARC Core v1.1.0 Benchmarking System Demo")
    print("=" * 70)
    print(f"ARC Core Version: {getattr(arc_core, '__version__', '1.1.0')}")
    print("This demo shows how to evaluate and compare AI models using ARC Core's")
    print("comprehensive benchmarking infrastructure.")
    print()
    
    try:
        # Step 1: Create benchmark suite
        suite = create_sample_benchmark_suite()
        
        # Step 2: Demonstrate metrics
        base_metrics, arc_metrics = demonstrate_metrics_creation()
        
        # Step 3: Save results
        results_file = save_benchmark_results(suite, base_metrics, arc_metrics)
        
        # Step 4: Show CLI usage
        demonstrate_cli_usage()
        
        # Final summary
        print("\n" + "=" * 70)
        print("🎉 Demo Complete! Key Takeaways:")
        print("✅ ARC Core provides comprehensive model evaluation capabilities")
        print("✅ Easy-to-use API for creating benchmark suites and metrics")
        print("✅ Professional JSON export for research and documentation")
        print("✅ Powerful CLI tools for batch evaluation and comparison")
        print("✅ Scientifically rigorous performance measurement")
        print()
        print("🔬 Perfect for AI researchers measuring the impact of:")
        print("   • Biological learning mechanisms")
        print("   • Continual learning without catastrophic forgetting")
        print("   • Advanced memory systems and consciousness models")
        print()
        print(f"📁 Results saved to: {results_file}")
        print("🚀 Ready to benchmark your own models with ARC Core!")
        
    except Exception as e:
        print(f"❌ Error during demo: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
