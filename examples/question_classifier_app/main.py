"""
Main application using the graph-based DSPy framework
"""
import dspy
from .graph import create_question_classifier_graph


def main() -> None:
    """Main application entry point using DSPy graph framework"""
    # Configure DSPy 
    lm = dspy.LM("openai/gpt-4o-mini")
    dspy.configure(lm=lm)
    
    # Enable DSPy observability
    dspy.enable_logging()
    
    # Create the graph
    try:
        graph = create_question_classifier_graph()
    except Exception as e:
        print(f"Failed to create graph: {e}")
        return
    
    # Print graph structure
    print("Graph Structure:")
    print(graph.visualize())
    print()
    
    # Run test cases
    test_cases = [
        ("What is the capital of France?", "Factual Question"),
        ("What is 123 + 456?", "Tool Use Question"),
        ("Write a short poem about a cat.", "Creative Question"),
    ]
    
    for question, description in test_cases:
        print(f"\n🔍 Testing: {description}")
        print(f"Question: {question}")
        
        try:
            # Execute graph with full observability
            result = graph.run(question=question)
            
            print(f"✅ Final Result: {result.get('response', 'No response generated')}")
            
            # Show graph metadata
            metadata = result.get('_graph_metadata', {})
            print(f"⏱️  Execution time: {metadata.get('execution_time', 0):.3f}s")
            print(f"📊 Total usage: {metadata.get('total_usage', {})}")
            print(f"🔄 Execution order: {' -> '.join(metadata.get('execution_order', []))}")
            
        except Exception as e:
            print(f"❌ Execution failed: {e}")


if __name__ == "__main__":
    main()