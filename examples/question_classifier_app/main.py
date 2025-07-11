"""
Main application using the graph-based DSPy workflow
"""
import dspy
from .workflow import create_question_classifier_workflow


def main() -> None:
    """Main application entry point using DSPy graph workflow"""
    # Configure DSPy 
    lm = dspy.LM("openai/gpt-4o-mini")
    dspy.configure(lm=lm)
    
    # Enable DSPy observability
    dspy.enable_logging()
    
    # Create the workflow
    try:
        workflow = create_question_classifier_workflow()
    except Exception as e:
        print(f"Failed to create workflow: {e}")
        return
    
    # Print workflow structure
    print("Workflow Structure:")
    print(workflow.visualize())
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
            # Execute workflow with full observability
            result = workflow.run(question=question)
            
            print(f"✅ Final Result: {result.get('response', 'No response generated')}")
            
            # Show workflow metadata
            metadata = result.get('_workflow_metadata', {})
            print(f"⏱️  Execution time: {metadata.get('execution_time', 0):.3f}s")
            print(f"📊 Total usage: {metadata.get('total_usage', {})}")
            print(f"🔄 Execution order: {' -> '.join(metadata.get('execution_order', []))}")
            
        except Exception as e:
            print(f"❌ Execution failed: {e}")


if __name__ == "__main__":
    main()