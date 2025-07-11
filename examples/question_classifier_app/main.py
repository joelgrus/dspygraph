"""
Main application entry point
"""
from dspy_langgraph import configure_dspy
from . import create_graph

def main() -> None:
    """Main application entry point"""
    # Configure DSPy
    configure_dspy()
    
    # Create the application graph
    try:
        app = create_graph()
    except FileNotFoundError:
        return
    
    # Run test cases
    test_cases = [
        ("What is the capital of France?", "Factual Question"),
        ("What is 123 + 456?", "Tool Use Question"),
        ("Write a short poem about a cat.", "Creative Question"),
    ]
    
    for question, description in test_cases:
        print(f"\n--- Running Agent: {description} ---")
        result = app.invoke({"question": question})
        print(f"Final Result: {result.get('response', 'No response generated')}")

if __name__ == "__main__":
    main()