"""
Graph-based implementation for the question classifier application
"""
import dspy
from dspygraph import Graph, START, END
from .nodes import QuestionClassifierNode, FactualAnswerNode, CreativeResponseNode, ToolUseNode


def create_question_classifier_graph() -> Graph:
    """
    Create the question classifier graph
    
    Returns:
        Configured Graph ready to run
    """
    # Create graph
    graph = Graph("QuestionClassifier")
    
    # Create and add nodes
    classifier = QuestionClassifierNode("classifier")
    factual = FactualAnswerNode("factual_answer")
    creative = CreativeResponseNode("creative_response") 
    tool_use = ToolUseNode("tool_use")
    
    # Add nodes to graph
    graph.add_node(classifier)
    graph.add_node(factual)
    graph.add_node(creative)
    graph.add_node(tool_use)
    
    # Add explicit START edge
    graph.add_edge(START, "classifier")
    
    # Define routing logic
    def route_by_classification(state):
        """Route based on classification result"""
        classification = state.get("classification", "")
        if "factual" in classification:
            return "factual"
        elif "creative" in classification:
            return "creative"
        elif "tool_use" in classification:
            return "tool_use"
        else:
            return "unknown"
    
    # Add conditional edges from classifier to response nodes
    graph.add_conditional_edges(
        "classifier",
        {
            "factual": "factual_answer",
            "creative": "creative_response", 
            "tool_use": "tool_use",
            "unknown": END
        },
        route_by_classification
    )
    
    # Add explicit END edges for each response type
    graph.add_edge("factual_answer", END)
    graph.add_edge("creative_response", END)
    graph.add_edge("tool_use", END)
    
    # Load compiled classifier
    try:
        classifier.load_compiled("compiled_classifier.json")
        print("Loaded compiled classifier for graph")
    except:
        print("Warning: No compiled classifier found. Run compile_classifier.py first")
    
    return graph


def run_question_classifier(question: str) -> dict:
    """
    Run the question classifier graph on a single question
    
    Args:
        question: The question to classify and answer
        
    Returns:
        Complete graph execution result
    """
    # Configure DSPy
    lm = dspy.LM("openai/gpt-4o-mini")
    dspy.configure(lm=lm)
    
    # Create and run graph
    graph = create_question_classifier_graph()
    
    # Execute with the question
    result = graph.run(question=question)
    
    return result


if __name__ == "__main__":
    # Test the graph
    test_questions = [
        "What is the capital of France?",
        "What is 123 + 456?", 
        "Write a short poem about a cat."
    ]
    
    for question in test_questions:
        print(f"\n🤔 Question: {question}")
        result = run_question_classifier(question)
        print(f"💡 Answer: {result.get('response', 'No response generated')}")