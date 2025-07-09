"""
Graph definition for the question classifier application
"""
from langgraph.graph import StateGraph, END
from .types import AgentState
from .agents import QuestionClassifier, FactualAnswerModule, CreativeResponseModule, ToolUseModule

def route_question(state: AgentState) -> str:
    """
    Route question based on classification
    
    Args:
        state: Current agent state
        
    Returns:
        Path to route to
    """
    classification = state["classification"]
    if "tool_use" in classification:
        return "tool_use_path"
    elif "factual" in classification:
        return "factual_path"
    elif "creative" in classification:
        return "creative_path"
    else:
        print(f"No specific path for classification: {classification}. Ending.")
        return END

def create_graph() -> StateGraph:
    """
    Create and return the compiled question classifier graph
    
    Returns:
        Compiled LangGraph application
    """
    # Initialize agent nodes
    classifier = QuestionClassifier()
    tool_use = ToolUseModule()
    factual = FactualAnswerModule()
    creative = CreativeResponseModule()
    
    # Load compiled classifier
    try:
        classifier.load_compiled("compiled_classifier.json")
        print("Compiled classifier loaded.")
    except FileNotFoundError:
        print("Compiled classifier not found. Run compilation script first: python compile_classifier.py")
        raise
    
    # Build the LangGraph workflow
    workflow = StateGraph(AgentState)
    
    workflow.add_node("classify", classifier)
    workflow.add_node("tool_use", tool_use)
    workflow.add_node("factual_answer", factual)
    workflow.add_node("creative_response", creative)
    
    workflow.set_entry_point("classify")
    
    workflow.add_conditional_edges(
        "classify",
        route_question,
        {
            "tool_use_path": "tool_use",
            "factual_path": "factual_answer",
            "creative_path": "creative_response",
            END: END,
        },
    )
    
    workflow.add_edge("tool_use", END)
    workflow.add_edge("factual_answer", END)
    workflow.add_edge("creative_response", END)
    
    return workflow.compile()