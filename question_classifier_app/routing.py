"""
Routing logic for the question classifier application
"""
from langgraph.graph import END
from .types import AgentState

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