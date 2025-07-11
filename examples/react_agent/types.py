"""
Type definitions for the React agent example
"""
from typing import TypedDict, List, Dict, Any, Optional, Literal

class ReactState(TypedDict):
    """State structure for the React agent graph"""
    question: str
    thoughts: List[str]
    actions: List[str] 
    observations: List[str]
    current_thought: str
    current_action: str
    current_observation: str
    final_answer: Optional[str]
    step_count: int
    max_steps: int

ActionType = Literal['calculate', 'search', 'finish']

class ToolResult(TypedDict):
    """Result from tool execution"""
    success: bool
    result: str
    error: Optional[str]