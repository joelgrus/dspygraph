"""
Type definitions for the question classifier application
"""
from typing import TypedDict, Literal

class AgentState(TypedDict):
    """State structure for the question classifier agent graph"""
    question: str
    classification: str
    response: str
    tool_output: str

QuestionCategory = Literal['factual', 'creative', 'tool_use', 'unknown']