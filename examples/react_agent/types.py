"""
Type definitions for the React agent example
"""

from typing import Literal, TypedDict


class ReactState(TypedDict):
    """State structure for the React agent graph"""

    question: str
    thoughts: list[str]
    actions: list[str]
    observations: list[str]
    current_thought: str
    current_action: str
    current_observation: str
    final_answer: str | None
    step_count: int
    max_steps: int


ActionType = Literal["calculate", "search", "finish"]


class ToolResult(TypedDict):
    """Result from tool execution"""

    success: bool
    result: str
    error: str | None
