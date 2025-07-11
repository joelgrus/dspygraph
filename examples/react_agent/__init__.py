"""
React Agent Example

A DSPy graph-based implementation of a ReAct (Reasoning + Acting) agent that can use tools
to solve problems through iterative reasoning and action execution.
"""

from .graph import create_react_agent_graph, run_react_agent
from .nodes import MaxStepsNode, ReactAgentNode, ToolExecutorNode
from .tools import CalculatorTool, SearchTool, execute_tool, get_available_tools
from .types import ActionType, ReactState, ToolResult

__all__ = [
    "create_react_agent_graph",
    "run_react_agent",
    "ReactAgentNode",
    "ToolExecutorNode",
    "MaxStepsNode",
    "CalculatorTool",
    "SearchTool",
    "get_available_tools",
    "execute_tool",
    "ReactState",
    "ActionType",
    "ToolResult",
]
