"""
DSPy-LangGraph Framework

A reusable framework for building agents that combine DSPy modules with LangGraph workflows.
"""
from .base import AgentNode
from .config import configure_dspy, get_lm

__all__ = ["AgentNode", "configure_dspy", "get_lm"]
__version__ = "0.1.0"