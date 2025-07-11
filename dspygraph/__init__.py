"""
DSPy Graph Framework

A minimal framework for building graph-based workflows with DSPy nodes.
"""
from .node import Node
from .workflow import Workflow, START, END

__all__ = ["Node", "Workflow", "START", "END"]
__version__ = "0.1.0"