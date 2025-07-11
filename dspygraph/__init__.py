"""
DSPy Graph Framework

A minimal framework for building graph-based workflows with DSPy nodes.
"""

from .graph import END, START, Graph
from .node import Node

__all__ = ["Node", "Graph", "START", "END"]
__version__ = "0.1.0"
