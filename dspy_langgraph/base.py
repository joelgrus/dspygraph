"""
Base AgentNode class for DSPy-LangGraph integration
"""
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, TypeVar, Generic
import dspy
import os

StateType = TypeVar('StateType')

class AgentNode(ABC, Generic[StateType]):
    """
    Base class that unifies DSPy modules with LangGraph nodes
    
    This class provides a clean abstraction for creating agents that work
    seamlessly with both DSPy's module system and LangGraph's state management.
    """
    
    def __init__(self, compile_path: Optional[str] = None) -> None:
        """
        Initialize the agent node
        
        Args:
            compile_path: Optional path to load/save compiled model
        """
        self.module = self._create_module()
        self.compile_path = compile_path
        self._is_compiled = False
        
    @abstractmethod
    def _create_module(self) -> dspy.Module:
        """
        Create the DSPy module for this agent
        
        Returns:
            The DSPy module instance
        """
        pass
    
    @abstractmethod
    def _process_state(self, state: StateType) -> Dict[str, Any]:
        """
        Process LangGraph state and return state updates
        
        Args:
            state: The current state from LangGraph
            
        Returns:
            Dictionary of state updates to apply
        """
        pass
    
    def __call__(self, state: StateType) -> Dict[str, Any]:
        """
        LangGraph node function interface
        
        Args:
            state: The current state from LangGraph
            
        Returns:
            Dictionary of state updates to apply
        """
        return self._process_state(state)
    
    def load_compiled(self, path: Optional[str] = None) -> None:
        """
        Load a compiled version of the module
        
        Args:
            path: Optional path to load from, defaults to compile_path
            
        Raises:
            ValueError: If no path is provided
            FileNotFoundError: If the compiled file doesn't exist
        """
        load_path = path or self.compile_path
        if not load_path:
            raise ValueError("No compile path provided")
        if not os.path.exists(load_path):
            raise FileNotFoundError(f"Compiled module not found at {load_path}")
        
        self.module.load(load_path)
        self._is_compiled = True
        
    def save_compiled(self, path: Optional[str] = None) -> None:
        """
        Save the compiled module
        
        Args:
            path: Optional path to save to, defaults to compile_path
            
        Raises:
            ValueError: If no path is provided
        """
        save_path = path or self.compile_path
        if not save_path:
            raise ValueError("No compile path provided")
        
        self.module.save(save_path)
        
    @property
    def is_compiled(self) -> bool:
        """Check if module is compiled"""
        return self._is_compiled