"""
Base AgentNode class for DSPy-LangGraph integration
"""
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, TypeVar, Generic, List
import dspy
from dspy.teleprompt import Teleprompter
import os

StateType = TypeVar('StateType')

class AgentNode(ABC, Generic[StateType]):
    """
    Base class that unifies DSPy modules with LangGraph nodes
    
    This class provides a clean abstraction for creating agents that work
    seamlessly with both DSPy's module system and LangGraph's state management.
    """
    
    def __init__(self) -> None:
        """Initialize the agent node"""
        self.module = self._create_module()
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
    
    def compile(self, compiler: Teleprompter, trainset: List[dspy.Example], 
                compile_path: Optional[str] = None) -> None:
        """
        Compile using the provided DSPy compiler and training data
        
        Args:
            compiler: DSPy teleprompter instance (e.g., BootstrapFewShot, MIPRO, COPRO)
            trainset: Training data for compilation
            compile_path: Optional path to save compiled model
        """
        compiled_module = compiler.compile(self.module, trainset=trainset)
        self.module = compiled_module
        self._is_compiled = True
        if compile_path:
            self.save_compiled(compile_path)
    
    def load_compiled(self, path: str) -> None:
        """
        Load a compiled module from file
        
        Args:
            path: Path to the compiled module file
            
        Raises:
            FileNotFoundError: If the compiled file doesn't exist
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Compiled module not found at {path}")
        
        self.module.load(path)
        self._is_compiled = True
        
    def save_compiled(self, path: str) -> None:
        """
        Save the compiled module to file
        
        Args:
            path: Path to save the compiled module
        """
        self.module.save(path)
        
    @property
    def is_compiled(self) -> bool:
        """Check if module is compiled"""
        return self._is_compiled