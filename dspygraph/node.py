"""
DSPy Node abstraction with built-in observability
"""
import time
import uuid
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Callable
import dspy
from dspy.teleprompt import Teleprompter


class Node(ABC):
    """
    A DSPy node that wraps a DSPy module with observability and compilation support
    """
    
    def __init__(self, name: str):
        self.name = name
        self.node_id = str(uuid.uuid4())
        self.module = self._create_module()
        self.compiled = False
        self._execution_count = 0
        
    @abstractmethod
    def _create_module(self) -> dspy.Module:
        """Create the DSPy module for this node"""
        pass
    
    @abstractmethod
    def process(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Process the state and return updates"""
        pass
    
    def compile(self, compiler: Teleprompter, trainset: List[dspy.Example], 
                compile_path: Optional[str] = None) -> None:
        """
        Compile this node's module
        
        Args:
            compiler: DSPy teleprompter instance
            trainset: Training data for compilation
            compile_path: Optional path to save compiled model
        """
        print(f"[{self.name}] Starting compilation...")
        
        with dspy.track_usage() as usage:
            compiled_module = compiler.compile(self.module, trainset=trainset)
            self.module = compiled_module
            self.compiled = True
            
        print(f"[{self.name}] Compilation complete. Usage: {usage.get_total_tokens()}")
        
        if compile_path:
            self.save_compiled(compile_path)
    
    def load_compiled(self, path: str) -> None:
        """Load a compiled module from file"""
        try:
            self.module.load(path)
            self.compiled = True
            print(f"[{self.name}] Loaded compiled module from {path}")
        except Exception as e:
            print(f"[{self.name}] Failed to load compiled module: {e}")
            raise
    
    def save_compiled(self, path: str) -> None:
        """Save the compiled module to file"""
        self.module.save(path)
        print(f"[{self.name}] Saved compiled module to {path}")
    
    def __call__(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute this node with full observability
        
        Args:
            state: Current workflow state
            
        Returns:
            State updates from this node's execution
        """
        execution_id = str(uuid.uuid4())
        self._execution_count += 1
        
        print(f"[{self.name}] Starting execution {self._execution_count} (id: {execution_id[:8]})")
        
        # Execute node processing with tracking
        start_time = time.time()
        try:
            with dspy.track_usage() as usage:
                outputs = self.process(state)
            
            execution_time = time.time() - start_time
            usage_stats = usage.get_total_tokens()
            
            print(f"[{self.name}] Execution complete in {execution_time:.3f}s")
            print(f"[{self.name}] Token usage: {usage_stats}")
            print(f"[{self.name}] Outputs: {list(outputs.keys())}")
            
            # Add metadata to outputs
            outputs["_node_metadata"] = {
                "node_name": self.name,
                "node_id": self.node_id,
                "execution_id": execution_id,
                "execution_count": self._execution_count,
                "execution_time": execution_time,
                "compiled": self.compiled,
                "usage": usage_stats
            }
            
            return outputs
            
        except Exception as e:
            execution_time = time.time() - start_time
            print(f"[{self.name}] Execution failed after {execution_time:.3f}s: {e}")
            raise
    
    def __repr__(self) -> str:
        return f"Node(name='{self.name}', compiled={self.compiled}, executions={self._execution_count})"


    def ensure_compiled(self, compile_path: Optional[str] = None) -> None:
        """Ensure this node is compiled, loading from file if available"""
        if self.compiled:
            return
            
        if compile_path:
            try:
                self.load_compiled(compile_path)
                return
            except:
                pass
        
        raise RuntimeError(f"Node '{self.name}' requires compilation but no compiled module found")