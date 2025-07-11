"""
DSPy Graph execution engine
"""
import time
import uuid
from typing import Dict, Any, List, Optional, Callable, Set, Union
from collections import defaultdict, deque
import dspy

from .node import Node

# Module-level constants for graph control
START = "__START__"
END = "__END__"


class Graph:
    """
    A graph execution engine for DSPy nodes with arbitrary topology support
    """
    
    def __init__(self, name: str = "Graph"):
        self.name = name
        self.graph_id = str(uuid.uuid4())
        self.nodes: Dict[str, Node] = {}
        self.edges: List[tuple] = []
        self.start_nodes: Set[str] = set()
        self._execution_count = 0
        
    def add_node(self, node: Node) -> 'Graph':
        """
        Add a node to the graph
        
        Args:
            node: DSPyNode instance to add
            
        Returns:
            Self for method chaining
        """
        if node.name in self.nodes:
            raise ValueError(f"Node '{node.name}' already exists in graph")
            
        self.nodes[node.name] = node
        print(f"[{self.name}] Added node: {node.name}")
        return self
    
    def add_edge(self, from_node: str, to_node: str, 
                 condition: Optional[Callable[[Dict[str, Any]], bool]] = None) -> 'Graph':
        """
        Add an edge between nodes
        
        Args:
            from_node: Source node name (or START)
            to_node: Target node name (or END)
            condition: Optional condition function to evaluate before following edge
            
        Returns:
            Self for method chaining
        """
        # Validate nodes exist (unless START/END)
        if from_node != START and from_node not in self.nodes:
            raise ValueError(f"Source node '{from_node}' not found")
        if to_node != END and to_node not in self.nodes:
            raise ValueError(f"Target node '{to_node}' not found")
            
        # Track start nodes
        if from_node == START:
            self.start_nodes.add(to_node)
            
        self.edges.append((from_node, to_node, condition))
        print(f"[{self.name}] Added edge: {from_node} -> {to_node}")
        return self
    
    def add_conditional_edges(self, from_node: str, 
                            conditions: Dict[str, str],
                            condition_fn: Callable[[Dict[str, Any]], str]) -> 'Graph':
        """
        Add conditional edges based on state evaluation
        
        Args:
            from_node: Source node name
            conditions: Mapping of condition results to target nodes (or END)
            condition_fn: Function that evaluates state and returns condition key
            
        Returns:
            Self for method chaining
        """
        for condition_key, to_node in conditions.items():
            condition = lambda state, key=condition_key: condition_fn(state) == key
            self.add_edge(from_node, to_node, condition)
        return self
    
    def _get_ready_nodes(self, completed: Set[str], state: Dict[str, Any]) -> List[str]:
        """Get nodes that are ready to execute (all dependencies met and conditions satisfied)"""
        ready = []
        
        for node_name in self.nodes:
            if node_name in completed:
                continue
                
            # Check if this node has any incoming edges (including from START)
            incoming_edges = [(from_node, condition) for from_node, to_node, condition in self.edges if to_node == node_name]
            
            if not incoming_edges:
                # Legacy: This is a start node (no incoming edges) - keep for backwards compatibility
                if node_name in self.start_nodes:
                    ready.append(node_name)
            else:
                # Check if any incoming edge is satisfied
                for from_node, condition in incoming_edges:
                    # Handle START specially - it's always "completed"
                    if from_node == START or from_node in completed:
                        if condition is None or condition(state):
                            ready.append(node_name)
                            break  # Only need one satisfied edge
                
        return ready
    
    def _validate_graph(self) -> None:
        """Validate the graph for common issues"""
        if not self.nodes:
            raise ValueError("Graph has no nodes")
            
        if not self.start_nodes and self.edges:
            raise ValueError("Graph has edges but no start nodes defined")
            
        # Check for cycles (simple detection)
        visited = set()
        rec_stack = set()
        
        def has_cycle(node: str) -> bool:
            visited.add(node)
            rec_stack.add(node)
            
            for from_node, to_node, _ in self.edges:
                if from_node == node:
                    if to_node not in visited:
                        if has_cycle(to_node):
                            return True
                    elif to_node in rec_stack:
                        return True
            
            rec_stack.remove(node)
            return False
        
        for node_name in self.nodes:
            if node_name not in visited:
                if has_cycle(node_name):
                    raise ValueError("Graph contains cycles")
    
    def run(self, **initial_state) -> Dict[str, Any]:
        """
        Execute the graph
        
        Args:
            **initial_state: Initial state values
            
        Returns:
            Final graph state
        """
        execution_id = str(uuid.uuid4())
        self._execution_count += 1
        
        print(f"\n{'='*60}")
        print(f"[{self.name}] Starting execution {self._execution_count}")
        print(f"[{self.name}] Execution ID: {execution_id}")
        print(f"[{self.name}] Initial state: {list(initial_state.keys())}")
        print(f"{'='*60}")
        
        start_time = time.time()
        
        # Initialize state outside try block
        state = dict(initial_state)
        
        try:
            # Validate graph structure
            self._validate_graph()
            
            # Initialize tracking
            completed = set()
            node_execution_order = []
            total_usage = defaultdict(int)
            
            # Track graph metadata
            state["_graph_metadata"] = {
                "graph_name": self.name,
                "graph_id": self.graph_id,
                "execution_id": execution_id,
                "execution_count": self._execution_count,
                "start_time": start_time
            }
            
            # Main execution loop
            while True:
                ready_nodes = self._get_ready_nodes(completed, state)
                
                if not ready_nodes:
                    # No more nodes ready - check if this is expected
                    remaining = set(self.nodes.keys()) - completed
                    if remaining:
                        print(f"[{self.name}] Workflow complete. Skipped nodes: {remaining}")
                    break
                
                # Check if any ready node should terminate early
                should_terminate = self._check_for_termination(completed, state)
                if should_terminate:
                    print(f"[{self.name}] Workflow terminated early via END")
                    break
                
                print(f"\n[{self.name}] Ready to execute: {ready_nodes}")
                
                # Execute ready nodes (could be parallelized here)
                for node_name in ready_nodes:
                    node = self.nodes[node_name]
                    
                    try:
                        # Execute node with full observability
                        with dspy.track_usage() as usage:
                            node_outputs = node(state)
                        
                        # Update state with node outputs, protecting metadata
                        for key, value in node_outputs.items():
                            if key != "_graph_metadata":  # Protect graph metadata
                                state[key] = value
                        
                        # Track execution
                        completed.add(node_name)
                        node_execution_order.append(node_name)
                        
                        # Accumulate usage stats
                        node_usage = usage.get_total_tokens()
                        for key, value in node_usage.items():
                            total_usage[key] += value
                            
                    except Exception as e:
                        print(f"[{self.name}] Node '{node_name}' failed: {e}")
                        raise
            
            execution_time = time.time() - start_time
            
            # Add final metadata
            state["_graph_metadata"].update({
                "execution_order": node_execution_order,
                "execution_time": execution_time,
                "total_usage": dict(total_usage),
                "nodes_executed": len(completed),
                "success": True
            })
            
            print(f"\n{'='*60}")
            print(f"[{self.name}] Execution complete in {execution_time:.3f}s")
            print(f"[{self.name}] Nodes executed: {' -> '.join(node_execution_order)}")
            print(f"[{self.name}] Total usage: {dict(total_usage)}")
            print(f"{'='*60}\n")
            
            return state
            
        except Exception as e:
            execution_time = time.time() - start_time
            print(f"\n[{self.name}] Execution failed after {execution_time:.3f}s: {e}")
            
            # Add failure metadata
            if "_graph_metadata" in state:
                state["_graph_metadata"].update({
                    "execution_time": execution_time,
                    "success": False,
                    "error": str(e)
                })
            
            raise
    
    def visualize(self) -> str:
        """Generate a simple text visualization of the graph"""
        lines = [f"DSPy Graph: {self.name}"]
        lines.append(f"Nodes: {len(self.nodes)}")
        lines.append(f"Edges: {len(self.edges)}")
        lines.append("")
        
        lines.append("Nodes:")
        for name, node in self.nodes.items():
            start_indicator = " (START)" if name in self.start_nodes else ""
            compile_indicator = " [COMPILED]" if node.compiled else ""
            lines.append(f"  {name}{start_indicator}{compile_indicator}")
        
        lines.append("")
        lines.append("Edges:")
        for from_node, to_node, condition in self.edges:
            condition_indicator = " [CONDITIONAL]" if condition else ""
            lines.append(f"  {from_node} -> {to_node}{condition_indicator}")
            
        return "\n".join(lines)
    
    def _check_for_termination(self, completed: Set[str], state: Dict[str, Any]) -> bool:
        """Check if any completed node routes to END"""
        for from_node, to_node, condition in self.edges:
            if to_node == END and from_node in completed:
                if condition is None or condition(state):
                    return True
        return False
    
    def __repr__(self) -> str:
        return f"Graph(name='{self.name}', nodes={len(self.nodes)}, edges={len(self.edges)})"