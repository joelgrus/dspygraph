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
    
    def _get_ready_nodes(self, completed: Set[str], state: Dict[str, Any], executed_this_iteration: Set[str] = None) -> List[str]:
        """Get nodes that are ready to execute (all dependencies met and conditions satisfied)"""
        if executed_this_iteration is None:
            executed_this_iteration = set()
            
        ready = []
        
        for node_name in self.nodes:
            # Don't execute a node twice in the same iteration
            if node_name in executed_this_iteration:
                continue
                
            # Check if this node has any incoming edges (including from START)
            incoming_edges = [(from_node, condition) for from_node, to_node, condition in self.edges if to_node == node_name]
            
            if not incoming_edges:
                # Legacy: This is a start node (no incoming edges) - keep for backwards compatibility
                if node_name in self.start_nodes and node_name not in completed:
                    ready.append(node_name)
            else:
                # Check if any incoming edge is satisfied
                node_ready = False
                for from_node, condition in incoming_edges:
                    # Handle START specially - it's always "completed"
                    if from_node == START:
                        if condition is None or condition(state):
                            # Only run START edges if node hasn't been completed yet
                            if node_name not in completed:
                                node_ready = True
                                break
                    elif from_node in completed:
                        if condition is None or condition(state):
                            # If this node hasn't been executed yet, always allow first execution
                            if node_name not in completed:
                                node_ready = True
                                break
                            # For re-execution (cycles): allow if there are outgoing edges
                            # This enables cycles but relies on iteration limits for protection
                            else:
                                outgoing_edges = [edge for edge in self.edges if edge[0] == node_name]
                                if outgoing_edges:
                                    node_ready = True
                                    break
                
                if node_ready:
                    ready.append(node_name)
                
        return ready
    
    def _validate_graph(self) -> None:
        """Validate the graph for common issues"""
        if not self.nodes:
            raise ValueError("Graph has no nodes")
            
        if not self.start_nodes and self.edges:
            raise ValueError("Graph has edges but no start nodes defined")
    
    def run(self, max_iterations: int = 100, max_node_executions: int = 10, **initial_state) -> Dict[str, Any]:
        """
        Execute the graph
        
        Args:
            max_iterations: Maximum total iterations before stopping (default: 100)
            max_node_executions: Maximum executions per node before warning (default: 10)
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
            node_execution_counts = defaultdict(int)  # Track executions per node
            iteration_count = 0  # Track total iterations
            nodes_executed_this_iteration = set()  # Track nodes executed in current iteration
            
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
                iteration_count += 1
                
                # Check max iterations
                if iteration_count > max_iterations:
                    print(f"\n[{self.name}] WARNING: Reached maximum iterations ({max_iterations})")
                    print(f"[{self.name}] Stopping execution to prevent infinite loop")
                    state["_graph_metadata"]["stopped_reason"] = f"max_iterations_reached ({max_iterations})"
                    break
                
                # Reset per-iteration tracking
                nodes_executed_this_iteration = set()
                
                ready_nodes = self._get_ready_nodes(completed, state, nodes_executed_this_iteration)
                
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
                    
                    # Track executions
                    node_execution_counts[node_name] += 1
                    nodes_executed_this_iteration.add(node_name)
                    
                    if node_execution_counts[node_name] > max_node_executions:
                        print(f"\n[{self.name}] WARNING: Node '{node_name}' has been executed {node_execution_counts[node_name]} times")
                        print(f"[{self.name}] This may indicate an infinite loop in your graph logic")
                    
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
                "success": True,
                "total_iterations": iteration_count,
                "node_execution_counts": dict(node_execution_counts)
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