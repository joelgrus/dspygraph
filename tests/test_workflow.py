"""
Tests for Graph functionality
"""
import pytest
import dspy
from unittest.mock import Mock, patch
from typing import Dict, Any

from dspygraph import Graph, Node, START, END


class SimpleTestNode(Node):
    """Simple node for testing workflows"""
    
    def __init__(self, name: str, output_key: str = "output", output_value: str = "result"):
        self.output_key = output_key
        self.output_value = output_value
        super().__init__(name)
    
    def _create_module(self) -> dspy.Module:
        return Mock()  # Don't need real DSPy module for workflow tests
    
    def process(self, state: Dict[str, Any]) -> Dict[str, Any]:
        return {self.output_key: self.output_value}


class ConditionalTestNode(Node):
    """Node that reads state for conditional testing"""
    
    def _create_module(self) -> dspy.Module:
        return Mock()
    
    def process(self, state: Dict[str, Any]) -> Dict[str, Any]:
        # Echo the input for routing decisions
        return {"route": state.get("input", "default")}


class TestGraph:
    """Test suite for Graph"""
    
    def setup_method(self):
        """Setup for each test"""
        # Mock DSPy configuration
        with patch('dspy.configure'), patch('dspy.track_usage') as mock_track:
            mock_usage = Mock()
            mock_usage.get_total_tokens.return_value = {}
            mock_track.return_value.__enter__.return_value = mock_usage
            mock_track.return_value.__exit__.return_value = None
    
    def test_workflow_initialization(self):
        """Test basic workflow creation"""
        graph = Graph("test_workflow")
        
        assert graph.name == "test_workflow"
        assert graph.graph_id is not None
        assert len(graph.nodes) == 0
        assert len(graph.edges) == 0
        assert len(graph.start_nodes) == 0
        assert graph._execution_count == 0
    
    def test_add_node(self):
        """Test adding nodes to workflow"""
        graph = Graph("test")
        node1 = SimpleTestNode("node1")
        node2 = SimpleTestNode("node2")
        
        with patch('builtins.print'):  # Suppress output
            graph.add_node(node1)
            graph.add_node(node2)
        
        assert len(graph.nodes) == 2
        assert "node1" in graph.nodes
        assert "node2" in graph.nodes
        assert graph.nodes["node1"] == node1
    
    def test_add_duplicate_node(self):
        """Test adding duplicate node names"""
        graph = Graph("test")
        node1 = SimpleTestNode("duplicate")
        node2 = SimpleTestNode("duplicate")
        
        with patch('builtins.print'):
            graph.add_node(node1)
            
            with pytest.raises(ValueError, match="already exists"):
                graph.add_node(node2)
    
    def test_add_edge_basic(self):
        """Test adding basic edges"""
        graph = Graph("test")
        node1 = SimpleTestNode("node1")
        node2 = SimpleTestNode("node2")
        
        with patch('builtins.print'):
            graph.add_node(node1)
            graph.add_node(node2)
            graph.add_edge("node1", "node2")
        
        assert len(graph.edges) == 1
        from_node, to_node, condition = graph.edges[0]
        assert from_node == "node1"
        assert to_node == "node2"
        assert condition is None
    
    def test_add_edge_with_condition(self):
        """Test adding conditional edges"""
        graph = Graph("test")
        node1 = SimpleTestNode("node1")
        node2 = SimpleTestNode("node2")
        
        def test_condition(state):
            return state.get("test", False)
        
        with patch('builtins.print'):
            graph.add_node(node1)
            graph.add_node(node2)
            graph.add_edge("node1", "node2", condition=test_condition)
        
        assert len(graph.edges) == 1
        from_node, to_node, condition = graph.edges[0]
        assert condition is not None
        assert condition({"test": True}) == True
        assert condition({"test": False}) == False
    
    def test_add_edge_start(self):
        """Test adding edges from START"""
        graph = Graph("test")
        node1 = SimpleTestNode("node1")
        
        with patch('builtins.print'):
            graph.add_node(node1)
            graph.add_edge(START, "node1")
        
        assert "node1" in graph.start_nodes
        assert len(graph.edges) == 1
    
    def test_add_edge_end(self):
        """Test adding edges to END"""
        graph = Graph("test")
        node1 = SimpleTestNode("node1")
        
        with patch('builtins.print'):
            graph.add_node(node1)
            graph.add_edge("node1", END)
        
        assert len(graph.edges) == 1
        from_node, to_node, condition = graph.edges[0]
        assert to_node == END
    
    def test_add_edge_missing_nodes(self):
        """Test adding edges with missing nodes"""
        graph = Graph("test")
        
        with patch('builtins.print'):
            # Missing source node (not START)
            with pytest.raises(ValueError, match="Source node 'missing' not found"):
                graph.add_edge("missing", "also_missing")
            
            # Missing target node (not END)  
            node1 = SimpleTestNode("node1")
            graph.add_node(node1)
            with pytest.raises(ValueError, match="Target node 'missing' not found"):
                graph.add_edge("node1", "missing")
    
    def test_add_conditional_edges(self):
        """Test adding conditional edges"""
        graph = Graph("test")
        node1 = SimpleTestNode("node1")
        node2 = SimpleTestNode("node2")
        node3 = SimpleTestNode("node3")
        
        def router(state):
            return state.get("route", "default")
        
        with patch('builtins.print'):
            graph.add_node(node1)
            graph.add_node(node2)
            graph.add_node(node3)
            
            graph.add_conditional_edges(
                "node1",
                {"path1": "node2", "path2": "node3", "end": END},
                router
            )
        
        assert len(graph.edges) == 3
        
        # Test the conditions
        edge_conditions = [(from_node, to_node, condition) for from_node, to_node, condition in graph.edges]
        
        # All should be from node1
        for from_node, to_node, condition in edge_conditions:
            assert from_node == "node1"
            assert condition is not None
    
    @patch('dspy.track_usage')
    def test_simple_workflow_execution(self, mock_track_usage):
        """Test executing a simple linear workflow"""
        # Setup mock
        mock_usage = Mock()
        mock_usage.get_total_tokens.return_value = {}
        mock_track_usage.return_value.__enter__.return_value = mock_usage
        mock_track_usage.return_value.__exit__.return_value = None
        
        graph = Graph("test")
        node1 = SimpleTestNode("node1", "step1", "value1")
        node2 = SimpleTestNode("node2", "step2", "value2")
        
        with patch('builtins.print'):
            graph.add_node(node1)
            graph.add_node(node2)
            graph.add_edge(START, "node1")
            graph.add_edge("node1", "node2")
            graph.add_edge("node2", END)
        
        # Execute workflow
        result = graph.run(initial_input="test")
        
        # Verify results
        assert result["initial_input"] == "test"
        assert result["step1"] == "value1"
        assert result["step2"] == "value2"
        assert "_graph_metadata" in result
        
        metadata = result["_graph_metadata"]
        assert metadata["graph_name"] == "test"
        assert metadata["execution_order"] == ["node1", "node2"]
        assert metadata["success"] == True
    
    @patch('dspy.track_usage')
    def test_conditional_workflow_execution(self, mock_track_usage):
        """Test executing a workflow with conditional routing"""
        # Setup mock
        mock_usage = Mock()
        mock_usage.get_total_tokens.return_value = {}
        mock_track_usage.return_value.__enter__.return_value = mock_usage
        mock_track_usage.return_value.__exit__.return_value = None
        
        graph = Graph("test")
        classifier = ConditionalTestNode("classifier")
        path1_node = SimpleTestNode("path1", "result", "went_path1")
        path2_node = SimpleTestNode("path2", "result", "went_path2")
        
        def router(state):
            route = state.get("route", "default")
            if route == "option1":
                return "path1"
            elif route == "option2":
                return "path2"
            else:
                return "end"
        
        with patch('builtins.print'):
            graph.add_node(classifier)
            graph.add_node(path1_node)
            graph.add_node(path2_node)
            
            graph.add_edge(START, "classifier")
            graph.add_conditional_edges(
                "classifier",
                {"path1": "path1", "path2": "path2", "end": END},
                router
            )
            graph.add_edge("path1", END)
            graph.add_edge("path2", END)
        
        # Test path1
        result1 = graph.run(input="option1")
        assert result1["result"] == "went_path1"
        assert "path1" in result1["_graph_metadata"]["execution_order"]
        assert "path2" not in result1["_graph_metadata"]["execution_order"]
        
        # Test path2
        result2 = graph.run(input="option2")
        assert result2["result"] == "went_path2"
        assert "path2" in result2["_graph_metadata"]["execution_order"]
        assert "path1" not in result2["_graph_metadata"]["execution_order"]
        
        # Test early termination
        result3 = graph.run(input="unknown")
        assert "result" not in result3  # No path taken
        assert result3["_graph_metadata"]["execution_order"] == ["classifier"]
    
    def test_workflow_validation_no_nodes(self):
        """Test workflow validation with no nodes"""
        graph = Graph("empty")
        
        with pytest.raises(ValueError, match="has no nodes"):
            graph.run()
    
    def test_workflow_repr(self):
        """Test workflow string representation"""
        graph = Graph("test")
        node1 = SimpleTestNode("node1")
        
        with patch('builtins.print'):
            graph.add_node(node1)
            graph.add_edge(START, "node1")
        
        repr_str = repr(graph)
        assert "test" in repr_str
        assert "nodes=1" in repr_str
        assert "edges=1" in repr_str
    
    def test_workflow_visualize(self):
        """Test workflow visualization"""
        graph = Graph("test")
        node1 = SimpleTestNode("node1")
        node2 = SimpleTestNode("node2")
        
        with patch('builtins.print'):
            graph.add_node(node1)
            graph.add_node(node2)
            graph.add_edge(START, "node1")
            graph.add_edge("node1", "node2")
            graph.add_edge("node2", END)
        
        viz = graph.visualize()
        
        assert "DSPy Graph: test" in viz
        assert "Nodes: 2" in viz
        assert "Edges: 3" in viz
        assert "node1 (START)" in viz
        assert "__START__ -> node1" in viz
        assert "node2 -> __END__" in viz
    
    @patch('dspy.track_usage')
    def test_execution_count_increments(self, mock_track_usage):
        """Test that execution count increments"""
        # Setup mock
        mock_usage = Mock()
        mock_usage.get_total_tokens.return_value = {}
        mock_track_usage.return_value.__enter__.return_value = mock_usage
        mock_track_usage.return_value.__exit__.return_value = None
        
        graph = Graph("test")
        node1 = SimpleTestNode("node1")
        
        with patch('builtins.print'):
            graph.add_node(node1)
            graph.add_edge(START, "node1")
            graph.add_edge("node1", END)
        
        assert graph._execution_count == 0
        
        graph.run()
        assert graph._execution_count == 1
        
        graph.run()
        assert graph._execution_count == 2