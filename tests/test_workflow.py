"""
Tests for Workflow functionality
"""
import pytest
import dspy
from unittest.mock import Mock, patch
from typing import Dict, Any

from dspygraph import Workflow, Node, START, END


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


class TestWorkflow:
    """Test suite for Workflow"""
    
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
        workflow = Workflow("test_workflow")
        
        assert workflow.name == "test_workflow"
        assert workflow.workflow_id is not None
        assert len(workflow.nodes) == 0
        assert len(workflow.edges) == 0
        assert len(workflow.start_nodes) == 0
        assert workflow._execution_count == 0
    
    def test_add_node(self):
        """Test adding nodes to workflow"""
        workflow = Workflow("test")
        node1 = SimpleTestNode("node1")
        node2 = SimpleTestNode("node2")
        
        with patch('builtins.print'):  # Suppress output
            workflow.add_node(node1)
            workflow.add_node(node2)
        
        assert len(workflow.nodes) == 2
        assert "node1" in workflow.nodes
        assert "node2" in workflow.nodes
        assert workflow.nodes["node1"] == node1
    
    def test_add_duplicate_node(self):
        """Test adding duplicate node names"""
        workflow = Workflow("test")
        node1 = SimpleTestNode("duplicate")
        node2 = SimpleTestNode("duplicate")
        
        with patch('builtins.print'):
            workflow.add_node(node1)
            
            with pytest.raises(ValueError, match="already exists"):
                workflow.add_node(node2)
    
    def test_add_edge_basic(self):
        """Test adding basic edges"""
        workflow = Workflow("test")
        node1 = SimpleTestNode("node1")
        node2 = SimpleTestNode("node2")
        
        with patch('builtins.print'):
            workflow.add_node(node1)
            workflow.add_node(node2)
            workflow.add_edge("node1", "node2")
        
        assert len(workflow.edges) == 1
        from_node, to_node, condition = workflow.edges[0]
        assert from_node == "node1"
        assert to_node == "node2"
        assert condition is None
    
    def test_add_edge_with_condition(self):
        """Test adding conditional edges"""
        workflow = Workflow("test")
        node1 = SimpleTestNode("node1")
        node2 = SimpleTestNode("node2")
        
        def test_condition(state):
            return state.get("test", False)
        
        with patch('builtins.print'):
            workflow.add_node(node1)
            workflow.add_node(node2)
            workflow.add_edge("node1", "node2", condition=test_condition)
        
        assert len(workflow.edges) == 1
        from_node, to_node, condition = workflow.edges[0]
        assert condition is not None
        assert condition({"test": True}) == True
        assert condition({"test": False}) == False
    
    def test_add_edge_start(self):
        """Test adding edges from START"""
        workflow = Workflow("test")
        node1 = SimpleTestNode("node1")
        
        with patch('builtins.print'):
            workflow.add_node(node1)
            workflow.add_edge(START, "node1")
        
        assert "node1" in workflow.start_nodes
        assert len(workflow.edges) == 1
    
    def test_add_edge_end(self):
        """Test adding edges to END"""
        workflow = Workflow("test")
        node1 = SimpleTestNode("node1")
        
        with patch('builtins.print'):
            workflow.add_node(node1)
            workflow.add_edge("node1", END)
        
        assert len(workflow.edges) == 1
        from_node, to_node, condition = workflow.edges[0]
        assert to_node == END
    
    def test_add_edge_missing_nodes(self):
        """Test adding edges with missing nodes"""
        workflow = Workflow("test")
        
        with patch('builtins.print'):
            # Missing source node (not START)
            with pytest.raises(ValueError, match="Source node 'missing' not found"):
                workflow.add_edge("missing", "also_missing")
            
            # Missing target node (not END)  
            node1 = SimpleTestNode("node1")
            workflow.add_node(node1)
            with pytest.raises(ValueError, match="Target node 'missing' not found"):
                workflow.add_edge("node1", "missing")
    
    def test_add_conditional_edges(self):
        """Test adding conditional edges"""
        workflow = Workflow("test")
        node1 = SimpleTestNode("node1")
        node2 = SimpleTestNode("node2")
        node3 = SimpleTestNode("node3")
        
        def router(state):
            return state.get("route", "default")
        
        with patch('builtins.print'):
            workflow.add_node(node1)
            workflow.add_node(node2)
            workflow.add_node(node3)
            
            workflow.add_conditional_edges(
                "node1",
                {"path1": "node2", "path2": "node3", "end": END},
                router
            )
        
        assert len(workflow.edges) == 3
        
        # Test the conditions
        edge_conditions = [(from_node, to_node, condition) for from_node, to_node, condition in workflow.edges]
        
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
        
        workflow = Workflow("test")
        node1 = SimpleTestNode("node1", "step1", "value1")
        node2 = SimpleTestNode("node2", "step2", "value2")
        
        with patch('builtins.print'):
            workflow.add_node(node1)
            workflow.add_node(node2)
            workflow.add_edge(START, "node1")
            workflow.add_edge("node1", "node2")
            workflow.add_edge("node2", END)
        
        # Execute workflow
        result = workflow.run(initial_input="test")
        
        # Verify results
        assert result["initial_input"] == "test"
        assert result["step1"] == "value1"
        assert result["step2"] == "value2"
        assert "_workflow_metadata" in result
        
        metadata = result["_workflow_metadata"]
        assert metadata["workflow_name"] == "test"
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
        
        workflow = Workflow("test")
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
            workflow.add_node(classifier)
            workflow.add_node(path1_node)
            workflow.add_node(path2_node)
            
            workflow.add_edge(START, "classifier")
            workflow.add_conditional_edges(
                "classifier",
                {"path1": "path1", "path2": "path2", "end": END},
                router
            )
            workflow.add_edge("path1", END)
            workflow.add_edge("path2", END)
        
        # Test path1
        result1 = workflow.run(input="option1")
        assert result1["result"] == "went_path1"
        assert "path1" in result1["_workflow_metadata"]["execution_order"]
        assert "path2" not in result1["_workflow_metadata"]["execution_order"]
        
        # Test path2
        result2 = workflow.run(input="option2")
        assert result2["result"] == "went_path2"
        assert "path2" in result2["_workflow_metadata"]["execution_order"]
        assert "path1" not in result2["_workflow_metadata"]["execution_order"]
        
        # Test early termination
        result3 = workflow.run(input="unknown")
        assert "result" not in result3  # No path taken
        assert result3["_workflow_metadata"]["execution_order"] == ["classifier"]
    
    def test_workflow_validation_no_nodes(self):
        """Test workflow validation with no nodes"""
        workflow = Workflow("empty")
        
        with pytest.raises(ValueError, match="has no nodes"):
            workflow.run()
    
    def test_workflow_repr(self):
        """Test workflow string representation"""
        workflow = Workflow("test")
        node1 = SimpleTestNode("node1")
        
        with patch('builtins.print'):
            workflow.add_node(node1)
            workflow.add_edge(START, "node1")
        
        repr_str = repr(workflow)
        assert "test" in repr_str
        assert "nodes=1" in repr_str
        assert "edges=1" in repr_str
    
    def test_workflow_visualize(self):
        """Test workflow visualization"""
        workflow = Workflow("test")
        node1 = SimpleTestNode("node1")
        node2 = SimpleTestNode("node2")
        
        with patch('builtins.print'):
            workflow.add_node(node1)
            workflow.add_node(node2)
            workflow.add_edge(START, "node1")
            workflow.add_edge("node1", "node2")
            workflow.add_edge("node2", END)
        
        viz = workflow.visualize()
        
        assert "DSPy Workflow: test" in viz
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
        
        workflow = Workflow("test")
        node1 = SimpleTestNode("node1")
        
        with patch('builtins.print'):
            workflow.add_node(node1)
            workflow.add_edge(START, "node1")
            workflow.add_edge("node1", END)
        
        assert workflow._execution_count == 0
        
        workflow.run()
        assert workflow._execution_count == 1
        
        workflow.run()
        assert workflow._execution_count == 2