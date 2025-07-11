"""
Tests for error conditions and edge cases
"""
import pytest
import dspy
from unittest.mock import Mock, patch
from typing import Dict, Any

from dspygraph import Graph, Node, START, END


class FailingNode(Node):
    """Node that fails during execution for testing"""
    
    def __init__(self, name: str, fail_on_process: bool = False, fail_on_create: bool = False):
        self.fail_on_process = fail_on_process
        self.fail_on_create = fail_on_create
        super().__init__(name)
    
    def _create_module(self) -> dspy.Module:
        if self.fail_on_create:
            raise RuntimeError("Failed to create module")
        return Mock()
    
    def process(self, state: Dict[str, Any]) -> Dict[str, Any]:
        if self.fail_on_process:
            raise RuntimeError("Node processing failed")
        return {"success": True}


class DeadlockNode(Node):
    """Node for testing deadlock conditions"""
    
    def _create_module(self) -> dspy.Module:
        return Mock()
    
    def process(self, state: Dict[str, Any]) -> Dict[str, Any]:
        return {"processed": True}


class TestErrorConditions:
    """Test suite for error conditions and edge cases"""
    
    def setup_method(self):
        """Setup for each test"""
        with patch('dspy.configure'), patch('dspy.track_usage') as mock_track:
            mock_usage = Mock()
            mock_usage.get_total_tokens.return_value = {}
            mock_track.return_value.__enter__.return_value = mock_usage
            mock_track.return_value.__exit__.return_value = None
    
    @patch('dspy.track_usage')
    def test_node_execution_failure(self, mock_track_usage):
        """Test handling of node execution failures"""
        mock_usage = Mock()
        mock_usage.get_total_tokens.return_value = {}
        mock_track_usage.return_value.__enter__.return_value = mock_usage
        mock_track_usage.return_value.__exit__.return_value = None
        
        graph = Graph("test")
        failing_node = FailingNode("failing", fail_on_process=True)
        
        with patch('builtins.print'):
            graph.add_node(failing_node)
            graph.add_edge(START, "failing")
            graph.add_edge("failing", END)
        
        with pytest.raises(RuntimeError, match="Node processing failed"):
            graph.run(input="test")
    
    def test_node_creation_failure(self):
        """Test handling of node creation failures"""
        with pytest.raises(RuntimeError, match="Failed to create module"):
            FailingNode("failing", fail_on_create=True)
    
    def test_workflow_no_start_nodes(self):
        """Test workflow with no start nodes"""
        graph = Graph("test")
        node1 = DeadlockNode("node1")
        node2 = DeadlockNode("node2")
        
        with patch('builtins.print'):
            graph.add_node(node1)
            graph.add_node(node2)
            # No START edges and no legacy is_start
            graph.add_edge("node1", "node2")
        
        # Should fail validation since there are edges but no start nodes
        with patch('dspy.track_usage') as mock_track:
            mock_usage = Mock()
            mock_usage.get_total_tokens.return_value = {}
            mock_track.return_value.__enter__.return_value = mock_usage
            mock_track.return_value.__exit__.return_value = None
            
            with pytest.raises(ValueError, match="has edges but no start nodes"):
                graph.run(input="test")
    
    def test_workflow_empty_run(self):
        """Test running empty workflow"""
        graph = Graph("empty")
        
        with pytest.raises(ValueError, match="has no nodes"):
            graph.run(input="test")
    
    def test_missing_required_state(self):
        """Test node that expects missing state"""
        class RequiringNode(Node):
            def _create_module(self):
                return Mock()
            
            def process(self, state):
                # Access required key that might not exist
                required_value = state["required_key"]  # This could raise KeyError
                return {"result": required_value}
        
        graph = Graph("test")
        node = RequiringNode("requiring")
        
        with patch('builtins.print'):
            graph.add_node(node)
            graph.add_edge(START, "requiring")
        
        with patch('dspy.track_usage') as mock_track:
            mock_usage = Mock()
            mock_usage.get_total_tokens.return_value = {}
            mock_track.return_value.__enter__.return_value = mock_usage
            mock_track.return_value.__exit__.return_value = None
            
            with pytest.raises(KeyError, match="required_key"):
                graph.run(input="test")  # No required_key provided
    
    def test_node_returns_invalid_output(self):
        """Test node that returns invalid output"""
        class InvalidOutputNode(Node):
            def _create_module(self):
                return Mock()
            
            def process(self, state):
                return "not a dict"  # Should return dict
        
        graph = Graph("test")
        node = InvalidOutputNode("invalid")
        
        with patch('builtins.print'):
            graph.add_node(node)
            graph.add_edge(START, "invalid")
        
        with patch('dspy.track_usage') as mock_track:
            mock_usage = Mock()
            mock_usage.get_total_tokens.return_value = {}
            mock_track.return_value.__enter__.return_value = mock_usage
            mock_track.return_value.__exit__.return_value = None
            
            # This should cause an error when trying to update state
            with pytest.raises(AttributeError):
                graph.run(input="test")
    
    def test_infinite_loop_protection(self):
        """Test protection against infinite loops in cycles"""
        graph = Graph("test")
        node1 = DeadlockNode("node1")
        node2 = DeadlockNode("node2")
        node3 = DeadlockNode("node3")
        
        with patch('builtins.print'):
            graph.add_node(node1)
            graph.add_node(node2)
            graph.add_node(node3)
            
            # Create circular dependency - now allowed!
            graph.add_edge(START, "node1")
            graph.add_edge("node1", "node2")
            graph.add_edge("node2", "node3")
            graph.add_edge("node3", "node1")  # Creates cycle
        
        with patch('dspy.track_usage') as mock_track:
            mock_usage = Mock()
            mock_usage.get_total_tokens.return_value = {}
            mock_track.return_value.__enter__.return_value = mock_usage
            mock_track.return_value.__exit__.return_value = None
            
            # Should complete but hit max iterations protection
            result = graph.run(input="test", max_iterations=5, max_node_executions=3)
            
            # Verify loop protection worked
            metadata = result["_graph_metadata"]
            assert metadata["total_iterations"] >= 5  # Should hit max iterations (may be off by one)
            assert "stopped_reason" in metadata  # Should indicate why it stopped
            assert metadata["stopped_reason"].startswith("max_iterations_reached")
    
    def test_conditional_edge_with_invalid_router(self):
        """Test conditional edges with router that raises exceptions"""
        def failing_router(state):
            raise ValueError("Router failed")
        
        graph = Graph("test")
        node1 = DeadlockNode("node1")
        node2 = DeadlockNode("node2")
        
        with patch('builtins.print'):
            graph.add_node(node1)
            graph.add_node(node2)
            
            graph.add_edge(START, "node1")
            graph.add_conditional_edges(
                "node1",
                {"success": "node2"},
                failing_router
            )
        
        with patch('dspy.track_usage') as mock_track:
            mock_usage = Mock()
            mock_usage.get_total_tokens.return_value = {}
            mock_track.return_value.__enter__.return_value = mock_usage
            mock_track.return_value.__exit__.return_value = None
            
            with pytest.raises(ValueError, match="Router failed"):
                graph.run(input="test")
    
    def test_edge_to_nonexistent_conditional_target(self):
        """Test conditional routing to non-existent target"""
        def router_to_missing(state):
            return "missing_target"
        
        graph = Graph("test")
        node1 = DeadlockNode("node1")
        
        with patch('builtins.print'):
            graph.add_node(node1)
            graph.add_edge(START, "node1")
            
            # This should fail since we're trying to add edge to nonexistent node
            with pytest.raises(ValueError, match="Target node 'nonexistent' not found"):
                graph.add_conditional_edges(
                    "node1",
                    {"missing_target": "nonexistent"},  # nonexistent node
                    router_to_missing
                )
    
    def test_dspy_configuration_missing(self):
        """Test behavior when DSPy is not configured"""
        # This is harder to test since we mock DSPy in setup
        # But we can test that our nodes handle DSPy errors gracefully
        
        class DSPyErrorNode(Node):
            def _create_module(self):
                # Simulate DSPy configuration error
                raise RuntimeError("DSPy not configured")
            
            def process(self, state):
                return {"test": "value"}
        
        with pytest.raises(RuntimeError, match="DSPy not configured"):
            DSPyErrorNode("error_node")
    
    def test_compilation_with_invalid_data(self):
        """Test compilation with invalid training data"""
        node = FailingNode("test")
        
        # Mock compiler that fails
        mock_compiler = Mock()
        mock_compiler.compile.side_effect = ValueError("Invalid training data")
        
        with pytest.raises(ValueError, match="Invalid training data"):
            node.compile(mock_compiler, [], compile_path="test.json")
    
    def test_load_compiled_with_corrupt_file(self):
        """Test loading corrupt compiled file"""
        node = FailingNode("test")
        
        # Mock module load that fails
        node.module.load = Mock()
        node.module.load.side_effect = Exception("Corrupt file")
        
        with patch('os.path.exists', return_value=True):
            with pytest.raises(Exception, match="Corrupt file"):
                node.load_compiled("corrupt.json")
    
    @patch('dspy.track_usage')
    def test_workflow_execution_metadata_on_failure(self, mock_track_usage):
        """Test that workflow metadata is preserved on failure"""
        mock_usage = Mock()
        mock_usage.get_total_tokens.return_value = {}
        mock_track_usage.return_value.__enter__.return_value = mock_usage
        mock_track_usage.return_value.__exit__.return_value = None
        
        graph = Graph("test")
        good_node = DeadlockNode("good")
        bad_node = FailingNode("bad", fail_on_process=True)
        
        with patch('builtins.print'):
            graph.add_node(good_node)
            graph.add_node(bad_node)
            graph.add_edge(START, "good")
            graph.add_edge("good", "bad")
        
        try:
            graph.run(input="test")
            assert False, "Should have raised exception"
        except RuntimeError:
            # Expected failure
            pass
        
        # Graph should have incremented execution count even on failure
        assert graph._execution_count == 1
    
    def test_state_key_collision(self):
        """Test handling of state key collisions"""
        class CollidingNode(Node):
            def _create_module(self):
                return Mock()
            
            def process(self, state):
                # Try to overwrite metadata
                return {"_graph_metadata": "hacked"}
        
        graph = Graph("test")
        node = CollidingNode("colliding")
        
        with patch('builtins.print'):
            graph.add_node(node)
            graph.add_edge(START, "colliding")
            graph.add_edge("colliding", END)
        
        with patch('dspy.track_usage') as mock_track:
            mock_usage = Mock()
            mock_usage.get_total_tokens.return_value = {}
            mock_track.return_value.__enter__.return_value = mock_usage
            mock_track.return_value.__exit__.return_value = None
            
            result = graph.run(input="test")
            
            # Graph metadata should be protected from node outputs
            assert isinstance(result["_graph_metadata"], dict)
            assert result["_graph_metadata"]["graph_name"] == "test"
            # The node's attempted override should be ignored