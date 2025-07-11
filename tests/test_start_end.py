"""
Tests for START/END behavior and edge cases
"""
import pytest
import dspy
from unittest.mock import Mock, patch
from typing import Dict, Any

from dspygraph import Graph, Node, START, END


class MockNode(Node):
    """Mock node for testing START/END behavior"""
    
    def __init__(self, name: str, output_data: Dict[str, Any] = None):
        self.output_data = output_data or {}
        super().__init__(name)
    
    def _create_module(self) -> dspy.Module:
        return Mock()
    
    def process(self, state: Dict[str, Any]) -> Dict[str, Any]:
        return self.output_data.copy()


class TestStartEndBehavior:
    """Test suite for START/END behavior"""
    
    def setup_method(self):
        """Setup for each test"""
        with patch('dspy.configure'), patch('dspy.track_usage') as mock_track:
            mock_usage = Mock()
            mock_usage.get_total_tokens.return_value = {}
            mock_track.return_value.__enter__.return_value = mock_usage
            mock_track.return_value.__exit__.return_value = None
    
    def test_start_constant_value(self):
        """Test START constant properties"""
        assert START == "__START__"
        assert isinstance(START, str)
    
    def test_end_constant_value(self):
        """Test END constant properties"""
        assert END == "__END__"
        assert isinstance(END, str)
    
    def test_start_end_different(self):
        """Test START and END are different"""
        assert START != END
    
    @patch('dspy.track_usage')
    def test_single_start_node(self, mock_track_usage):
        """Test workflow with single start node"""
        mock_usage = Mock()
        mock_usage.get_total_tokens.return_value = {}
        mock_track_usage.return_value.__enter__.return_value = mock_usage
        mock_track_usage.return_value.__exit__.return_value = None
        
        graph = Graph("test")
        start_node = MockNode("start", {"step": "started"})
        
        with patch('builtins.print'):
            graph.add_node(start_node)
            graph.add_edge(START, "start")
            graph.add_edge("start", END)
        
        result = graph.run(input="test")
        
        assert result["step"] == "started"
        assert result["_graph_metadata"]["execution_order"] == ["start"]
        assert "start" in graph.start_nodes
    
    @patch('dspy.track_usage') 
    def test_multiple_start_nodes(self, mock_track_usage):
        """Test workflow with multiple parallel start nodes"""
        mock_usage = Mock()
        mock_usage.get_total_tokens.return_value = {}
        mock_track_usage.return_value.__enter__.return_value = mock_usage
        mock_track_usage.return_value.__exit__.return_value = None
        
        graph = Graph("test")
        start1 = MockNode("start1", {"path1": "data1"})
        start2 = MockNode("start2", {"path2": "data2"})
        merger = MockNode("merger", {"merged": "combined"})
        
        with patch('builtins.print'):
            graph.add_node(start1)
            graph.add_node(start2) 
            graph.add_node(merger)
            
            # Both start from START
            graph.add_edge(START, "start1")
            graph.add_edge(START, "start2")
            
            # Both feed into merger
            graph.add_edge("start1", "merger")
            graph.add_edge("start2", "merger")
            graph.add_edge("merger", END)
        
        result = graph.run(input="test")
        
        # All data should be present
        assert result["path1"] == "data1"
        assert result["path2"] == "data2"
        assert result["merged"] == "combined"
        
        # Execution order should include all nodes
        execution_order = result["_graph_metadata"]["execution_order"]
        assert "start1" in execution_order
        assert "start2" in execution_order
        assert "merger" in execution_order
        assert len(execution_order) == 3
    
    @patch('dspy.track_usage')
    def test_conditional_end_termination(self, mock_track_usage):
        """Test conditional routing to END"""
        mock_usage = Mock()
        mock_usage.get_total_tokens.return_value = {}
        mock_track_usage.return_value.__enter__.return_value = mock_usage
        mock_track_usage.return_value.__exit__.return_value = None
        
        graph = Graph("test")
        router = MockNode("router", {"decision": "terminate"})
        processor = MockNode("processor", {"processed": "data"})
        
        def routing_logic(state):
            decision = state.get("decision", "continue")
            return "end" if decision == "terminate" else "continue"
        
        with patch('builtins.print'):
            graph.add_node(router)
            graph.add_node(processor)
            
            graph.add_edge(START, "router")
            graph.add_conditional_edges(
                "router",
                {"end": END, "continue": "processor"},
                routing_logic
            )
            graph.add_edge("processor", END)
        
        # Test early termination
        result = graph.run(input="test")
        assert result["decision"] == "terminate"
        assert "processed" not in result  # Should not reach processor
        assert result["_graph_metadata"]["execution_order"] == ["router"]
        
        # Test continuation by changing router output
        router.output_data = {"decision": "continue"}
        result2 = graph.run(input="test2")
        assert result2["decision"] == "continue"
        assert result2["processed"] == "data"
        assert result2["_graph_metadata"]["execution_order"] == ["router", "processor"]
    
    @patch('dspy.track_usage')
    def test_multiple_end_paths(self, mock_track_usage):
        """Test multiple paths leading to END"""
        mock_usage = Mock()
        mock_usage.get_total_tokens.return_value = {}
        mock_track_usage.return_value.__enter__.return_value = mock_usage
        mock_track_usage.return_value.__exit__.return_value = None
        
        graph = Graph("test")
        splitter = MockNode("splitter", {"route": "path1"})
        path1 = MockNode("path1", {"result": "path1_result"})
        path2 = MockNode("path2", {"result": "path2_result"})
        
        def router(state):
            return state.get("route", "path1")
        
        with patch('builtins.print'):
            graph.add_node(splitter)
            graph.add_node(path1)
            graph.add_node(path2)
            
            graph.add_edge(START, "splitter")
            graph.add_conditional_edges(
                "splitter",
                {"path1": "path1", "path2": "path2"},
                router
            )
            # Both paths end at END
            graph.add_edge("path1", END)
            graph.add_edge("path2", END)
        
        # Test path1
        result1 = graph.run(input="test")
        assert result1["result"] == "path1_result"
        assert "path1" in result1["_graph_metadata"]["execution_order"]
        assert "path2" not in result1["_graph_metadata"]["execution_order"]
        
        # Test path2
        splitter.output_data = {"route": "path2"}
        result2 = graph.run(input="test")
        assert result2["result"] == "path2_result"
        assert "path2" in result2["_graph_metadata"]["execution_order"]
        assert "path1" not in result2["_graph_metadata"]["execution_order"]
    
    def test_start_edge_validation(self):
        """Test START edge validation"""
        graph = Graph("test")
        node1 = MockNode("node1")
        
        with patch('builtins.print'):
            graph.add_node(node1)
            
            # Should work - START to existing node
            graph.add_edge(START, "node1")
            
            # Should fail - START to non-existent node
            with pytest.raises(ValueError, match="Target node 'missing' not found"):
                graph.add_edge(START, "missing")
    
    def test_end_edge_validation(self):
        """Test END edge validation"""
        graph = Graph("test")
        node1 = MockNode("node1")
        
        with patch('builtins.print'):
            graph.add_node(node1)
            
            # Should work - existing node to END
            graph.add_edge("node1", END)
            
            # Should fail - non-existent node to END  
            with pytest.raises(ValueError, match="Source node 'missing' not found"):
                graph.add_edge("missing", END)
    
    def test_start_node_tracking(self):
        """Test that START edges properly track start nodes"""
        graph = Graph("test")
        node1 = MockNode("node1")
        node2 = MockNode("node2")
        
        with patch('builtins.print'):
            graph.add_node(node1)
            graph.add_node(node2)
            
            # Add START edges
            graph.add_edge(START, "node1")
            graph.add_edge(START, "node2")
        
        assert "node1" in graph.start_nodes
        assert "node2" in graph.start_nodes
        assert len(graph.start_nodes) == 2
    
    @patch('dspy.track_usage')
    def test_implicit_vs_explicit_end(self, mock_track_usage):
        """Test both implicit and explicit END behavior"""
        mock_usage = Mock()
        mock_usage.get_total_tokens.return_value = {}
        mock_track_usage.return_value.__enter__.return_value = mock_usage
        mock_track_usage.return_value.__exit__.return_value = None
        
        graph = Graph("test")
        node1 = MockNode("node1", {"step": "first"})
        node2 = MockNode("node2", {"step": "second"})
        
        with patch('builtins.print'):
            graph.add_node(node1)
            graph.add_node(node2)
            
            graph.add_edge(START, "node1")
            graph.add_edge("node1", "node2")
            # node2 has no outgoing edges - implicit end
        
        result = graph.run(input="test")
        
        # Should execute both nodes and end naturally
        assert result["step"] == "second"  # node2 overwrites node1
        execution_order = result["_graph_metadata"]["execution_order"]
        assert execution_order == ["node1", "node2"]
    
    def test_workflow_visualization_with_start_end(self):
        """Test workflow visualization includes START/END"""
        graph = Graph("test")
        node1 = MockNode("node1")
        
        with patch('builtins.print'):
            graph.add_node(node1)
            graph.add_edge(START, "node1")
            graph.add_edge("node1", END)
        
        viz = graph.visualize()
        
        # Should show START marker
        assert "node1 (START)" in viz
        # Should show START/END edges
        assert "__START__ -> node1" in viz
        assert "node1 -> __END__" in viz
    
    @patch('dspy.track_usage')
    def test_complex_start_end_workflow(self, mock_track_usage):
        """Test complex workflow with multiple START/END patterns"""
        mock_usage = Mock()
        mock_usage.get_total_tokens.return_value = {}
        mock_track_usage.return_value.__enter__.return_value = mock_usage
        mock_track_usage.return_value.__exit__.return_value = None
        
        graph = Graph("complex")
        
        # Create nodes
        init1 = MockNode("init1", {"init": "data1"})
        init2 = MockNode("init2", {"init": "data2"}) 
        processor = MockNode("processor", {"decision": "end_early"})
        finalizer = MockNode("finalizer", {"final": "done"})
        
        def router(state):
            return "early_end" if state.get("decision") == "end_early" else "continue"
        
        with patch('builtins.print'):
            graph.add_node(init1)
            graph.add_node(init2)
            graph.add_node(processor)
            graph.add_node(finalizer)
            
            # Multiple start points
            graph.add_edge(START, "init1")
            graph.add_edge(START, "init2")
            
            # Converge to processor
            graph.add_edge("init1", "processor")
            graph.add_edge("init2", "processor")
            
            # Conditional routing from processor
            graph.add_conditional_edges(
                "processor",
                {"early_end": END, "continue": "finalizer"},
                router
            )
            
            # Normal end
            graph.add_edge("finalizer", END)
        
        result = graph.run(input="test")
        
        # Should have parallel init, then early termination
        assert result["init"] == "data2"  # Last init wins
        assert result["decision"] == "end_early"
        assert "final" not in result  # Should not reach finalizer
        
        execution_order = result["_graph_metadata"]["execution_order"]
        assert "init1" in execution_order
        assert "init2" in execution_order  
        assert "processor" in execution_order
        assert "finalizer" not in execution_order