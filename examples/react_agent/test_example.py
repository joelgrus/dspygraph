"""
Simple test script for the React agent example
"""
import dspy
from unittest.mock import patch, Mock, MagicMock
from .graph import create_react_agent_graph
from .nodes import ReactAgentNode, ToolExecutorNode


def test_react_agent_without_api():
    """Test the React agent structure without making API calls"""
    
    print("🧪 Testing React Agent Structure...")
    
    # Test graph creation
    graph = create_react_agent_graph(max_steps=3)
    
    # Verify graph structure
    assert graph.name == "ReactAgent"
    assert len(graph.nodes) == 3
    assert "react_agent" in graph.nodes
    assert "tool_executor" in graph.nodes  
    assert "max_steps_checker" in graph.nodes
    
    # Verify edges (including cycle)
    edge_count = len(graph.edges)
    assert edge_count >= 6  # Should have edges including the cycle
    
    # Test graph visualization
    viz = graph.visualize()
    assert "ReactAgent" in viz
    assert "react_agent" in viz
    assert "tool_executor -> react_agent" in viz  # Verify cycle exists
    
    print("✅ Graph structure test passed!")
    
    # Test tools without API
    from .tools import execute_tool
    
    # Test calculator
    calc_result = execute_tool("calculator", "5 + 3")
    assert calc_result["success"] is True
    assert calc_result["result"] == "8"
    
    # Test search
    search_result = execute_tool("search", "capital of france") 
    assert search_result["success"] is True
    assert "Paris" in search_result["result"]
    
    print("✅ Tools test passed!")
    
    # Test graph execution with mocked DSPy
    print("🧪 Testing Graph Execution...")
    
    with patch('dspy.configure'), patch('dspy.LM') as mock_lm, \
         patch('dspy.track_usage') as mock_track:
        # Setup mocks
        mock_lm_instance = Mock()
        mock_lm.return_value = mock_lm_instance
        dspy.configure(lm=mock_lm_instance)
        
        # Mock usage tracking
        mock_usage = Mock()
        mock_usage.get_total_tokens.return_value = {}
        mock_track.return_value.__enter__.return_value = mock_usage
        mock_track.return_value.__exit__.return_value = None
        
        # Mock the React agent's DSPy module
        react_node = graph.nodes["react_agent"]
        mock_result = Mock()
        mock_result.thought = "I need to calculate 2 + 3"
        mock_result.action = "calculator: 2 + 3"
        react_node.module = Mock(return_value=mock_result)
        
        # Test execution with a simple query
        result = graph.run(
            question="What is 2 + 3?",
            max_steps=3,
            step_count=0,
            thoughts=[],
            actions=[],
            observations=[],
            max_iterations=10,
            max_node_executions=5
        )
        
        # Verify execution completed
        assert "_graph_metadata" in result
        metadata = result["_graph_metadata"]
        assert metadata["success"] is True
        assert metadata["total_iterations"] > 0
        assert "react_agent" in metadata["node_execution_counts"]
        
        print("✅ Graph execution test passed!")
        
    print("🎉 All tests passed! React agent is ready.")


if __name__ == "__main__":
    test_react_agent_without_api()