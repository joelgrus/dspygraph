"""
Tests for Node functionality
"""
import pytest
import dspy
from unittest.mock import Mock, patch
from typing import Dict, Any

from dspygraph import Node


class MockSignature(dspy.Signature):
    """Mock signature for testing"""
    input_text: str = dspy.InputField()
    output_text: str = dspy.OutputField()


class ExampleTestNode(Node):
    """Test node implementation"""
    
    def _create_module(self) -> dspy.Module:
        return dspy.Predict(MockSignature)
    
    def process(self, state: Dict[str, Any]) -> Dict[str, Any]:
        result = self.module(input_text=state["input"])
        return {"output": result.output_text}


class TestNode:
    """Test suite for Node"""
    
    def setup_method(self):
        """Setup for each test"""
        # Configure DSPy with a mock LM to avoid API calls
        with patch('dspy.LM') as mock_lm:
            mock_lm.return_value = Mock()
            dspy.configure(lm=mock_lm.return_value)
    
    def test_node_initialization(self):
        """Test basic node creation"""
        node = ExampleTestNode("test_node")
        
        assert node.name == "test_node"
        assert node.node_id is not None
        assert len(node.node_id) == 36  # UUID length
        assert node.compiled == False
        assert node._execution_count == 0
        assert node.module is not None
    
    def test_node_unique_ids(self):
        """Test that nodes get unique IDs"""
        node1 = ExampleTestNode("node1")
        node2 = ExampleTestNode("node2")
        
        assert node1.node_id != node2.node_id
    
    @patch('dspy.track_usage')
    def test_node_execution(self, mock_track_usage):
        """Test node execution with mocked DSPy"""
        # Setup mock usage tracker
        mock_usage = Mock()
        mock_usage.get_total_tokens.return_value = {"total": 100}
        mock_track_usage.return_value.__enter__.return_value = mock_usage
        mock_track_usage.return_value.__exit__.return_value = None
        
        # Setup mock DSPy module
        node = ExampleTestNode("test_node")
        mock_result = Mock()
        mock_result.output_text = "test output"
        node.module = Mock()
        node.module.return_value = mock_result
        
        # Test execution
        state = {"input": "test input"}
        result = node(state)
        
        # Verify execution
        assert node._execution_count == 1
        assert result["output"] == "test output"
        assert "_node_metadata" in result
        
        metadata = result["_node_metadata"]
        assert metadata["node_name"] == "test_node"
        assert metadata["execution_count"] == 1
        assert metadata["compiled"] == False
        assert "execution_time" in metadata
        assert "usage" in metadata
    
    def test_compilation_workflow(self):
        """Test compilation methods"""
        node = ExampleTestNode("test_node")
        
        # Mock the module and compilation
        mock_compiler = Mock()
        mock_compiled_module = Mock()
        mock_compiler.compile.return_value = mock_compiled_module
        
        trainset = [Mock()]
        
        # Test compilation
        with patch('builtins.print'):  # Suppress prints
            node.compile(mock_compiler, trainset, compile_path="test.json")
        
        assert node.compiled == True
        # The module should be replaced by the compiled one
        assert node.module == mock_compiled_module
        # Verify compiler was called with original module
        mock_compiler.compile.assert_called_once()
    
    def test_save_and_load_compiled(self):
        """Test save/load compiled module"""
        node = ExampleTestNode("test_node")
        
        # Mock module save/load
        node.module.save = Mock()
        node.module.load = Mock()
        
        # Test save
        with patch('builtins.print'):
            node.save_compiled("test.json")
        node.module.save.assert_called_once_with("test.json")
        
        # Test load
        with patch('builtins.print'), patch('os.path.exists', return_value=True):
            node.load_compiled("test.json")
        node.module.load.assert_called_once_with("test.json")
        assert node.compiled == True
    
    def test_load_compiled_file_not_found(self):
        """Test load_compiled with missing file"""
        node = ExampleTestNode("test_node")
        
        with patch('os.path.exists', return_value=False):
            with pytest.raises(FileNotFoundError):
                node.load_compiled("missing.json")
    
    def test_ensure_compiled_already_compiled(self):
        """Test ensure_compiled when already compiled"""
        node = ExampleTestNode("test_node")
        node.compiled = True
        
        # Should not raise or do anything
        node.ensure_compiled()
        assert node.compiled == True
    
    def test_ensure_compiled_with_file(self):
        """Test ensure_compiled with existing file"""
        node = ExampleTestNode("test_node")
        node.module.load = Mock()
        
        with patch('builtins.print'), patch('os.path.exists', return_value=True):
            node.ensure_compiled("test.json")
        
        assert node.compiled == True
        node.module.load.assert_called_once_with("test.json")
    
    def test_ensure_compiled_no_file(self):
        """Test ensure_compiled without file raises error"""
        node = ExampleTestNode("test_node")
        
        with pytest.raises(RuntimeError):
            node.ensure_compiled()
    
    def test_node_repr(self):
        """Test string representation"""
        node = ExampleTestNode("test_node")
        repr_str = repr(node)
        
        assert "test_node" in repr_str
        assert "compiled=False" in repr_str
        assert "executions=0" in repr_str
    
    def test_abstract_methods(self):
        """Test that Node is properly abstract"""
        # Should not be able to instantiate Node directly
        with pytest.raises(TypeError):
            Node("abstract_node")


class BadNode(Node):
    """Node with missing implementations for testing"""
    pass


class IncompleteNode(Node):
    """Node with only one method implemented"""
    def _create_module(self):
        return Mock()


def test_abstract_method_enforcement():
    """Test that abstract methods are enforced"""
    # BadNode missing both methods
    with pytest.raises(TypeError):
        BadNode("bad")
    
    # IncompleteNode missing process method
    with pytest.raises(TypeError):
        IncompleteNode("incomplete")