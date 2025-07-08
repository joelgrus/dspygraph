"""
Tool use agent
"""
import dspy
from typing import Dict, Any
from dspy_langgraph import AgentNode
from ..types import AgentState

class ToolUseModule(AgentNode[AgentState]):
    """Agent that handles tool-based queries using ReAct pattern"""
    
    def __init__(self, compile_path: str = "compiled_tool_use.json") -> None:
        super().__init__(compile_path)
    
    def _create_module(self) -> dspy.Module:
        return dspy.ReAct("query -> answer", tools=[self.dummy_tool])
    
    def dummy_tool(self, query: str) -> str:
        """Dummy tool for demonstration purposes"""
        return f"Executing dummy tool for query: {query}"
    
    def _process_state(self, state: AgentState) -> Dict[str, Any]:
        question = state["question"]
        prediction = self.module(query=question)
        tool_output = prediction.answer
        print(f"Tool used for: '{question}', output: '{tool_output}'")
        return {"tool_output": tool_output, "response": tool_output}