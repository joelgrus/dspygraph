"""
Creative response agent
"""
import dspy
from typing import Dict, Any
from dspy_langgraph import AgentNode
from ..types import AgentState

class CreativeResponseSignature(dspy.Signature):
    """Signature for creative response generation"""
    prompt: str = dspy.InputField(desc="The creative prompt")
    creative_output: str = dspy.OutputField(desc="The creative response")

class CreativeResponseModule(AgentNode[AgentState]):
    """Agent that generates creative responses"""
    
    def __init__(self) -> None:
        super().__init__()
    
    def _create_module(self) -> dspy.Module:
        return dspy.ChainOfThought(CreativeResponseSignature)
    
    def _process_state(self, state: AgentState) -> Dict[str, Any]:
        prompt = state["question"]
        prediction = self.module(prompt=prompt)
        creative_output = prediction.creative_output
        print(f"Creative response for: '{prompt}', response: '{creative_output}'")
        return {"response": creative_output}