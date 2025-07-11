"""
Factual answer agent
"""
import dspy
from typing import Dict, Any
from dspy_langgraph import AgentNode
from ..types import AgentState

class FactualAnswerModule(AgentNode[AgentState]):
    """Agent that provides factual answers using chain of thought reasoning"""
    
    def __init__(self) -> None:
        super().__init__()
    
    def _create_module(self) -> dspy.Module:
        return dspy.ChainOfThought("question -> answer")
    
    def _process_state(self, state: AgentState) -> Dict[str, Any]:
        question = state["question"]
        prediction = self.module(question=question)
        answer = prediction.answer
        print(f"Factual answer for: '{question}', answer: '{answer}'")
        return {"response": answer}