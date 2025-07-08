"""
Question classification agent
"""
import dspy
from typing import Dict, Any
from dspy_langgraph import AgentNode
from ..types import AgentState, QuestionCategory

class QuestionClassificationSignature(dspy.Signature):
    """Classifies a user's question into a specific category."""
    question: str = dspy.InputField(desc="The user's question")
    category: QuestionCategory = dspy.OutputField(
        desc="Classification category of the question (e.g., 'factual', 'creative', 'tool_use', 'unknown')"
    )

class QuestionClassifier(AgentNode[AgentState]):
    """Agent that classifies questions into categories"""
    
    def __init__(self, compile_path: str = "compiled_classifier.json") -> None:
        super().__init__(compile_path)
    
    def _create_module(self) -> dspy.Module:
        return dspy.Predict(QuestionClassificationSignature)
    
    def _process_state(self, state: AgentState) -> Dict[str, Any]:
        question = state["question"]
        prediction = self.module(question=question)
        classification_result = prediction.category
        print(f"Question: '{question}' classified as: '{classification_result}'")
        return {"classification": classification_result}