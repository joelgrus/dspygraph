"""
Simplified DSPy nodes for the question classifier application
"""

from typing import Any

import dspy

from dspygraph import Node

from .types import QuestionCategory


class QuestionClassificationSignature(dspy.Signature):
    """Classifies a user's question into a specific category."""

    question: str = dspy.InputField(desc="The user's question")
    category: QuestionCategory = dspy.OutputField(
        desc="Classification category of the question (e.g., 'factual', 'creative', 'tool_use', 'unknown')"
    )


class QuestionClassifierNode(Node):
    """Node that classifies questions into categories"""

    def _create_module(self) -> dspy.Module:
        return dspy.Predict(QuestionClassificationSignature)

    def process(self, state: dict[str, Any]) -> dict[str, Any]:
        """Classify the question from state"""
        result = self.module(question=state["question"])
        print(f"  -> Classified as: {result.category}")
        return {"classification": result.category}


class FactualAnswerNode(Node):
    """Node that provides factual answers using chain of thought reasoning"""

    def _create_module(self) -> dspy.Module:
        return dspy.ChainOfThought("question -> answer")

    def process(self, state: dict[str, Any]) -> dict[str, Any]:
        """Generate factual answer"""
        result = self.module(question=state["question"])
        print(f"  -> Factual answer: {result.answer[:100]}...")
        return {"response": result.answer}


class CreativeResponseSignature(dspy.Signature):
    """Signature for creative response generation"""

    prompt: str = dspy.InputField(desc="The creative prompt")
    creative_output: str = dspy.OutputField(desc="The creative response")


class CreativeResponseNode(Node):
    """Node that generates creative responses"""

    def _create_module(self) -> dspy.Module:
        return dspy.ChainOfThought(CreativeResponseSignature)

    def process(self, state: dict[str, Any]) -> dict[str, Any]:
        """Generate creative response"""
        result = self.module(prompt=state["question"])
        print(f"  -> Creative response: {result.creative_output[:100]}...")
        return {"response": result.creative_output}


class ToolUseNode(Node):
    """Node that handles tool-based queries using ReAct pattern"""

    def _create_module(self) -> dspy.Module:
        return dspy.ReAct("query -> answer", tools=[self.dummy_tool])

    def dummy_tool(self, query: str) -> str:
        """Dummy tool for demonstration purposes"""
        return f"Tool executed for: {query}"

    def process(self, state: dict[str, Any]) -> dict[str, Any]:
        """Execute tool and return result"""
        result = self.module(query=state["question"])
        print(f"  -> Tool output: {result.answer}")
        return {"tool_output": result.answer, "response": result.answer}
