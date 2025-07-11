#!/usr/bin/env python3
"""
Simple example demonstrating DSPy-LangGraph integration

This example shows how to create a basic agent using the framework.
"""
import dspy
from typing import Dict, Any
from dspy_langgraph import AgentNode, configure_dspy
from langgraph.graph import StateGraph, START, END


class SimpleState(dict):
    """Simple state type for demonstration"""
    question: str
    answer: str


class SimpleAnswerSignature(dspy.Signature):
    """Answer a question in a helpful way."""
    question: str = dspy.InputField(desc="The user's question")
    answer: str = dspy.OutputField(desc="A helpful answer to the question")


class SimpleAgent(AgentNode[SimpleState]):
    """A simple agent that answers questions"""
    
    def _create_module(self) -> dspy.Module:
        return dspy.Predict(SimpleAnswerSignature)
    
    def _process_state(self, state: SimpleState) -> Dict[str, Any]:
        question = state["question"]
        prediction = self.module(question=question)
        return {"answer": prediction.answer}


def create_simple_graph():
    """Create a simple graph with one agent"""
    # Create the agent
    agent = SimpleAgent()
    
    # Create the graph
    graph = StateGraph(SimpleState)
    graph.add_node("answer", agent)
    graph.add_edge(START, "answer")
    graph.add_edge("answer", END)
    
    return graph.compile()


def main():
    """Run the simple example"""
    # Configure DSPy
    configure_dspy()
    
    # Create the application
    app = create_simple_graph()
    
    # Test questions
    questions = [
        "What is Python?",
        "How do you make coffee?",
        "What is the meaning of life?",
    ]
    
    for question in questions:
        print(f"\n🤔 Question: {question}")
        result = app.invoke({"question": question})
        print(f"💡 Answer: {result['answer']}")


if __name__ == "__main__":
    main()