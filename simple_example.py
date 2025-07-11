#!/usr/bin/env python3
"""
Simple example demonstrating pure DSPy usage
"""

import dspy


class SimpleAnswerSignature(dspy.Signature):
    """Answer a question in a helpful way."""

    question: str = dspy.InputField(desc="The user's question")
    answer: str = dspy.OutputField(desc="A helpful answer to the question")


class SimpleAgent:
    """A simple agent that answers questions"""

    def __init__(self):
        self.module = dspy.Predict(SimpleAnswerSignature)

    def run(self, question: str) -> str:
        """Process a question and return an answer"""
        prediction = self.module(question=question)
        return prediction.answer


def main():
    """Run the simple example"""
    # Configure DSPy directly
    lm = dspy.LM("openai/gpt-4o-mini")
    dspy.configure(lm=lm)

    # Create the agent
    agent = SimpleAgent()

    # Test questions
    questions = [
        "What is Python?",
        "How do you make coffee?",
        "What is the meaning of life?",
    ]

    for question in questions:
        print(f"\n🤔 Question: {question}")
        answer = agent.run(question)
        print(f"💡 Answer: {answer}")


if __name__ == "__main__":
    main()
