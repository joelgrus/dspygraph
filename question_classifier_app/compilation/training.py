"""
Training data for the question classifier application
"""
import dspy
from typing import List

def get_training_data() -> List[dspy.Example]:
    """
    Get training data for question classification
    
    Returns:
        List of DSPy examples for training
    """
    return [
        dspy.Example(question="What is the capital of France?", category="factual").with_inputs("question"),
        dspy.Example(question="Who won the World Cup in 2022?", category="factual").with_inputs("question"),
        dspy.Example(question="What is 5 + 7?", category="tool_use").with_inputs("question"),
        dspy.Example(question="Calculate the square root of 64.", category="tool_use").with_inputs("question"),
        dspy.Example(question="Write a haiku about autumn leaves.", category="creative").with_inputs("question"),
        dspy.Example(question="Tell me a joke.", category="creative").with_inputs("question"),
        dspy.Example(question="Define photosynthesis.", category="factual").with_inputs("question"),
        dspy.Example(question="What's the population of Tokyo?", category="factual").with_inputs("question"),
        dspy.Example(question="Convert 10 miles to kilometers.", category="tool_use").with_inputs("question"),
        dspy.Example(question="Summarize this article: https://example.com/article", category="unknown").with_inputs("question"),
        dspy.Example(question="What's the current time?", category="tool_use").with_inputs("question"),
    ]