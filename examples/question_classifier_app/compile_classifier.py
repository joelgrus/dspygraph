#!/usr/bin/env python3
"""
Compilation script for the question classifier using graph nodes
"""

from typing import Any

import dspy
from dspy.teleprompt import BootstrapFewShot

from .nodes import QuestionClassifierNode


def classification_metric(
    pred_object: Any, true_category_str: str, trace: Any | None = None
) -> bool:
    """Metric for evaluating question classification accuracy"""
    return pred_object.category == true_category_str


TRAINING_DATA = [
    ("What is the capital of France?", "factual"),
    ("Who won the World Cup in 2022?", "factual"),
    ("What is 5 + 7?", "tool_use"),
    ("Calculate the square root of 64.", "tool_use"),
    ("Write a haiku about autumn leaves.", "creative"),
    ("Tell me a joke.", "creative"),
    ("Define photosynthesis.", "factual"),
    ("What's the population of Tokyo?", "factual"),
    ("Convert 10 miles to kilometers.", "tool_use"),
    ("Summarize this article: https://example.com/article", "unknown"),
    ("What's the current time?", "tool_use"),
]


def compile_classifier() -> None:
    """Compile the question classifier node"""
    print("Compiling QuestionClassifier node...")

    # Configure DSPy for compilation
    lm = dspy.LM("openai/gpt-4o-mini")
    dspy.configure(lm=lm)

    # Create classifier node
    classifier = QuestionClassifierNode("classifier")

    # Convert training data to DSPy Examples
    trainset = [
        dspy.Example(question=question, category=category).with_inputs("question")
        for question, category in TRAINING_DATA
    ]

    # Create compiler and compile
    compiler = BootstrapFewShot(metric=classification_metric)
    classifier.compile(compiler, trainset, compile_path="compiled_classifier.json")
    print("Compiled classifier saved to compiled_classifier.json")


def main() -> None:
    """Main compilation entry point"""
    print("Starting classifier node compilation...")
    compile_classifier()
    print("Compilation complete!")


if __name__ == "__main__":
    main()
