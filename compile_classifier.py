import dspy
import os
from dspy.teleprompt import BootstrapFewShot
from typing import Literal

# --- Configuration (same as in your main.py for consistency) ---
lm = dspy.LM("openai/gpt-4.1-nano")
dspy.configure(lm=lm)

# --- DSPy Signature and Module Definitions (copy/paste from main.py) ---
class QuestionClassificationSignature(dspy.Signature):
    # ... (definition as before) ...
    question: str = dspy.InputField(desc="The user's question")
    category: Literal['factual', 'creative', 'tool_use', 'unknown'] = dspy.OutputField(
        desc="""The category of the question."""
    )

class QuestionClassifier(dspy.Module):
    def __init__(self):
        super().__init__()
        self.classify = dspy.Predict(QuestionClassificationSignature)
    def forward(self, question):
        return self.classify(question=question)

# --- Metric and Training Data (copy/paste from main.py) ---
def classification_metric(pred_object, true_category_str, trace=None):
    return pred_object.category == true_category_str

trainset = [
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

# --- Compilation Logic ---
if __name__ == "__main__":
    classifier_instance = QuestionClassifier()
    compiled_classifier_path = "compiled_classifier.json"

    print("Starting compilation of QuestionClassifier...")
    compiled_classifier = BootstrapFewShot(metric=classification_metric).compile(
        classifier_instance,
        trainset=trainset
    )
    compiled_classifier.save(compiled_classifier_path)
    print(f"Compilation complete. Compiled classifier saved to {compiled_classifier_path}")