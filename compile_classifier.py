#!/usr/bin/env python3
"""
Compilation script using the clean separation of framework and application code
"""
from dspy.teleprompt import BootstrapFewShot
from dspy_langgraph import configure_dspy
from question_classifier_app import QuestionClassifier
from question_classifier_app.compilation import classification_metric, get_training_data

def compile_classifier() -> None:
    """Compile the question classifier"""
    print("Compiling QuestionClassifier...")
    
    # Configure DSPy for compilation
    configure_dspy("openai/gpt-4o-mini")  # Use same model as main app
    
    # Create classifier instance
    classifier = QuestionClassifier()
    
    # Get training data and create compiler
    trainset = get_training_data()
    compiler = BootstrapFewShot(metric=classification_metric)
    
    # Compile and save
    classifier.compile(compiler, trainset, compile_path="compiled_classifier.json")
    print("Compiled classifier saved to compiled_classifier.json")

def main() -> None:
    """Main compilation entry point"""
    print("Starting agent compilation...")
    compile_classifier()
    print("Compilation complete!")

if __name__ == "__main__":
    main()