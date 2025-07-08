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
    
    # Get training data
    trainset = get_training_data()
    
    # Compile using BootstrapFewShot
    compiled_classifier = BootstrapFewShot(metric=classification_metric).compile(
        classifier.module,
        trainset=trainset
    )
    
    # Save compiled model
    compiled_classifier.save(classifier.compile_path)
    print(f"Compiled classifier saved to {classifier.compile_path}")

def main() -> None:
    """Main compilation entry point"""
    print("Starting agent compilation...")
    compile_classifier()
    print("Compilation complete!")

if __name__ == "__main__":
    main()