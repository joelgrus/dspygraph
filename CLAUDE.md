# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This project demonstrates a clean architecture for integrating DSPy's intelligent modules with graph-based state management and routing capabilities. The system creates an intelligent agent that classifies user questions and routes them to appropriate specialized response modules.

## Architecture Design

### Framework vs Application Separation

The codebase is organized with clean separation between reusable framework code and application-specific implementations:

**Framework (dspygraph/):**
- Node base class: Unified abstraction for DSPy modules + graph nodes
- configure_dspy(): Shared DSPy configuration utilities
- Reusable across any DSPy + Graph project

**Application (examples/question_classifier_app/):**
- Specific agent implementations (classifier, factual, creative, tool_use)
- Application-specific types (AgentState, QuestionCategory)
- Routing logic and compilation utilities
- Training data and metrics

### Key Components

**Agent Implementations:**
- QuestionClassifier: Classifies questions into categories ('factual', 'creative', 'tool_use', 'unknown')
- ToolUseModule: Handles tool-based queries using DSPy's ReAct pattern
- FactualAnswerModule: Provides factual answers using ChainOfThought
- CreativeResponseModule: Generates creative responses

**Graph Management:**
- AgentState: TypedDict defining graph state (question, classification, response, tool_output)
- route_question(): Conditional routing based on classification results
- Graph StateGraph orchestrates the entire workflow

**Compilation API:**
- agent.compile(compiler, trainset, compile_path=None): Compile agent with DSPy compiler
- agent.load_compiled(path): Load pre-compiled agent from file
- agent.save_compiled(path): Save compiled agent to file

## Required Setup

### Model Configuration
The system requires OpenAI API access. Configure your OpenAI API key in your environment.
- Uses: `openai/gpt-4o-mini` for both compilation and runtime (configured in `dspygraph/constants.py`)

### Compiled Classifier
The main application requires a compiled classifier. Run this before using the system:
```bash
cd examples/question_classifier_app
python compile_classifier.py
```
This creates compiled_classifier.json which is loaded by the main application.

### Compilation Process
The compilation process uses DSPy's optimization system:
1. Create agent instance: classifier = QuestionClassifier()
2. Get training data: trainset = get_training_data()
3. Create compiler: compiler = BootstrapFewShot(metric=classification_metric)
4. Compile: classifier.compile(compiler, trainset, compile_path="compiled_classifier.json")

## Development Commands

```bash
# Install dependencies
uv sync

# Compile the classifier (required first time)
cd examples/question_classifier_app
python compile_classifier.py

# Run the main application
python main.py
```

## Dependencies
- `dspy>=2.6.27`: Core DSPy framework for language model programming
- Python 3.11+ required
- Python 3.11+ required

## Code Structure

```
dspygraph/                         # Reusable framework
├── base.py                        # Node base class
├── config.py                      # DSPy configuration utilities
└── constants.py                   # Framework constants

examples/                          # Example applications
└── question_classifier_app/       # Question classifier example
    ├── main.py                    # Main application entry point
    ├── compile_classifier.py      # Compilation script
    ├── types.py                   # AgentState and QuestionCategory
    ├── workflow.py                # Graph workflow definition
    └── nodes.py                   # Node implementations
```

## Important Files
- examples/question_classifier_app/main.py: Main application entry point with graph workflow
- examples/question_classifier_app/compile_classifier.py: Compiles and optimizes the question classifier
- examples/question_classifier_app/compiled_classifier.json: Serialized compiled classifier (generated)
- dspygraph/: Reusable framework for DSPy + Graph integration
- examples/question_classifier_app/: Application-specific implementations

## Usage Examples

### Manual Compilation
```python
from dspy.teleprompt import BootstrapFewShot
from examples.question_classifier_app import QuestionClassifier
from examples.question_classifier_app.compilation import classification_metric, get_training_data

classifier = QuestionClassifier()
trainset = get_training_data()
compiler = BootstrapFewShot(metric=classification_metric)
classifier.compile(compiler, trainset, compile_path="my_classifier.json")
```

### Loading Compiled Agent
```python
classifier = QuestionClassifier()
classifier.load_compiled("compiled_classifier.json")
```