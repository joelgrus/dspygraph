# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This project demonstrates a clean architecture for integrating DSPy's intelligent modules with LangGraph's state management and routing capabilities. The system creates an intelligent agent that classifies user questions and routes them to appropriate specialized response modules.

## Architecture Design

### Framework vs Application Separation

The codebase is organized with clean separation between reusable framework code and application-specific implementations:

**Framework (`dspy_langgraph/`):**
- `AgentNode` base class: Unified abstraction for DSPy modules + LangGraph nodes
- `configure_dspy()`: Shared DSPy configuration utilities
- Reusable across any DSPy + LangGraph project

**Application (`question_classifier_app/`):**
- Specific agent implementations (classifier, factual, creative, tool_use)
- Application-specific types (`AgentState`, `QuestionCategory`)
- Routing logic and compilation utilities
- Training data and metrics

### Key Components

**Agent Implementations:**
- `QuestionClassifier`: Classifies questions into categories ('factual', 'creative', 'tool_use', 'unknown')
- `ToolUseModule`: Handles tool-based queries using DSPy's ReAct pattern
- `FactualAnswerModule`: Provides factual answers using ChainOfThought
- `CreativeResponseModule`: Generates creative responses

**Workflow Management:**
- `AgentState`: TypedDict defining workflow state (question, classification, response, tool_output)
- `route_question()`: Conditional routing based on classification results
- LangGraph StateGraph orchestrates the entire workflow

## Required Setup

### Model Configuration
The system requires OpenAI API access. Configure your OpenAI API key in your environment.
- Uses: `openai/gpt-4o-mini` for both compilation and runtime

### Compiled Classifier
The main application requires a compiled classifier. Run this before using the system:
```bash
python compile_classifier.py
```
This creates `compiled_classifier.json` which is loaded by the main application.

## Development Commands

```bash
# Compile the classifier (required first time)
python compile_classifier.py

# Run the main application
python main.py

# Install dependencies
uv sync
```

## Dependencies
- `dspy>=2.6.27`: Core DSPy framework for language model programming
- `langgraph>=0.5.1`: State graph framework for workflow management
- Python 3.11+ required

## Code Structure

```
dspy_langgraph/                    # Reusable framework
├── base.py                        # AgentNode base class
└── config.py                      # DSPy configuration utilities

question_classifier_app/           # Application-specific code
├── types.py                       # AgentState and QuestionCategory
├── routing.py                     # Route logic
├── agents/                        # Agent implementations
│   ├── classifier.py
│   ├── factual.py
│   ├── creative.py
│   └── tool_use.py
└── compilation/                   # Training data and metrics
    ├── metrics.py
    └── training.py

main.py                           # Main application entry point
compile_classifier.py             # Compilation script
```

## Important Files
- `main.py`: Main application entry point with agent workflow
- `compile_classifier.py`: Compiles and optimizes the question classifier
- `compiled_classifier.json`: Serialized compiled classifier (generated)
- `dspy_langgraph/`: Reusable framework for DSPy + LangGraph integration
- `question_classifier_app/`: Application-specific implementations