# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a DSPy + LangGraph integration project that demonstrates how to combine DSPy's intelligent modules with LangGraph's state management and routing capabilities. The system creates an agent that classifies user questions and routes them to appropriate specialized modules.

## Key Architecture Components

### DSPy Modules
- **QuestionClassifier**: Classifies questions into categories ('factual', 'creative', 'tool_use', 'unknown')
- **ToolUseModule**: Handles tool-based queries using DSPy's ReAct pattern
- **FactualAnswerModule**: Provides factual answers using ChainOfThought
- **CreativeResponseModule**: Generates creative responses

### LangGraph Integration
- **AgentState**: TypedDict defining the workflow state (question, classification, response, tool_output)
- **Workflow**: StateGraph that routes questions based on classification results
- **Conditional Routing**: Routes to different modules based on question classification

## Required Setup

### Model Configuration
The system requires OpenAI API access. Configure your OpenAI API key in your environment:
- Main system uses: `openai/gpt-4o-mini`
- Compilation script uses: `openai/gpt-4.1-nano`

### Compiled Classifier
The main application requires a compiled classifier. Run this before using the system:
```bash
python compile_classifier.py
```
This creates `compiled_classifier.json` which is loaded by the main application.

## Running the System

### Basic Usage
```bash
python main.py
```
This runs predefined test cases for factual, tool use, and creative questions.

### Development Commands
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

## Important Files
- `main.py`: Main application entry point with agent workflow
- `compile_classifier.py`: Compiles and optimizes the question classifier
- `compiled_classifier.json`: Serialized compiled classifier (generated)
- `pyproject.toml`: Project configuration and dependencies