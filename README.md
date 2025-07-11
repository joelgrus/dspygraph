# DSPy Graph Framework

An intelligent question-answering system that demonstrates clean architecture for combining DSPy's powerful language model programming with graph-based state management and routing capabilities.

## What This Project Does

This system creates an intelligent agent that:
1. **Classifies** incoming questions into categories (factual, creative, tool-use, or unknown)
2. **Routes** each question to the most appropriate specialized response module
3. **Generates** tailored responses using different reasoning patterns for each category

### Example Interactions

```
Question: "What is the capital of France?"
-> Classified as: factual
-> Routed to: FactualAnswerModule
-> Response: "The capital of France is Paris."

Question: "Write a haiku about programming"
-> Classified as: creative  
-> Routed to: CreativeResponseModule
-> Response: [Generated haiku]

Question: "What is 15 x 24?"
-> Classified as: tool_use
-> Routed to: ToolUseModule  
-> Response: [Calculated result]
```

## Key Features

### Clean Architecture
- **Reusable Framework**: dspygraph/ provides a base Node class that can be used for any DSPy project
- **Application-Specific Code**: question_classifier_app/ contains the specific implementations for this question-answering system
- **Clear Separation**: Framework concerns are separated from application logic

### Intelligent Routing
- Uses DSPy's compilation system to optimize question classification
- Conditional routing based on question type
- Specialized response modules for different reasoning patterns

### Production-Ready
- Compiled models for optimized performance
- Proper error handling and validation
- Type safety with comprehensive type annotations
- Clean compilation API with explicit paths

## Quick Start

### Prerequisites
- Python 3.11+
- OpenAI API key (set as environment variable)

### Installation
```bash
# Clone and navigate to the project
git clone <repository-url>
cd dspygraph

# Install dependencies
uv sync
```

### Running the Examples

#### Simple Example (Quick Start)
```bash
# Run the basic example (no compilation needed)
python simple_example.py
```

This shows basic DSPy graph integration with a single agent that answers questions.

#### Question Classifier App (Advanced Example)
```bash
# 1. Compile the classifier (required first time)
python -m examples.question_classifier_app.compile_classifier

# 2. Run the main application
python -m examples.question_classifier_app.main
```

This demonstrates an intelligent routing system that classifies questions and routes them to specialized response modules.

#### React Agent (Tool Integration Example)
```bash
# Run the React agent (no compilation needed)
python -m examples.react_agent.main

# Or run the demonstration
python -m examples.react_agent.workflow
```

This showcases a ReAct (Reasoning + Acting) agent that uses iterative reasoning with tool execution, demonstrating graph-based loops and state management.

## How It Works

### Architecture Overview

```
User Question -> QuestionClassifier -> Router -> Specialized Module -> Response
```

1. **Question Classification**: DSPy module analyzes the question and assigns a category
2. **Intelligent Routing**: Graph routes to the appropriate response module
3. **Specialized Processing**: Each module uses different reasoning patterns:
   - **Factual**: Chain-of-thought reasoning for factual questions
   - **Creative**: Optimized for creative content generation
   - **Tool Use**: ReAct pattern for computational tasks
4. **Response Generation**: Tailored response based on question type

### Framework Design

The project showcases a reusable pattern for DSPy + Graph integration:

- **Node**: Base class that unifies DSPy modules with graph nodes
- **Clean Interfaces**: Each node implements both DSPy module creation and graph state processing
- **Compilation Support**: Built-in support for DSPy's optimization system

### Compilation API

The framework provides a clean API for compiling agents:

```python
# Create agent and compiler
agent = QuestionClassifier()
compiler = BootstrapFewShot(metric=classification_metric)
trainset = get_training_data()

# Compile with optional save path
agent.compile(compiler, trainset, compile_path="my_model.json")

# Load compiled model
agent.load_compiled("my_model.json")

# Save compiled model
agent.save_compiled("my_model.json")
```

## Extending the System

### Adding New Question Types
1. Create a new agent in question_classifier_app/agents/
2. Add the new category to QuestionCategory type
3. Update training data and routing logic
4. Recompile the classifier

### Creating New Applications
The dspygraph/ framework can be reused for entirely different applications:

```python
from dspygraph import Node, configure_dspy

class MyCustomAgent(Node):
    def _create_module(self):
        return dspy.ChainOfThought("input -> output")
    
    def _process_state(self, state):
        # Your custom logic here
        return {"result": "processed"}
```

## Technical Details

### Dependencies
- **DSPy**: Language model programming framework
- **Graph Engine**: State graph framework for complex workflows
- **OpenAI**: Language model provider

### Project Structure
```
dspygraph/                         # Reusable framework
├── base.py                        # Node base class
├── config.py                      # DSPy configuration
└── constants.py                   # Framework constants

examples/                          # Example applications
├── question_classifier_app/       # Question classifier example
│   ├── main.py                    # Main application entry point
│   ├── compile_classifier.py      # Compilation script
│   ├── workflow.py                # Graph workflow definition
│   ├── nodes.py                   # Node implementations
│   └── types.py                   # Application types
└── react_agent/                   # React agent with tools example
    ├── main.py                    # Interactive React agent
    ├── workflow.py                # Graph workflow with reasoning loops
    ├── nodes.py                   # React agent and tool executor nodes
    ├── tools.py                   # Calculator and search tools
    └── types.py                   # State and result types

simple_example.py                  # Basic framework demo
```

## Contributing

This project demonstrates patterns for:
- Clean architecture in AI systems
- DSPy best practices
- Graph integration
- Type-safe Python development

Feel free to use this as a template for your own DSPy + Graph projects!

## License

[Add your license here]