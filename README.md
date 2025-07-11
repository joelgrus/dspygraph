# DSPy Graph Framework

A lightweight framework for building graph-based workflows with DSPy nodes. Combine DSPy's powerful language model programming with flexible graph execution, conditional routing, and state management.

## Installation

```bash
pip install dspygraph
```

## Quick Start

```python
import dspy
from dspygraph import Node, Graph, START, END

# Configure DSPy
lm = dspy.LM("openai/gpt-4o-mini")
dspy.configure(lm=lm)

# Create a simple node
class QuestionAnswerNode(Node):
    def _create_module(self):
        return dspy.ChainOfThought("question -> answer")
    
    def process(self, state):
        result = self.module(question=state["question"])
        return {"answer": result.answer}

# Create and run a graph
graph = Graph("MyGraph")
graph.add_node(QuestionAnswerNode("qa"))
graph.add_edge(START, "qa")
graph.add_edge("qa", END)

result = graph.run(question="What is the capital of France?")
print(result["answer"])
```

## Core Concepts

### Node
The base class for all graph nodes. Extend it to create custom DSPy-powered components:

```python
class MyNode(Node):
    def _create_module(self):
        # Return any DSPy module
        return dspy.ChainOfThought("input -> output")
    
    def process(self, state):
        # Process the state and return updates
        result = self.module(input=state["input"])
        return {"output": result.output}
```

### Graph
The execution engine that manages nodes and their connections:

```python
graph = Graph("MyGraph")
graph.add_node(my_node)
graph.add_edge(START, "my_node")
graph.add_edge("my_node", END)

# Conditional routing
graph.add_conditional_edges(
    "classifier",
    {"route_a": "node_a", "route_b": "node_b"},
    lambda state: "route_a" if state["condition"] else "route_b"
)
```

## Features

- **🔗 Graph-based execution**: Build complex workflows with conditional routing and cycles
- **🤖 DSPy integration**: Seamlessly integrate with DSPy's language model programming
- **🔄 State management**: Automatic state passing between nodes with full observability
- **⚡ Flexible routing**: Support for conditional edges and dynamic graph execution
- **🛡️ Error handling**: Built-in protection against infinite loops and execution failures
- **📊 Observability**: Complete execution tracking with timing, token usage, and metadata

## Example Applications

This repository includes complete example applications that demonstrate the framework's capabilities:

### 1. Question Classifier System
An intelligent agent that:
- **Classifies** incoming questions into categories (factual, creative, tool-use, or unknown)
- **Routes** each question to the most appropriate specialized response module  
- **Generates** tailored responses using different reasoning patterns for each category

### 2. ReAct Agent
A reasoning and acting agent that:
- Uses iterative reasoning with tool execution
- Demonstrates graph-based loops and state management
- Includes calculator and search tools

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
python -m examples.react_agent.graph
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
│   ├── graph.py                   # Graph workflow definition
│   ├── nodes.py                   # Node implementations
│   └── types.py                   # Application types
└── react_agent/                   # React agent with tools example
    ├── main.py                    # Interactive React agent
    ├── graph.py                   # Graph workflow with reasoning loops
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