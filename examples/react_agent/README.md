# React Agent Example

A DSPy graph-based implementation of a ReAct (Reasoning + Acting) agent that demonstrates iterative problem-solving through reasoning and tool execution.

## What This Example Does

This React agent showcases:

1. **Iterative Reasoning**: Uses DSPy's Chain-of-Thought for step-by-step problem analysis
2. **Tool Integration**: Executes tools (calculator, search) based on reasoning decisions
3. **Action Planning**: Determines what tools to use and when to finish
4. **Loop Control**: Manages the reasoning → acting → observing cycle with proper termination

### Architecture

```
START → ReactAgent → MaxStepsChecker → ToolExecutor → [Loop back or END]
```

The agent follows the ReAct pattern:
- **Reason**: Analyze the problem and decide on next action
- **Act**: Execute tools or provide final answer
- **Observe**: Process tool results and continue reasoning

## Example Interactions

```bash
$ python -m examples.react_agent.main

💭 Enter your question: What is 15 * 24 + 100?

🤔 Thought: I need to calculate 15 * 24 first, then add 100 to the result.
🎯 Action: calculate: 15 * 24
👀 Observation: Tool 'calculator' returned: 360

🤔 Thought: Now I need to add 100 to 360 to get the final answer.
🎯 Action: calculate: 360 + 100
👀 Observation: Tool 'calculator' returned: 460

🤔 Thought: I have calculated the complete expression. 15 * 24 = 360, and 360 + 100 = 460.
🎯 Action: finish: 460
✅ Final Answer: 460
```

## Quick Start

### Prerequisites
- Python 3.11+
- OpenAI API key (set as environment variable)
- dspygraph framework installed

### Running the Example

```bash
# Interactive mode
python -m examples.react_agent.main

# Or run the demonstration
python -m examples.react_agent.graph
```

## Available Tools

### Calculator Tool
- **Purpose**: Performs mathematical calculations
- **Input**: Mathematical expressions like `2 + 3`, `sqrt(16)`, `15 * 24`
- **Supported**: Basic arithmetic, sqrt, pow, sin, cos, tan, log, exp

### Search Tool  
- **Purpose**: Mock information retrieval
- **Input**: Search queries like `capital of France`, `population of Tokyo`
- **Knowledge**: Predefined responses for common topics

## How It Works

### 1. React Agent Node
Uses DSPy's `ChainOfThought` with a custom signature:
```python
class ReactReasoningSignature(dspy.Signature):
    question: str = dspy.InputField(desc="The question to solve")
    previous_steps: str = dspy.InputField(desc="Previous thoughts, actions, and observations")
    thought: str = dspy.OutputField(desc="Current reasoning step")
    action: str = dspy.OutputField(desc="Action to take")
```

### 2. Tool Executor Node
Parses agent actions and executes appropriate tools:
- `calculate: <expression>` → Calculator tool
- `search: <query>` → Search tool  
- `finish: <answer>` → Terminate with final answer

### 3. Graph Workflow
Implements the reasoning loop with conditional routing:
```python
def route_from_tool_executor(state):
    if state.get("final_answer"):
        return "end"  # Task completed
    return "continue"  # Continue reasoning loop
```

### 4. Loop Control
- **Max Steps**: Prevents infinite loops (default: 5 steps)
- **Natural Termination**: Agent decides when to finish
- **Error Handling**: Graceful handling of tool failures

## Example Use Cases

### Mathematical Problem Solving
```
Question: "What is the square root of 144 plus 5 times 3?"
Process:
1. Calculate sqrt(144) = 12
2. Calculate 5 * 3 = 15  
3. Calculate 12 + 15 = 27
4. Finish with answer: 27
```

### Information Retrieval with Calculation
```
Question: "What is the capital of France and what is 2 + 2?"
Process:
1. Search for capital of France → Paris
2. Calculate 2 + 2 → 4
3. Finish with combined answer
```

### Multi-Step Problem Solving
```
Question: "If I have $100 and spend 25% on food, how much do I have left?"
Process:
1. Calculate 25% of 100 → 25
2. Calculate 100 - 25 → 75
3. Finish with answer: $75
```

## File Structure

```
react_agent/
├── main.py                   # Interactive application
├── graph.py                  # Graph workflow definition  
├── nodes.py                  # React agent and tool executor nodes
├── tools.py                  # Calculator and search tools
├── types.py                  # Type definitions
└── README.md                 # This documentation
```

## Customization

### Adding New Tools

1. **Create Tool Class**:
```python
class MyCustomTool:
    def __init__(self):
        self.name = "my_tool"
        self.description = "Description of what the tool does"
    
    def execute(self, input_data: str) -> ToolResult:
        # Tool implementation
        return {"success": True, "result": "output", "error": None}
```

2. **Register in tools.py**:
```python
def get_available_tools():
    return {
        "calculator": CalculatorTool(),
        "search": SearchTool(),
        "my_tool": MyCustomTool()  # Add your tool
    }
```

3. **Update Agent Instructions**: The agent will automatically learn to use new tools based on their descriptions.

### Modifying Reasoning

The React agent's reasoning can be customized by:
- **Changing the Signature**: Modify prompts and field descriptions
- **Adjusting Context**: Change how previous steps are formatted
- **Adding Constraints**: Include rules or limitations in the signature

### Extending the Graph

Add new nodes for specialized processing:
```python
class ValidationNode(Node):
    """Validates tool results before continuing"""
    def process(self, state):
        # Custom validation logic
        return updated_state

# Add to graph
graph.add_node(ValidationNode("validator"))
graph.add_edge("tool_executor", "validator")
```

## Advanced Features

### State Management
The React agent maintains comprehensive state:
- **Reasoning History**: All thoughts, actions, and observations
- **Step Counting**: Tracks progress and prevents infinite loops
- **Context Building**: Provides rich context for decision making

### Error Recovery
- **Tool Failures**: Agent can reason about errors and try alternatives
- **Invalid Actions**: Graceful handling of malformed action commands
- **Max Steps**: Automatic termination with partial results

### Observability
- **Step-by-Step Output**: See reasoning process in real-time
- **Execution Metadata**: Timing, usage, and path information
- **DSPy Integration**: Full observability with DSPy's tracking

## Performance Considerations

- **Step Limits**: Balance thoroughness vs. efficiency
- **Tool Complexity**: Simple tools execute faster
- **Context Size**: Longer histories increase token usage
- **Caching**: Consider tool result caching for repeated queries

## Comparison with Question Classifier

| Feature | Question Classifier | React Agent |
|---------|-------------------|-------------|
| **Pattern** | Classification → Route → Response | Iterative Reasoning → Action → Observation |
| **Tools** | None (direct responses) | Calculator, Search, Extensible |
| **Steps** | Single pass | Multi-step with loops |
| **State** | Simple routing state | Rich reasoning history |
| **Use Cases** | Known question types | Open-ended problem solving |

Both examples demonstrate different strengths of the DSPy graph framework:
- **Question Classifier**: Efficient routing and specialized responses
- **React Agent**: Flexible reasoning and tool integration

## Contributing

This example demonstrates:
- Advanced graph patterns with loops
- Tool integration in DSPy workflows  
- State management across iterations
- Conditional routing and termination logic

Use this as a foundation for building more sophisticated reasoning agents!