# Question Classifier App

An intelligent question-answering system that demonstrates advanced DSPy graph integration with classification, routing, and specialized response modules.

## What This App Does

This application creates an intelligent agent that:

1. **Classifies** incoming questions into categories:
   - `factual`: Factual questions requiring accurate information
   - `creative`: Creative requests like poems, stories, jokes
   - `tool_use`: Computational tasks requiring calculations
   - `unknown`: Questions that don't fit other categories

2. **Routes** each question to the most appropriate specialized response module

3. **Generates** tailored responses using different reasoning patterns for each category

## Example Interactions

```bash
$ python main.py

--- Running Agent: Factual Question ---
Question: 'What is the capital of France?' classified as: 'factual'
Final Result: The capital of France is Paris. It has been the capital since 987 AD...

--- Running Agent: Tool Use Question ---
Question: 'What is 123 + 456?' classified as: 'tool_use'
Final Result: I need to calculate 123 + 456. Let me compute this: 123 + 456 = 579

--- Running Agent: Creative Question ---
Question: 'Write a short poem about a cat.' classified as: 'creative'
Final Result: Whiskers twitching in the light, A feline friend both day and night...
```

## Quick Start

### Prerequisites
- Python 3.11+
- OpenAI API key (set as environment variable)
- dspygraph framework installed

### Running the App

```bash
# 1. Compile the classifier (required first time)
python -m examples.question_classifier_app.compile_classifier

# 2. Run the main application
python -m examples.question_classifier_app.main
```

## How It Works

### Architecture

```
User Question → QuestionClassifier → Router → Specialized Module → Response
```

### Components

#### 1. Question Classifier (`agents/classifier.py`)
- Uses DSPy's `Predict` module with optimized prompts
- Trained on example questions for each category
- Outputs one of four categories: factual, creative, tool_use, unknown

#### 2. Specialized Response Modules

**Factual Module** (`agents/factual.py`)
- Uses `ChainOfThought` reasoning for accuracy
- Optimized for providing factual information

**Creative Module** (`agents/creative.py`)
- Uses `Predict` for creative content generation
- Optimized for generating poems, stories, jokes

**Tool Use Module** (`agents/tool_use.py`)
- Uses `ReAct` pattern for computational tasks
- Handles calculations, conversions, and tool-based queries

#### 3. Graph Workflow (`graph.py`)
- Manages state flow between classification and response
- Handles routing logic based on classification results
- Provides error handling and fallback responses

### Compilation Process

The system uses DSPy's compilation system for optimization:

```python
# 1. Create training data
trainset = [
    ("What is the capital of France?", "factual"),
    ("Write a haiku about autumn", "creative"),
    ("What is 5 + 7?", "tool_use"),
    # ... more examples
]

# 2. Compile with BootstrapFewShot
compiler = BootstrapFewShot(metric=classification_metric)
classifier.compile(compiler, trainset, compile_path="compiled_classifier.json")
```

## File Structure

```
question_classifier_app/
├── main.py                    # Main application entry point
├── compile_classifier.py      # Compilation script
├── graph.py                   # Graph workflow definition
├── types.py                   # TypedDict definitions
├── agents/                    # Agent implementations
│   ├── classifier.py          # Question classification
│   ├── factual.py            # Factual response module
│   ├── creative.py           # Creative response module
│   └── tool_use.py           # Tool-based response module
└── compiled_classifier.json   # Compiled model (generated)
```

## Customization

### Adding New Question Categories

1. **Update Types**: Add new category to `QuestionCategory` in `types.py`
2. **Training Data**: Add examples in `compile_classifier.py`
3. **Create Agent**: Add new agent module in `agents/`
4. **Update Routing**: Modify routing logic in `graph.py`
5. **Recompile**: Run `python compile_classifier.py`

### Modifying Response Modules

Each response module is a standard DSPy graph `Node`:

```python
class CustomResponseModule(Node):
    def _create_module(self) -> dspy.Module:
        return dspy.ChainOfThought("question -> response")
    
    def _process_state(self, state: AgentState) -> Dict[str, Any]:
        # Your custom logic here
        return {"response": "Custom response"}
```

## Training Data

The classifier is trained on diverse examples:

- **Factual**: Geography, history, science questions
- **Creative**: Poetry, stories, jokes, creative writing
- **Tool Use**: Math, calculations, conversions
- **Unknown**: Ambiguous or out-of-scope questions

## Performance

- **Classification Accuracy**: Optimized through DSPy compilation
- **Response Quality**: Specialized modules for each question type
- **Compilation Time**: ~30 seconds for initial training
- **Runtime**: Fast inference with compiled models

## Extending the System

This app serves as a template for building more complex routing systems:

- Add new question categories
- Implement different response strategies  
- Integrate with external tools and APIs
- Add memory and conversation history
- Implement user feedback loops

The modular design makes it easy to experiment with different DSPy modules, compilation strategies, and routing logic.