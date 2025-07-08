import dspy
from langgraph.graph import StateGraph, END
from typing import TypedDict, Literal
import os

# 1. Define your DSPy Modules (the "intelligence" components)
# Define a Signature for the QuestionClassifier
class QuestionClassificationSignature(dspy.Signature):
    """Classifies a user's question into a specific category."""
    question: str = dspy.InputField(desc="The user's question")
    category: Literal['factual', 'creative', 'tool_use', 'unknown'] = dspy.OutputField(
        desc="Classification category of the question (e.g., 'factual', 'creative', 'tool_use', 'unknown')"
    )


class QuestionClassifier(dspy.Module):
    def __init__(self):
        super().__init__()
        self.classify = dspy.Predict(QuestionClassificationSignature)

    def forward(self, question):
        return self.classify(question=question) # Returns the full prediction object


class ToolUseModule(dspy.Module):
    def __init__(self):
        super().__init__()
        self.react = dspy.ReAct("query -> answer", tools=[self.dummy_tool])

    def dummy_tool(self, query: str) -> str:
        return f"Executing dummy tool for query: {query}"

    def forward(self, query):
        return self.react(query=query) # Return the full prediction object

class FactualAnswerModule(dspy.Module):
    def __init__(self):
        super().__init__()
        self.generate_answer = dspy.ChainOfThought("question -> answer")

    def forward(self, question):
        return self.generate_answer(question=question) # Return the full prediction object


class CreativeResponseSignature(dspy.Signature):
    prompt: str = dspy.InputField(desc="The creative prompt")
    creative_output: str = dspy.OutputField(desc="The creative response")


class CreativeResponseModule(dspy.Module):
    def __init__(self):
        super().__init__()
        self.generate_creative = dspy.ChainOfThought(CreativeResponseSignature) # Use CreativeResponseSignature
    def forward(self, prompt):
        return self.generate_creative(prompt=prompt)


# 2. Define the State for LangGraph
class AgentState(TypedDict):
    question: str
    classification: str # This will now store the 'category' string
    response: str
    tool_output: str # This might be None initially

# 3. Initialize DSPy models (configure your LLM)
lm = dspy.LM("openai/gpt-4o-mini")
dspy.configure(lm=lm)

# 4. Define the Nodes for LangGraph
classifier_node = QuestionClassifier()
tool_node = ToolUseModule()
factual_node = FactualAnswerModule()
creative_node = CreativeResponseModule()

compiled_classifier_path = "compiled_classifier.json"

# --- Production-ready loading only ---
if not os.path.exists(compiled_classifier_path):
    raise FileNotFoundError(
        f"Compiled classifier not found at {compiled_classifier_path}. "
        "Please run the compilation script (e.g., 'python compile_classifier.py') first."
    )

print(f"Loading compiled classifier from {compiled_classifier_path}...")
classifier_node.load(compiled_classifier_path)
print("Compiled classifier loaded.")


def classify_question(state: AgentState):
    question = state["question"]
    prediction_object = classifier_node(question=question)
    # Access the 'category' field, as per the updated signature
    classification_result = prediction_object.category 
    print(f"Question: '{question}' classified as: '{classification_result}'")
    return {"classification": classification_result}

def use_tool(state: AgentState):
    question = state["question"]
    tool_prediction = tool_node(query=question)
    tool_output = tool_prediction.answer
    print(f"Tool used for: '{question}', output: '{tool_output}'")
    return {"tool_output": tool_output, "response": tool_output}

def get_factual_answer(state: AgentState):
    question = state["question"]
    factual_prediction = factual_node(question=question)
    answer = factual_prediction.answer
    print(f"Factual answer for: '{question}', answer: '{answer}'")
    return {"response": answer}

def get_creative_response(state: AgentState):
    prompt = state["question"]
    creative_prediction = creative_node(prompt=prompt)
    creative_output = creative_prediction.creative_output
    print(f"Creative response for: '{prompt}', response: '{creative_output}'")
    return {"response": creative_output}

# 5. Define the Conditional Edge (Router)
def route_question(state: AgentState):
    classification = state["classification"]
    if "tool_use" in classification:
        return "tool_use_path"
    elif "factual" in classification:
        return "factual_path"
    elif "creative" in classification:
        return "creative_path"
    else:
        print(f"No specific path for classification: {classification}. Ending.")
        return END

# 6. Build the LangGraph
workflow = StateGraph(AgentState)

workflow.add_node("classify", classify_question)
workflow.add_node("tool_use", use_tool)
workflow.add_node("factual_answer", get_factual_answer)
workflow.add_node("creative_response", get_creative_response)

workflow.set_entry_point("classify")

workflow.add_conditional_edges(
    "classify",
    route_question,
    {
        "tool_use_path": "tool_use",
        "factual_path": "factual_answer",
        "creative_path": "creative_response",
        END: END,
    },
)

workflow.add_edge("tool_use", END)
workflow.add_edge("factual_answer", END)
workflow.add_edge("creative_response", END)

app = workflow.compile()

# 7. Run the agent
print("\n--- Running Agent: Factual Question ---")
result_factual = app.invoke({"question": "What is the capital of France?"})
print(f"Final Result: {result_factual['response']}")

print("\n--- Running Agent: Tool Use Question ---")
result_tool = app.invoke({"question": "What is 123 + 456?"})
print(f"Final Result: {result_tool['response']}")

print("\n--- Running Agent: Creative Question (Unhandled Path) ---")
result_creative = app.invoke({"question": "Write a short poem about a cat."})
print(f"Final Result: {result_creative.get('response', 'No specific response due to unhandled classification.')}")