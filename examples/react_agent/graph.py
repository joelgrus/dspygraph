"""
Graph-based React agent workflow
"""

import dspy

from dspygraph import END, START, Graph

from .nodes import MaxStepsNode, ReactAgentNode, ToolExecutorNode
from .tools import get_available_tools


def create_react_agent_graph(max_steps: int = 5) -> Graph:
    """
    Create the React agent graph with reasoning loop

    Args:
        max_steps: Maximum number of reasoning steps before forcing completion

    Returns:
        Configured Graph ready to run
    """
    # Create graph
    graph = Graph("ReactAgent")

    # Create nodes
    react_agent = ReactAgentNode("react_agent")
    tool_executor = ToolExecutorNode("tool_executor")
    max_steps_checker = MaxStepsNode("max_steps_checker")

    # Add nodes to graph
    graph.add_node(react_agent)
    graph.add_node(tool_executor)
    graph.add_node(max_steps_checker)

    # Start with React agent
    graph.add_edge(START, "react_agent")

    # After reasoning, check max steps
    graph.add_edge("react_agent", "max_steps_checker")

    # Route from max steps checker
    def route_from_max_steps(state):
        """Route based on whether we've hit max steps"""
        if state.get("final_answer"):
            return "end"  # Max steps reached, force end
        return "tool_executor"  # Continue to tool execution

    graph.add_conditional_edges(
        "max_steps_checker",
        {"end": END, "tool_executor": "tool_executor"},
        route_from_max_steps,
    )

    # Route from tool executor based on action type
    def route_from_tool_executor(state):
        """Route based on whether we have a final answer"""
        if state.get("final_answer"):
            return "end"  # Task completed
        return "continue"  # Continue reasoning loop

    graph.add_conditional_edges(
        "tool_executor",
        {"end": END, "continue": "react_agent"},
        route_from_tool_executor,
    )

    return graph


def run_react_agent(question: str, max_steps: int = 5) -> dict:
    """
    Run the React agent on a single question

    Args:
        question: The question for the agent to solve
        max_steps: Maximum reasoning steps

    Returns:
        Complete graph execution result
    """
    # Configure DSPy
    lm = dspy.LM("openai/gpt-4o-mini")
    dspy.configure(lm=lm)

    # Create and run graph
    graph = create_react_agent_graph(max_steps=max_steps)

    # Execute with the question
    result = graph.run(
        question=question,
        max_steps=max_steps,
        step_count=0,
        thoughts=[],
        actions=[],
        observations=[],
    )

    return result


def demonstrate_react_agent():
    """Demonstrate the React agent with various question types"""

    print("🤖 React Agent Demonstration")
    print("=" * 50)

    # Show available tools
    tools = get_available_tools()
    print("\n🔧 Available Tools:")
    for name, tool in tools.items():
        print(f"  - {name}: {tool.description}")

    test_questions = [
        "What is 15 * 24 + 100?",
        "What is the capital of France and what is 2 + 2?",
        "Calculate the square root of 144 and tell me about Python programming",
    ]

    for i, question in enumerate(test_questions, 1):
        print(f"\n{'=' * 50}")
        print(f"🔍 Question {i}: {question}")
        print("-" * 50)

        try:
            result = run_react_agent(question, max_steps=6)

            print(
                f"\n✅ Final Answer: {result.get('final_answer', 'No final answer provided')}"
            )

            # Show execution metadata
            metadata = result.get("_graph_metadata", {})
            print(f"⏱️  Steps taken: {result.get('step_count', 0)}")
            print(
                f"🔄 Execution path: {' → '.join(metadata.get('execution_order', []))}"
            )

        except Exception as e:
            print(f"❌ Error: {e}")


if __name__ == "__main__":
    demonstrate_react_agent()
