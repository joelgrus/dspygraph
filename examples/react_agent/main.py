"""
Main application for the React agent example
"""

import dspy

from .graph import create_react_agent_graph, demonstrate_react_agent
from .tools import get_available_tools


def main() -> None:
    """Main application entry point for React agent"""
    # Configure DSPy
    lm = dspy.LM("openai/gpt-4o-mini")
    dspy.configure(lm=lm)

    # Enable DSPy observability
    dspy.enable_logging()

    print("🤖 React Agent - Reasoning and Acting with Tools")
    print("=" * 55)

    # Show available tools
    tools = get_available_tools()
    print("\n🔧 Available Tools:")
    for name, tool in tools.items():
        print(f"  • {name}: {tool.description}")

    # Create the graph
    try:
        graph = create_react_agent_graph(max_steps=6)
    except Exception as e:
        print(f"Failed to create React agent graph: {e}")
        return

    # Print graph structure
    print("\n📊 Graph Structure:")
    print(graph.visualize())
    print()

    # Interactive mode
    print("🎯 React Agent is ready! (type 'demo' for demonstration, 'quit' to exit)")
    print("-" * 55)

    while True:
        try:
            user_input = input("\n💭 Enter your question: ").strip()

            if user_input.lower() in ["quit", "exit", "q"]:
                print("👋 Goodbye!")
                break
            elif user_input.lower() == "demo":
                print("\n🎬 Running demonstration...")
                demonstrate_react_agent()
                continue
            elif not user_input:
                continue

            print(f"\n🔍 Processing: {user_input}")
            print("-" * 40)

            # Execute graph with full observability
            result = graph.run(
                question=user_input,
                max_steps=6,
                step_count=0,
                thoughts=[],
                actions=[],
                observations=[],
            )

            print(
                f"\n✅ Final Answer: {result.get('final_answer', 'No answer provided')}"
            )

            # Show graph metadata
            metadata = result.get("_graph_metadata", {})
            print(f"⏱️  Execution time: {metadata.get('execution_time', 0):.3f}s")
            print(f"📊 Total usage: {metadata.get('total_usage', {})}")
            print(f"🔄 Steps: {result.get('step_count', 0)}")
            print(
                f"🎯 Execution path: {' → '.join(metadata.get('execution_order', []))}"
            )

        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Execution failed: {e}")


if __name__ == "__main__":
    main()
