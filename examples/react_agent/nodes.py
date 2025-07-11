"""
React agent nodes using DSPy and the graph framework
"""

from typing import Any

import dspy

from dspygraph import Node

from .tools import execute_tool, get_available_tools


class ReactReasoningSignature(dspy.Signature):
    """Signature for React reasoning - generates thought and action"""

    question: str = dspy.InputField(desc="The question to solve")
    previous_steps: str = dspy.InputField(
        desc="Previous thoughts, actions, and observations"
    )
    thought: str = dspy.OutputField(
        desc="Current reasoning step - what you're thinking about the problem"
    )
    action: str = dspy.OutputField(
        desc="Action to take: 'calculator: <expression>' or 'search: <query>' or 'finish: <final_answer>'"
    )


class ReactAgentNode(Node):
    """React agent that reasons and acts iteratively"""

    def _create_module(self) -> dspy.Module:
        # Create a dynamic signature with available tools
        tools = get_available_tools()
        tool_descriptions = []
        for name, tool in tools.items():
            tool_descriptions.append(f"'{name}: <input>' - {tool.description}")

        tools_text = " or ".join(tool_descriptions)
        action_desc = f"Action to take: {tools_text} or 'finish: <final_answer>'"

        # Create signature with dynamic tool information
        class DynamicReactSignature(dspy.Signature):
            """Signature for React reasoning with available tools"""

            question: str = dspy.InputField(desc="The question to solve")
            previous_steps: str = dspy.InputField(
                desc="Previous thoughts, actions, and observations"
            )
            thought: str = dspy.OutputField(
                desc="Current reasoning step - what you're thinking about the problem"
            )
            action: str = dspy.OutputField(desc=action_desc)

        return dspy.ChainOfThought(DynamicReactSignature)

    def process(self, state: dict[str, Any]) -> dict[str, Any]:
        """Process one step of React reasoning"""

        # Build context from previous steps
        previous_steps = self._build_context(state)

        # Get reasoning from DSPy module
        result = self.module(question=state["question"], previous_steps=previous_steps)

        print(f"  🤔 Thought: {result.thought}")
        print(f"  🎯 Action: {result.action}")

        # Update state with new thought and action
        new_state = {
            "current_thought": result.thought,
            "current_action": result.action,
            "step_count": state.get("step_count", 0) + 1,
        }

        # Add to history
        if "thoughts" not in state:
            new_state["thoughts"] = []
            new_state["actions"] = []
            new_state["observations"] = []
        else:
            new_state["thoughts"] = state["thoughts"] + [result.thought]
            new_state["actions"] = state["actions"] + [result.action]

        return new_state

    def _build_context(self, state: dict[str, Any]) -> str:
        """Build context string from previous steps"""
        if not state.get("thoughts"):
            return "This is the first step."

        context_parts = []

        for i, (thought, action, obs) in enumerate(
            zip(
                state.get("thoughts", []),
                state.get("actions", []),
                state.get("observations", []),
                strict=False,
            )
        ):
            context_parts.append(f"Step {i + 1}:")
            context_parts.append(f"  Thought: {thought}")
            context_parts.append(f"  Action: {action}")
            context_parts.append(f"  Observation: {obs}")

        return "\n".join(context_parts)


class ToolExecutorNode(Node):
    """Executes tools based on the agent's action"""

    def _create_module(self) -> dspy.Module:
        # This node doesn't use DSPy, just returns a mock module
        return dspy.Predict("dummy -> dummy")

    def process(self, state: dict[str, Any]) -> dict[str, Any]:
        """Execute the requested tool action"""
        action = state.get("current_action", "")

        # Parse the action
        tool_name, tool_input = self._parse_action(action)

        if tool_name == "finish":
            print(f"  ✅ Final Answer: {tool_input}")
            return {
                "final_answer": tool_input,
                "current_observation": f"Task completed with answer: {tool_input}",
            }

        # Execute the tool
        tool_result = execute_tool(tool_name, tool_input)

        if tool_result["success"]:
            observation = f"Tool '{tool_name}' returned: {tool_result['result']}"
            print(f"  👀 Observation: {observation}")
        else:
            observation = f"Tool '{tool_name}' failed: {tool_result['error']}"
            print(f"  ❌ Error: {observation}")

        # Update observations history
        new_observations = state.get("observations", []) + [observation]

        return {"current_observation": observation, "observations": new_observations}

    def _parse_action(self, action: str) -> tuple[str, str]:
        """Parse action string into tool name and input"""
        action = action.strip()

        # Look for patterns like "calculate: 2 + 3" or "search: capital of france"
        if ":" in action:
            parts = action.split(":", 1)
            tool_name = parts[0].strip().lower()
            tool_input = parts[1].strip()
            return tool_name, tool_input

        # Handle "finish" without colon
        if action.lower().startswith("finish"):
            return "finish", action[6:].strip()

        # Default - treat whole action as finish
        return "finish", action


class MaxStepsNode(Node):
    """Checks if we've reached maximum steps"""

    def _create_module(self) -> dspy.Module:
        return dspy.Predict("dummy -> dummy")

    def process(self, state: dict[str, Any]) -> dict[str, Any]:
        """Check if we've reached max steps"""
        step_count = state.get("step_count", 0)
        max_steps = state.get("max_steps", 5)

        if step_count >= max_steps:
            print(f"  ⚠️  Reached maximum steps ({max_steps}). Forcing completion.")
            return {
                "final_answer": "I've reached the maximum number of reasoning steps. Based on my analysis so far, I may need more information or a different approach to fully solve this problem.",
                "current_observation": f"Reached maximum steps limit ({max_steps})",
            }

        return {}
