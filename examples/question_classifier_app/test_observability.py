"""
Test DSPy's built-in observability features with our graph framework
"""
import dspy
from dspy.utils.callback import BaseCallback
from .graph import create_question_classifier_graph


class DetailedCallback(BaseCallback):
    """Custom callback to demonstrate DSPy's observability hooks"""
    
    def on_module_start(self, call_id, instance, inputs):
        print(f"📝 Module Start - Call ID: {call_id[:8]}, Module: {type(instance).__name__}")
        print(f"   Inputs: {list(inputs.keys()) if isinstance(inputs, dict) else 'N/A'}")
    
    def on_module_end(self, call_id, outputs, exception):
        if exception:
            print(f"❌ Module Failed - Call ID: {call_id[:8]}, Error: {exception}")
        else:
            print(f"✅ Module Complete - Call ID: {call_id[:8]}")
    
    def on_lm_start(self, call_id, instance, inputs):
        print(f"🤖 LM Call Start - Call ID: {call_id[:8]}, Model: {getattr(instance, 'model', 'Unknown')}")
    
    def on_lm_end(self, call_id, outputs, exception):
        if exception:
            print(f"❌ LM Call Failed - Call ID: {call_id[:8]}, Error: {exception}")
        else:
            print(f"✅ LM Call Complete - Call ID: {call_id[:8]}")


def test_observability():
    """Test the graph with full DSPy observability enabled"""
    
    # Configure DSPy with observability
    lm = dspy.LM("openai/gpt-4o-mini")
    dspy.configure(lm=lm)
    
    # Enable logging and callbacks
    dspy.enable_logging()
    callback = DetailedCallback()
    dspy.settings.configure(callbacks=[callback])
    
    print("🔬 Testing DSPy Graph Framework with Full Observability")
    print("=" * 60)
    
    # Create graph
    graph = create_question_classifier_graph()
    
    # Test with usage tracking
    question = "What is the meaning of life?"
    
    print(f"\n🤔 Question: {question}")
    print("-" * 40)
    
    with dspy.track_usage() as usage:
        result = graph.run(question=question)
    
    print("\n📊 Usage Summary:")
    print(f"Total Usage: {usage.get_total_tokens()}")
    
    # Inspect DSPy history
    print("\n📜 DSPy History (last 3 interactions):")
    try:
        history = dspy.inspect_history(3)
        if history:
            for i, entry in enumerate(history):
                print(f"  {i+1}. Model: {entry.get('model', 'Unknown')}")
                print(f"     Usage: {entry.get('usage', {})}")
                print(f"     Cost: ${entry.get('cost', 0):.6f}")
                print(f"     Timestamp: {entry.get('timestamp', 'Unknown')}")
                print()
        else:
            print("  No history available")
    except Exception as e:
        print(f"  Could not retrieve history: {e}")
    
    print(f"✨ Final Answer: {result.get('response', 'No response')}")


if __name__ == "__main__":
    test_observability()