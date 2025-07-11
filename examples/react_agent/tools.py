"""
Simple tools for the React agent
"""
import re
import math
from typing import Dict, Any
from .types import ToolResult


class CalculatorTool:
    """Simple calculator tool for mathematical expressions"""
    
    def __init__(self):
        self.name = "calculator"
        self.description = "Performs mathematical calculations. Input should be a mathematical expression like '2 + 3' or 'sqrt(16)'"
    
    def execute(self, expression: str) -> ToolResult:
        """Execute a mathematical calculation"""
        try:
            # Clean the expression
            expression = expression.strip()
            
            # Simple safety check - only allow certain characters
            allowed_chars = set('0123456789+-*/.() sqrt,pow,sin,cos,tan,log,exp')
            if not all(c in allowed_chars or c.isspace() for c in expression):
                return {
                    "success": False,
                    "result": "",
                    "error": "Invalid characters in expression"
                }
            
            # Replace common function names with math module equivalents
            expression = expression.replace('sqrt', 'math.sqrt')
            expression = expression.replace('pow', 'math.pow')
            expression = expression.replace('sin', 'math.sin')
            expression = expression.replace('cos', 'math.cos')
            expression = expression.replace('tan', 'math.tan')
            expression = expression.replace('log', 'math.log')
            expression = expression.replace('exp', 'math.exp')
            
            # Evaluate the expression
            result = eval(expression, {"__builtins__": {}, "math": math})
            
            return {
                "success": True,
                "result": str(result),
                "error": None
            }
            
        except Exception as e:
            return {
                "success": False,
                "result": "",
                "error": f"Calculation error: {str(e)}"
            }


class SearchTool:
    """Mock search tool that provides predefined answers to common questions"""
    
    def __init__(self):
        self.name = "search"
        self.description = "Searches for information on the internet. Provide a search query as input."
        
        # Mock knowledge base
        self.knowledge_base = {
            "capital of france": "Paris is the capital of France.",
            "population of tokyo": "Tokyo has a population of approximately 14 million people in the city proper and 38 million in the greater metropolitan area.",
            "height of mount everest": "Mount Everest is 8,848.86 meters (29,031.7 feet) tall.",
            "speed of light": "The speed of light in a vacuum is approximately 299,792,458 meters per second.",
            "python programming": "Python is a high-level, interpreted programming language known for its simplicity and readability.",
            "artificial intelligence": "Artificial Intelligence (AI) is the simulation of human intelligence in machines that are programmed to think and learn.",
            "climate change": "Climate change refers to long-term shifts in global temperatures and weather patterns, primarily caused by human activities.",
            "quantum physics": "Quantum physics is the branch of physics that studies matter and energy at the smallest scales, typically atoms and subatomic particles."
        }
    
    def execute(self, query: str) -> ToolResult:
        """Execute a search query"""
        try:
            query_lower = query.lower().strip()
            
            # Try to find a match in our knowledge base
            for key, value in self.knowledge_base.items():
                if key in query_lower or any(word in query_lower for word in key.split()):
                    return {
                        "success": True,
                        "result": value,
                        "error": None
                    }
            
            # Default response for unknown queries
            return {
                "success": True,
                "result": f"I found some general information about '{query}', but I don't have specific details in my knowledge base. This is a mock search tool with limited information.",
                "error": None
            }
            
        except Exception as e:
            return {
                "success": False,
                "result": "",
                "error": f"Search error: {str(e)}"
            }


def get_available_tools() -> Dict[str, Any]:
    """Get all available tools as a dictionary"""
    return {
        "calculator": CalculatorTool(),
        "search": SearchTool()
    }


def execute_tool(tool_name: str, input_data: str) -> ToolResult:
    """Execute a tool by name"""
    tools = get_available_tools()
    
    if tool_name not in tools:
        return {
            "success": False,
            "result": "",
            "error": f"Unknown tool: {tool_name}. Available tools: {list(tools.keys())}"
        }
    
    return tools[tool_name].execute(input_data)