"""
DSPy configuration utilities
"""
import dspy
from functools import lru_cache
from typing import Optional

@lru_cache(maxsize=1)
def get_lm(model_name: str = "openai/gpt-4o-mini") -> dspy.LM:
    """
    Get and configure the language model for DSPy
    
    Args:
        model_name: Name of the model to use
        
    Returns:
        Configured DSPy language model
    """
    lm = dspy.LM(model_name)
    dspy.configure(lm=lm)
    return lm

def configure_dspy(model_name: str = "openai/gpt-4o-mini") -> None:
    """
    Configure DSPy with the specified model
    
    Args:
        model_name: Name of the model to use
    """
    get_lm(model_name)