"""
Metrics for evaluating question classification
"""
from typing import Any, Optional

def classification_metric(pred_object: Any, true_category_str: str, trace: Optional[Any] = None) -> bool:
    """
    Metric for evaluating question classification accuracy
    
    Args:
        pred_object: The prediction object from DSPy
        true_category_str: The true category string
        trace: Optional trace information
        
    Returns:
        True if classification is correct, False otherwise
    """
    return pred_object.category == true_category_str