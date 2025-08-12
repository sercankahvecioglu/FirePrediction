import numpy as np
from scipy.ndimage import binary_closing

def heatmap_smoothing(predictions, kernel_size=5):
    """
    Apply morphological closing to fire risk predictions to create more realistic
    spatially coherent risk zones.
    
    Args:
        predictions: numpy array with fire risk predictions (0, 1, 2)
        kernel_size: size of morphological kernel
    
    Returns:
        numpy array with smoothed predictions
    """
    processed_preds = predictions.copy()
    kernel = np.ones((kernel_size, kernel_size))
    
    # clean moderate and high risk areas separately
    for risk_class in [1, 2]:
        class_mask = (predictions == risk_class)
        if np.any(class_mask):  # only process if class exists
            cleaned_mask = binary_closing(class_mask, structure=kernel)
            processed_preds[cleaned_mask] = risk_class
    
    # ensure high risk overwrites moderate if areas are overlapping
    high_risk_mask = (predictions == 2)
    if np.any(high_risk_mask):
        cleaned_high = binary_closing(high_risk_mask, structure=kernel)
        processed_preds[cleaned_high] = 2
    
    return processed_preds