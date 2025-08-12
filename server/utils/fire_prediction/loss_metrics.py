import numpy as np
from tqdm import tqdm
import torch 
import torch.nn as nn
import torch.nn.functional as F


# Generalized Dice Loss for multiclass segmentation with unbalanced classes
class GeneralizedDiceLoss(nn.Module):
    """Generalized Dice Loss with inverse frequency weighting
    Based on: https://arxiv.org/pdf/1707.03237

    Args:
        predict: A tensor of logits, shape (batch_size, n_classes, height, width) output by the model
        target: A tensor of shape (batch_size, height, width) containing class indices
        class_weights: Optional tensor of shape (n_classes,) with weights for each class
    """
    def __init__(self, class_weights=None):
        super().__init__()
        self.smooth = 1e-6  # parameter to avoid division by 0
        self.class_weights = class_weights

    def forward(self, predict, target):
        # predict shape: (batch_size, n_classes, height, width)
        # target shape: (batch_size, height, width)

        batch_size, n_classes, height, width = predict.shape
        
        # get probabilities from logits
        predict = torch.softmax(predict, dim=1)
        
        # convert target to one-hot encoding
        target_one_hot = F.one_hot(target, num_classes=n_classes).permute(0, 3, 1, 2).float()   # permute to obtain (batch_size, n_classes, height, width)
        
        # flatten tensors for easier computation
        predict = predict.view(batch_size, n_classes, -1)  # (batch_size, n_classes, H*W)
        target_one_hot = target_one_hot.view(batch_size, n_classes, -1)  # (batch_size, n_classes, H*W)
        
        # compute intersection and union for each class
        intersection = (predict * target_one_hot).sum(dim=2)  # (batch_size, n_classes)
        predict_sum = predict.sum(dim=2)  # (batch_size, n_classes)
        target_sum = target_one_hot.sum(dim=2)  # (batch_size, n_classes)
        
        # apply class weights if provided (Generalized Dice Loss)
        if self.class_weights is not None:
            # weights shape: (n_classes,) -> (1, n_classes) for broadcasting
            weights = self.class_weights.unsqueeze(0).to(predict.device)
            
            # Apply weights to both numerator and denominator as per paper
            weighted_intersection = (weights * intersection).sum(dim=1)  # (batch_size,)
            weighted_union = (weights * (predict_sum + target_sum)).sum(dim=1)  # (batch_size,)
            
            # Generalized dice coefficient
            dice = (2.0 * weighted_intersection + self.smooth) / (weighted_union + self.smooth)
        else:
            # standard unweighted dice loss
            union = predict_sum + target_sum
            dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
            dice = dice.mean(dim=1)  # average across classes
        
        # dice loss is 1 - dice coefficient, averaged across batch
        dice_loss = 1.0 - dice.mean()
        
        return dice_loss


# class weights calculation needed for weighted GDLoss
def calculate_class_weights(train_loader, n_classes=3):
    """Calculate inverse frequency weights normalized to sum to 1"""
    print("Calculating class weights from training data...")
    class_counts = torch.zeros(n_classes)
    
    for x, y, coords in tqdm(train_loader, desc="Computing class frequencies"):
        y_flat = y.flatten()
        unique, counts = torch.unique(y_flat, return_counts=True)
        for cls, count in zip(unique, counts):
            if cls < n_classes:  # safety check
                class_counts[cls] += count.item()
    
    total_pixels = class_counts.sum()
    class_frequencies = class_counts / total_pixels
    
    # Inverse frequency weights
    inverse_weights = 1.0 / (class_frequencies + 1e-6)  # add small epsilon to avoid division by zero
    
    # Normalize weights to sum to 1
    normalized_weights = inverse_weights / inverse_weights.sum()
    
    print(f"Class frequencies: {class_frequencies}")
    print(f"Inverse weights (normalized): {normalized_weights}")
    
    return normalized_weights


# mean IOU calculation helper function
def compute_mean_iou(y_true, y_pred, n_classes=3):
    """
    Compute mean Intersection over Union (IoU) for multi-class segmentation
    
    Args:
        y_true: True labels (flattened)
        y_pred: Predicted labels (flattened) 
        n_classes: Number of classes
        
    Returns:
        mean_iou: Mean IoU across all classes
    """
    ious = []
    
    for class_id in range(n_classes):
        # true positives, false positives, false negatives
        tp = np.sum((y_true == class_id) & (y_pred == class_id))
        fp = np.sum((y_true != class_id) & (y_pred == class_id))
        fn = np.sum((y_true == class_id) & (y_pred != class_id))
        
        # IoU = TP / (TP + FP + FN)
        iou = tp / (tp + fp + fn)
        
        ious.append(iou)
    
    return np.mean(ious)


# sensitivity (recall) for the minority classes, to verify the detection of risky areas
def compute_sensitivity_classes_1_2(y_true, y_pred):
    """
    Compute sensitivity (recall) for classes 1 and 2, averaged
    
    Args:
        y_true: True labels (flattened)
        y_pred: Predicted labels (flattened)
        
    Returns:
        avg_sensitivity: Average sensitivity for classes 1 and 2
    """
    sensitivities = []
    
    for class_id in [1, 2]:
        # True positives and false negatives
        tp = np.sum((y_true == class_id) & (y_pred == class_id))
        fn = np.sum((y_true == class_id) & (y_pred != class_id))
        
        # Sensitivity = TP / (TP + FN)
        if tp + fn > 0:
            sensitivity = tp / (tp + fn)
        else:
            sensitivity = 0.0  # No true positives exist for this class
            
        sensitivities.append(sensitivity)
    
    return np.mean(sensitivities)
