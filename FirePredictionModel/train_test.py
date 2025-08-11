import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split  
from models.unet import UNet
import numpy as np
from PIL import Image
from tqdm import tqdm
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import glob
import os   
import pickle
import matplotlib
from models.geodata_extraction import *
from models.datasets import Sent2Dataset


class MulticlassDiceLoss(nn.Module):
    """Multiclass Dice Loss for Pytorch

    Args:
        predict: A tensor of logits, shape (batch_size, n_classes, height, width) output by the model
        target: A tensor of shape (batch_size, height, width) containing class indices
    """
    def __init__(self, n_classes=3):
        super().__init__()
        self.smooth = 1e-6  # parameter to avoid division by 0
        self.n_classes = n_classes

    def forward(self, predict, target):
        # predict shape: (batch_size, n_classes, height, width)
        # target shape: (batch_size, height, width)

        batch_size, n_classes, height, width = predict.shape

        assert self.n_classes == n_classes
        
        # get probabilities from logits
        predict = torch.softmax(predict, dim=1)
        
        # convert target to one-hot encoding
        target_one_hot = F.one_hot(target, num_classes=n_classes).permute(0, 3, 1, 2).float()   # permute to obtain (batch_size, n_classes, height, width)
        
        # flatten tensors for easier computation
        predict = predict.view(batch_size, n_classes, -1)  # (batch_size, n_classes, H*W)
        target_one_hot = target_one_hot.view(batch_size, n_classes, -1)  # (batch_size, n_classes, H*W)
        
        # compute dice coefficient for each class
        intersection = (predict * target_one_hot).sum(dim=2)  # (batch_size, n_classes)
        union = predict.sum(dim=2) + target_one_hot.sum(dim=2)  # (batch_size, n_classes)
        
        # dice coefficient: 2 * intersection / (union + smooth)
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        
        # average across classes and batch
        dice_loss = 1.0 - dice.mean()
        
        return dice_loss


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
        # True positives, false positives, false negatives
        tp = np.sum((y_true == class_id) & (y_pred == class_id))
        fp = np.sum((y_true != class_id) & (y_pred == class_id))
        fn = np.sum((y_true == class_id) & (y_pred != class_id))
        
        # IoU = TP / (TP + FP + FN)
        if tp + fp + fn > 0:
            iou = tp / (tp + fp + fn)
        else:
            iou = 1.0  # Perfect score for classes not present in either true or pred
            
        ious.append(iou)
    
    return np.mean(ious)


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




# main train loop
def train_model(model, train_loader, val_loader, criterion, optimizer, device, epochs=10):
    model.to(device)
    best_val_loss = float('inf')

    not_improved_epochs = 0

    for epoch in range(epochs):

        # put model in training mode (needed for batch normalization)
        model.train()
        train_loss = 0.0
        all_preds = []
        all_labels = []

        for x, y, coords in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} (Train)"):
            x, y = x.to(device), y.to(device).long().squeeze(1)  # Remove channel dimension

            optimizer.zero_grad()
            # output logits
            outputs = model(x)

            # apply argmax along channel dimension (dim=1) to get predicted classes
            preds = torch.argmax(outputs, dim=1)
            
            # loss calculation
            loss = criterion(outputs, y)
            
            # backpropagation
            loss.backward()
            # gradient descent
            optimizer.step()

            train_loss += loss.item()

            preds = preds.cpu().detach()
            all_preds.extend(preds.numpy())
            all_labels.extend(y.cpu().numpy())

        # epoch training metrics calculation
        train_loss /= len(train_loader)
        
        # Flatten arrays for metrics calculation
        all_preds_flat = np.array(all_preds).flatten()
        all_labels_flat = np.array(all_labels).flatten()
        
        train_acc = accuracy_score(all_labels_flat, all_preds_flat)
        train_miou = compute_mean_iou(all_labels_flat, all_preds_flat, n_classes=3)
        train_sens_12 = compute_sensitivity_classes_1_2(all_labels_flat, all_preds_flat)

        # validation phase
        model.eval()
        val_loss = 0.0
        all_val_preds = []
        all_val_labels = []

        # no update of parameters
        with torch.no_grad():
            for x, y, coords in tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} (Val)"):
                x, y = x.to(device), y.to(device).long().squeeze(1)  # Remove channel dimension

                outputs = model(x)
                preds = torch.argmax(outputs, dim=1)
                loss = criterion(outputs, y)

                val_loss += loss.item()

                preds = preds.cpu().detach()
                all_val_preds.extend(preds.numpy())
                all_val_labels.extend(y.cpu().numpy())

        # validation metrics calculation
        val_loss /= len(val_loader)
        
        # Flatten arrays for metrics calculation
        all_val_preds_flat = np.array(all_val_preds).flatten()
        all_val_labels_flat = np.array(all_val_labels).flatten()
        
        val_acc = accuracy_score(all_val_labels_flat, all_val_preds_flat)
        val_miou = compute_mean_iou(all_val_labels_flat, all_val_preds_flat, n_classes=3)
        val_sens_12 = compute_sensitivity_classes_1_2(all_val_labels_flat, all_val_preds_flat)

        # print results update
        print(f"Epoch {epoch + 1}:")
        print(f"  Train Loss: {train_loss:.4f} | Acc: {train_acc:.4f} | mIoU: {train_miou:.4f} | Sens(1,2): {train_sens_12:.4f}")
        print(f"  Val Loss: {val_loss:.4f}   | Acc: {val_acc:.4f} | mIoU: {val_miou:.4f} | Sens(1,2): {val_sens_12:.4f}\n")

        # saving best model at each epoch, if val loss improves
        if val_loss < best_val_loss:
            not_improved_epochs = 0
            best_val_loss = val_loss
            # Save the entire model (architecture + weights)
            torch.save(model.state_dict(), f"/home/dario/Desktop/FirePrediction/best_prediction_model.pth")
            print(f"  --> Saved best model (val_loss improved)\n")
        else: 
            not_improved_epochs += 1
            if not_improved_epochs == 5:
                print(f"Model training stopped early due to no improvement for f{not_improved_epochs} epochs.\n")
                return model
                
        


    print("\nTraining completed!")
    return model

#----------------------------------------------------------

def evaluate_model(model, test_loader, device, filenames, output_dir='/home/dario/Desktop/FirePrediction/TEST_PREDS'):
    model.eval()
    all_preds = []
    all_labels = []
    
    sample_idx = 0

    with torch.no_grad():
        for batch_idx, (x, y, coords) in enumerate(tqdm(test_loader, desc="Testing")):
            
            x, y = x.to(device), y.to(device).long().squeeze(1)  # Remove channel dimension

            outputs = model(x)
            
            preds = torch.argmax(outputs, dim=1)
            
            preds = preds.cpu().detach()

            all_preds.extend(preds.numpy())
            all_labels.extend(y.cpu().numpy())

            os.makedirs(output_dir, exist_ok=True)

            # Save each prediction in the batch
            for j, pred in enumerate(preds):
                # Get raw prediction array
                pred_array = pred.numpy().squeeze()
                
                # Use batch_idx * batch_size + j to get correct filename index
                filename_idx = batch_idx * len(preds) + j
                original_filename = os.path.basename(filenames[filename_idx]).replace('.npy', '')
                
                # Save as .npy file
                np.save(os.path.join(output_dir, f'{original_filename}_prediction.npy'), pred_array)

        all_preds_flat = np.array(all_preds).flatten()
        all_labels_flat = np.array(all_labels).flatten()

    # metrics
    acc = accuracy_score(all_labels_flat, all_preds_flat)
    miou = compute_mean_iou(all_labels_flat, all_preds_flat, n_classes=3)
    sens_12 = compute_sensitivity_classes_1_2(all_labels_flat, all_preds_flat)

    print(f"\nTest Accuracy: {acc:.4f} | Test mIoU: {miou:.4f} | Test Sens(1,2): {sens_12:.4f}")

train_dataset = Sent2Dataset("/home/dario/Desktop/FirePrediction/TILES_BALANCED/inputs", 
                             "/home/dario/Desktop/FirePrediction/TILES_BALANCED/labels")
test_dataset = Sent2Dataset("/home/dario/Desktop/FirePrediction/TEST_INPUT_DATA",
                            "/home/dario/Desktop/FirePrediction/TEST_LABELS")

# --- train-val split ---
val_ratio = 0.2  # 80/20 split
train_len = int((1 - val_ratio) * len(train_dataset))
val_len = len(train_dataset) - train_len
train_subset, val_subset = random_split(train_dataset, [train_len, val_len], generator=torch.Generator().manual_seed(33))

train_loader = DataLoader(train_subset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_subset, batch_size=16, shuffle=False)
# -----------------------

n_channels = len(test_dataset.band_indices)
test_filenames = test_dataset.input_list
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)



# architecture definition (setting n channels as following input data channels)
unet_model = UNet(n_channels, out_channels=3).to('cuda')

"""
# Calculate proper class weights based on inverse frequency
# Class 0: 90.1%, Class 1: 6.5%, Class 2: 3.4%
total_samples = 9443300 + 684978 + 357482
class_0_weight = total_samples / (3 * 9443300)  # ≈ 0.37
class_1_weight = total_samples / (3 * 684978)   # ≈ 5.11
class_2_weight = total_samples / (3 * 357482)   # ≈ 9.78
class_weights = torch.tensor([class_0_weight, class_1_weight, class_2_weight]).to('cuda')
print(f"Calculated class weights: {class_weights}")
"""

# optimizer & loss definition 
optimizer = torch.optim.Adam(unet_model.parameters(), lr=0.001, weight_decay=1e-4)  
criterion = MulticlassDiceLoss(n_classes=3)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# For debugging, let's check the class distribution in the data
print("Checking class distribution in training data...")
class_counts = {0: 0, 1: 0, 2: 0}
for i, (x, y, coords) in enumerate(train_loader):
    if i >= 10:  # Check first 10 batches
        break
    y_flat = y.flatten()
    unique, counts = torch.unique(y_flat, return_counts=True)
    for cls, count in zip(unique, counts):
        class_counts[cls.item()] += count.item()

total_pixels = sum(class_counts.values())
print(f"Class distribution: {[(cls, count, f'{count/total_pixels*100:.1f}%') for cls, count in class_counts.items()]}")


trained_model = train_model(unet_model, train_loader, val_loader, criterion, optimizer, device, 25)
#---------------------------------------------------------------

# to reload previous model
unet_model.load_state_dict(torch.load('/home/dario/Desktop/FirePrediction/best_prediction_model.pth', weights_only=True))

evaluate_model(unet_model, test_loader, 'cuda', test_filenames)