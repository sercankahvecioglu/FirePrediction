import torch
from torch.utils.data import DataLoader, random_split, WeightedRandomSampler
from unet import UNet
import numpy as np
from tqdm import tqdm
import os   
from datasets import *
from tile_aware_split import create_tile_aware_split

def init_final_bias(model, target_prob=0.7):
    """
    Initialize final layer bias to favor high-risk predictions
    target_prob: initial output probability (0.5-0.8 recommended for fire risk)
    """
    
    # Convert probability to logit for sigmoid activation
    bias_value = np.log(target_prob / (1 - target_prob))
    
    # Find and initialize the final convolutional layer
    final_layers = [m for m in model.modules() if isinstance(m, torch.nn.Conv2d)]
    if final_layers:
        final_layers[-1].bias.data.fill_(bias_value)
        print(f"Initialized final layer bias to {bias_value:.3f} (target prob: {target_prob:.1f})")


#------------------------------------------------------------

# R-squared calculation function
def calculate_r2(predictions, targets):
    """Calculate R-squared coefficient"""
    # Flatten tensors for calculation
    pred_flat = predictions.view(-1)
    target_flat = targets.view(-1)
    
    # Calculate R-squared
    ss_res = torch.sum((target_flat - pred_flat) ** 2)
    ss_tot = torch.sum((target_flat - torch.mean(target_flat)) ** 2)
    r2 = 1 - (ss_res / ss_tot)
    return r2.item()

# main train loop
def train_model(model, train_loader, val_loader, criterion, optimizer, device, start_epoch=0, epochs=10):
    model.to(device)
    best_val_loss = float('inf')

    not_improved_epochs = 0

    for epoch in range(start_epoch, start_epoch+epochs):

        # put model in training mode (needed for batch normalization)
        model.train()
        train_loss = 0.0
        all_preds = []
        all_labels = []

        for x, y in tqdm(train_loader, desc=f"Epoch {epoch+1}/{start_epoch+epochs} (Train)"):
            x, y = x.to(device), y.to(device).float()  # Keep as float for regression

            optimizer.zero_grad()
            # output probabilities
            outputs = model(x)
            
            # loss calculation
            loss = criterion(outputs, y)
            
            # backpropagation
            loss.backward()
            # gradient descent
            optimizer.step()

            train_loss += loss.item()

            outputs = outputs.cpu().detach()

            all_preds.append(outputs)
            all_labels.append(y.cpu())

        # epoch training metrics calculation
        train_loss /= len(train_loader)
        
        # Concatenate all predictions and labels
        all_preds = torch.cat(all_preds, dim=0)
        all_labels = torch.cat(all_labels, dim=0)
        
        train_mae = torch.nn.functional.l1_loss(all_preds, all_labels).item()
        train_rmse = torch.sqrt(torch.nn.functional.mse_loss(all_preds, all_labels)).item()
        train_r2 = calculate_r2(all_preds, all_labels)

        # validation phase
        model.eval()
        val_loss = 0.0
        all_val_preds = []
        all_val_labels = []

        # no update of parameters
        with torch.no_grad():
            for x, y in tqdm(val_loader, desc=f"Epoch {epoch+1}/{start_epoch+epochs} (Val)"):
                x, y = x.to(device), y.to(device).float()  # Keep as float for regression

                outputs = model(x)
                loss = criterion(outputs, y)

                val_loss += loss.item()

                outputs = outputs.cpu().detach()
                all_val_preds.append(outputs)
                all_val_labels.append(y.cpu())

        # validation metrics calculation
        val_loss /= len(val_loader)
        
        # Concatenate all predictions and labels
        all_val_preds = torch.cat(all_val_preds, dim=0)
        all_val_labels = torch.cat(all_val_labels, dim=0)
        
        val_mae = torch.nn.functional.l1_loss(all_val_preds, all_val_labels).item()
        val_rmse = torch.sqrt(torch.nn.functional.mse_loss(all_val_preds, all_val_labels)).item()
        val_r2 = calculate_r2(all_val_preds, all_val_labels)

        # print results update
        print(f"Epoch {epoch + 1}:")
        print(f"  Train Loss: {train_loss:.6f} | MAE: {train_mae:.6f} | RMSE: {train_rmse:.6f} | R²: {train_r2:.6f}")
        print(f"  Val Loss: {val_loss:.6f}   | MAE: {val_mae:.6f} | RMSE: {val_rmse:.6f}   | R²: {val_r2:.6f}\n")

        if (epoch+1) % 5 == 0 or epoch == epochs - 1:  # Every 5 epochs + final epoch save checkpoint
            torch.save({
            'epoch': epoch+1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_val_loss': best_val_loss,
            'train_loss': train_loss,
            'val_loss': val_loss
        }, f'../trained_models/checkpoint_epoch_{epoch+1}.pth')
            print(f"Checkpoint saved at epoch {epoch+1}\n")                                                 

        # saving best model at each epoch, if val loss improves
        if val_loss < best_val_loss:
            not_improved_epochs = 0
            best_val_loss = val_loss
            # save model weights
            torch.save(model.state_dict(), f"/home/dario/Desktop/FirePrediction/server/utils/trained_models/best_fireprediction_model.pth")
            #-------------------------------------------
            # TODO: PLACEHOLDER FOR SAVING TO ONNX FORMAT
            #-------------------------------------------
            print(f"  --> Saved best model (val_loss improved)\n")
        else: 
            not_improved_epochs += 1
            if not_improved_epochs == 10:
                print(f"Model training stopped early due to no improvement for {not_improved_epochs} epochs.\n")
                return model

    print("\nTraining completed!")
    return model

#-----------------------------------------------------------------

# test evaluation loop
def evaluate_model(model, test_loader, device, filenames, output_dir='/home/dario/Desktop/FirePrediction/TEST_PREDS'):
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch_idx, (x, y, coords) in enumerate(tqdm(test_loader, desc="Testing")):
            
            x, y = x.to(device), y.to(device).float()  # Keep as float for regression

            outputs = model(x)
            
            outputs = outputs.cpu().detach()

            all_preds.append(outputs)
            all_labels.append(y.cpu())

            os.makedirs(output_dir, exist_ok=True)

            # Save each prediction in the batch
            for j, pred in enumerate(outputs):
                # Get raw prediction array
                pred_array = pred.numpy()
                
                # Use batch_idx * batch_size + j to get correct filename index
                filename_idx = batch_idx * len(outputs) + j
                original_filename = os.path.basename(filenames[filename_idx]).replace('.npy', '')
                
                # Save as .npy file
                np.save(os.path.join(output_dir, f'{original_filename}_prediction.npy'), pred_array)

        # Concatenate all predictions and labels
        all_preds = torch.cat(all_preds, dim=0)
        all_labels = torch.cat(all_labels, dim=0)

    # metrics
    mae = torch.nn.functional.l1_loss(all_preds, all_labels).item()
    rmse = torch.sqrt(torch.nn.functional.mse_loss(all_preds, all_labels)).item()
    r2 = calculate_r2(all_preds, all_labels)

    print(f"\nTest MAE: {mae:.6f} | Test RMSE: {rmse:.6f} | Test R²: {r2:.6f}")


#-------------------------main.py section-------------------------

train_dataset = TrainingDataset("/home/dario/Desktop/FirePrediction/inputs", 
                             "/home/dario/Desktop/FirePrediction/labels")
test_dataset = TestingDataset("/home/dario/Desktop/FirePrediction/TEST_INPUT_DATA",
                            "/home/dario/Desktop/FirePrediction/TEST_LABELS")

print(f"Loaded training dataset with {len(train_dataset)} samples")

# --- train-val split ---
# Use tile-aware split to prevent data leakage from augmented tiles
# This ensures that augmented versions of the same tile don't appear in both train and validation sets
print("Creating tile-aware train-validation split...")
train_subset, val_subset = create_tile_aware_split(train_dataset, val_ratio=0.2, seed=33)

train_loader = DataLoader(train_subset, batch_size=8, shuffle=True)
val_loader = DataLoader(val_subset, batch_size=8, shuffle=False)

n_channels = len(test_dataset.band_indices)
test_filenames = test_dataset.input_list
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

# architecture definition (setting n channels as following input data bands channels)
unet_model = UNet(n_channels, out_channels=1).to('cuda')
#init_final_bias(unet_model, target_prob=0.7) 


optimizer = torch.optim.Adam(unet_model.parameters(), lr=0.001)  

start_epoch = 0



# load and resume training from checkpoints
checkpoint = torch.load('../trained_models/checkpoint_epoch_15.pth')
unet_model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
start_epoch = checkpoint['epoch'] 
loss = checkpoint['train_loss']


scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=5
)

# optimizer & loss definition
device = 'cuda' if torch.cuda.is_available() else 'cpu'

class WeightedMSELoss(torch.nn.Module):
    def __init__(self, high_risk_weight=3.0, threshold=0.5):
        super().__init__()
        self.high_risk_weight = high_risk_weight
        self.threshold = threshold
        
    def forward(self, pred, target):
        mse = torch.nn.functional.mse_loss(pred, target, reduction='none')
        weights = torch.where(target > self.threshold, 
                            self.high_risk_weight, 1.0)
        return (weights * mse).mean()

# Use MSE loss for regression
criterion = WeightedMSELoss()

trained_model = train_model(unet_model, train_loader, val_loader, criterion, optimizer, device, start_epoch, 25)

#-----------------------------------------------------------------

# to reload previous model
unet_model.load_state_dict(torch.load('/home/dario/Desktop/FirePrediction/server/utils/trained_models/best_fireprediction_model.pth', weights_only=True))

evaluate_model(unet_model, test_loader, 'cuda', test_dataset.input_list)