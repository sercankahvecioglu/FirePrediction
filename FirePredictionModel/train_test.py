import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch 
import torch.nn as nn
from torch.utils.data import DataLoader, random_split  # <-- add random_split
from models.unet import UNet
import numpy as np
from PIL import Image
from tqdm import tqdm
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import glob
import os
import pickle
import matplotlib
from models.datasets import Sent2Dataset



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
            x, y = x.to(device), y.to(device).float()

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
            all_preds.extend(outputs.numpy())
            all_labels.extend(y.cpu().numpy())

        # epoch training metrics calculation
        train_loss /= len(train_loader)
        
        # Flatten arrays for metrics calculation
        all_preds_flat = np.array(all_preds).flatten()
        all_labels_flat = np.array(all_labels).flatten()
        
        train_mse = mean_squared_error(all_labels_flat, all_preds_flat)
        train_mae = mean_absolute_error(all_labels_flat, all_preds_flat)

        # validation phase
        model.eval()
        val_loss = 0.0
        all_val_preds = []
        all_val_labels = []

        # no update of parameters
        with torch.no_grad():
            for x, y, coords in tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} (Val)"):
                x, y = x.to(device), y.to(device).float()

                outputs = model(x)
                loss = criterion(outputs, y)

                val_loss += loss.item()

                outputs = outputs.cpu().detach()
                all_val_preds.extend(outputs.numpy())
                all_val_labels.extend(y.cpu().numpy())

        # validation metrics calculation
        val_loss /= len(val_loader)
        
        # Flatten arrays for metrics calculation
        all_val_preds_flat = np.array(all_val_preds).flatten()
        all_val_labels_flat = np.array(all_val_labels).flatten()
        
        val_mse = mean_squared_error(all_val_labels_flat, all_val_preds_flat)
        val_mae = mean_absolute_error(all_val_labels_flat, all_val_preds_flat)

        # print results update
        print(f"Epoch {epoch + 1}:")
        print(f"  Train Loss: {train_loss:.4f} | MSE: {train_mse:.4f} | MAE: {train_mae:.4f}")
        print(f"  Val Loss: {val_loss:.4f}   | MSE: {val_mse:.4f} | MAE: {val_mae:.4f}\n")

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

def evaluate_model(model, test_loader, device, filenames, output_dir='/home/dario/Desktop/FirePrediction/TEST_PRED_IMGS'):
    model.eval()
    all_preds = []
    all_labels = []
    
    sample_idx = 0

    with torch.no_grad():
        for batch_idx, (x, y, coords) in enumerate(tqdm(test_loader, desc="Testing")):
            
            x, y = x.to(device), y.to(device).float()

            outputs = model(x).squeeze(1)
            preds = outputs.cpu().detach()

            all_preds.extend(preds.numpy())
            all_labels.extend(y.cpu().numpy())

            os.makedirs(output_dir, exist_ok=True)

            # Save each prediction in the batch
            for j, pred in enumerate(preds):
                # Normalize prediction to 0-1 range
                pred_array = pred.numpy().squeeze()
                
                # Apply colormap (blue to red: 'coolwarm', 'RdBu_r', 'bwr')
                # Other options: 'viridis', 'plasma', 'inferno', 'magma', 'jet'
                colormap = matplotlib.colormaps.get_cmap('coolwarm')  # Blue to red colormap
                colored_pred = colormap(pred_array)
                
                # Convert to uint8 (0-255) and remove alpha channel if present
                colored_pred_uint8 = (colored_pred[:, :, :3] * 255).astype(np.uint8)
                
                # Create PIL image and save with original filename
                img = Image.fromarray(colored_pred_uint8, mode='RGB')
                # Use batch_idx * batch_size + j to get correct filename index
                filename_idx = batch_idx * len(preds) + j
                original_filename = os.path.basename(filenames[filename_idx]).replace('.npy', '')
                img.save(os.path.join(output_dir, f'{original_filename}_prediction.png'))

        all_preds_flat = np.array(all_preds).flatten()
        all_labels_flat = np.array(all_labels).flatten()

    mse = mean_squared_error(all_labels_flat, all_preds_flat)
    mae = mean_absolute_error(all_labels_flat, all_preds_flat)

    print(f"\nTest MAE: {mae:.4f} | Test MSE: {mse:.4f}")

#-------------------------------------------------------------------


train_dataset = Sent2Dataset("/home/dario/Desktop/FirePrediction/TILES_INPUT_DATA", 
                             "/home/dario/Desktop/FirePrediction/TILES_LABELS")
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
unet_model = UNet(n_channels).to('cuda')

"""
# optimizer & loss definition
optimizer = torch.optim.Adam(unet_model.parameters(), lr=0.001)
criterion = nn.MSELoss()
device = 'cuda' if torch.cuda.is_available() else 'cpu'


trained_model = train_model(unet_model, train_loader, val_loader, criterion, optimizer, device, 25)

"""
#---------------------------------------------------------------

# to reload previous model
unet_model.load_state_dict(torch.load('/home/dario/Desktop/FirePrediction/best_prediction_model.pth', weights_only=True))

evaluate_model(unet_model, test_loader, 'cuda', test_filenames)