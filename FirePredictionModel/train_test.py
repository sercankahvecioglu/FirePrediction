import torch
from torch.utils.data import DataLoader, random_split  
from FirePredictionModel.unet import UNet
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score
import os   
from FirePredictionModel.datasets import Sent2Dataset
from FirePredictionModel.loss_metrics import GeneralizedDiceLoss, calculate_class_weights, compute_mean_iou, compute_sensitivity_classes_1_2

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
            torch.save(model.state_dict(), f"/home/dario/Desktop/FirePrediction/trained_models/best_fireprediction_model.pth")
            print(f"  --> Saved best model (val_loss improved)\n")
        else: 
            not_improved_epochs += 1
            if not_improved_epochs == 5:
                print(f"Model training stopped early due to no improvement for f{not_improved_epochs} epochs.\n")
                return model
                
        


    print("\nTraining completed!")
    return model

#-----------------------------------------------------------------

# test evaluation loop
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


#-------------------------main.py section-------------------------


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

# calculate proper class weights (for weighted dice loss) based on inverse frequency from training data
fire_class_weights = calculate_class_weights(train_loader, n_classes=3).to('cuda')

n_channels = len(test_dataset.band_indices)
test_filenames = test_dataset.input_list
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

# architecture definition (setting n channels as following input data bands channels)
unet_model = UNet(n_channels, out_channels=3).to('cuda')

# optimizer & loss definition 
optimizer = torch.optim.Adam(unet_model.parameters(), lr=0.001, weight_decay=1e-4)  
criterion = GeneralizedDiceLoss(class_weights=fire_class_weights)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

"""
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
"""

trained_model = train_model(unet_model, train_loader, val_loader, criterion, optimizer, device, 25)

#-----------------------------------------------------------------

# to reload previous model
unet_model.load_state_dict(torch.load('/home/dario/Desktop/FirePrediction/best_fireprediction_model.pth', weights_only=True))

evaluate_model(unet_model, test_loader, 'cuda', test_filenames)