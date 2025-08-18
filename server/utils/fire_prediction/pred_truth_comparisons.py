import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path
import glob

def create_prediction_comparison_plots(labels_dir="/home/dario/Desktop/FirePrediction/TEST_LABELS", 
                                     preds_dir="/home/dario/Desktop/FirePrediction/TEST_PREDS",
                                     output_dir="/home/dario/Desktop/FirePrediction/TEST_PREDS_VS_LABELS"):
    """
    Creates side-by-side comparison plots of predictions vs ground truth labels.
    
    Args:
        labels_dir: Directory containing ground truth .npy files
        preds_dir: Directory containing prediction .npy files  
        output_dir: Directory to save comparison plots
    """
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all label files
    label_files = glob.glob(os.path.join(labels_dir, "*.npy"))
    
    for label_file in label_files:
        # Get base filename without extension
        base_name = os.path.splitext(os.path.basename(label_file))[0]
        
        # Find corresponding prediction file
        pred_file = os.path.join(preds_dir, f"{base_name}_prediction.npy")
        
        if not os.path.exists(pred_file):
            print(f"Warning: Prediction file not found for {base_name}")
            continue
            
        try:
            # Load arrays
            label_array = np.load(label_file).squeeze()  # Remove extra dimension
            pred_array = np.load(pred_file).squeeze()  # Remove extra dimension
            
            # Calculate error - show absolute difference between prediction and ground truth
            error_array = np.abs(pred_array - label_array)
            accuracy = np.mean(label_array == pred_array)
            
            # Create three-subplot plot
            fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))
            
            # Plot ground truth with grayscale for continuous values
            im1 = ax1.imshow(label_array, cmap='coolwarm', vmin=0, vmax=1)
            ax1.set_title('Ground Truth')
            ax1.axis('off')
            plt.colorbar(im1, ax=ax1, shrink=0.8)
            
            # Plot prediction with grayscale for continuous values
            im2 = ax2.imshow(pred_array, cmap='coolwarm', vmin=0, vmax=1)
            ax2.set_title('Prediction')
            ax2.axis('off')
            plt.colorbar(im2, ax=ax2, shrink=0.8)
            
            # Plot error with sequential colormap
            im3 = ax3.imshow(error_array, cmap='Reds', vmin=0, vmax=1)
            ax3.set_title(f'Absolute Error |Pred - Truth| (MAE: {np.mean(error_array):.4f})')
            ax3.axis('off')
            plt.colorbar(im3, ax=ax3, shrink=0.8)
            
            # Add main title
            fig.suptitle(f'Comparison: {base_name}', fontsize=14)
            
            # Adjust layout and save
            plt.tight_layout()
            output_path = os.path.join(output_dir, f"{base_name}_comparison.png")
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            print(f"Saved comparison plot: {output_path}")
            
        except Exception as e:
            print(f"Error processing {base_name}: {str(e)}")
            continue
    
    print(f"\nComparison plots saved in: {output_dir}")

if __name__ == "__main__":
    create_prediction_comparison_plots()
