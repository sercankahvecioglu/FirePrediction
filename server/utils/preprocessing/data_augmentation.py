import numpy as np
import scipy as sp
import torch
import os
import glob
from pathlib import Path
from typing import Tuple, Dict, List, Optional
import random

# NOTE: input data should be of type (h, w, c)

def rotate_90(data):
    """Rotate array 90 degrees counterclockwise"""
    return np.rot90(data, k=1, axes=(-3, -2))  # Rotate spatial dimensions (h, w)

def rotate_180(data):
    """Rotate array 180 degrees"""
    return np.rot90(data, k=2, axes=(-3, -2))  # Rotate spatial dimensions (h, w)

def flip_horizontal(data):
    """Flip array horizontally (left-right mirror)"""
    return np.flip(data, axis=-2)  # Flip along width (columns)

def flip_vertical(data):
    """Flip array vertically (up-down mirror)"""
    return np.flip(data, axis=-3)  # Flip along height (rows)

def add_gaussian_noise(data, noise_factor=0.1, preserve_range=True):
    """
    Add Gaussian noise to input data
    
    Args:
        data (np.array): Input array of shape (h, w, c)
        noise_factor (float): Standard deviation of noise relative to data std
        preserve_range (bool): Whether to clip values to original data range
    
    Returns:
        np.array: Data with added Gaussian noise
    """
    noise = np.random.normal(0, noise_factor * np.std(data), data.shape)
    noisy_data = data + noise
    
    if preserve_range:
        # Clip to preserve original data range
        noisy_data = np.clip(noisy_data, np.min(data), np.max(data))
    
    return noisy_data.astype(data.dtype)


class DataAugmentation:
    """
    Data augmentation class for satellite imagery and corresponding labels.
    Applies spatial transformations to both data and labels, and optionally adds
    Gaussian noise to data only.
    """
    
    def __init__(self, 
                 apply_rotations: bool = True,
                 apply_flips: bool = True, 
                 apply_noise: bool = True,
                 noise_factor: float = 0.05,
                 augmentation_probability: float = 0.5,
                 seed: Optional[int] = None):
        """
        Initialize data augmentation parameters
        
        Args:
            apply_rotations (bool): Whether to apply rotation augmentations
            apply_flips (bool): Whether to apply flip augmentations  
            apply_noise (bool): Whether to apply noise augmentation to data
            noise_factor (float): Standard deviation factor for Gaussian noise
            augmentation_probability (float): Probability of applying each augmentation
            seed (int, optional): Random seed for reproducibility
        """
        self.apply_rotations = apply_rotations
        self.apply_flips = apply_flips
        self.apply_noise = apply_noise
        self.noise_factor = noise_factor
        self.augmentation_probability = augmentation_probability
        
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
    
    def _get_spatial_transformations(self) -> Dict:
        """Get available spatial transformation functions"""
        transformations = {}
        
        if self.apply_rotations:
            transformations.update({
                "rot90": rotate_90,
                "rot180": rotate_180,
            })
        
        if self.apply_flips:
            transformations.update({
                "flip_h": flip_horizontal,
                "flip_v": flip_vertical,
            })
            
        return transformations
    
    def augment_pair_all(self, data: np.ndarray, label: np.ndarray) -> List[Tuple[np.ndarray, np.ndarray, str]]:
        """
        Apply ALL augmentations to a data-label pair (for high-risk tiles)
        
        Args:
            data (np.ndarray): Input data array of shape (h, w, c)
            label (np.ndarray): Corresponding label array of shape (h, w, c) or (h, w, 1)
            
        Returns:
            List[Tuple[np.ndarray, np.ndarray, str]]: List of (augmented_data, augmented_label, transform_name) tuples
        """
        augmented_pairs = []
        spatial_transforms = self._get_spatial_transformations()
        
        # Apply ALL spatial transformations (both data and labels)
        for transform_name, transform_func in spatial_transforms.items():
            aug_data = transform_func(data)
            aug_label = transform_func(label)
            augmented_pairs.append((aug_data, aug_label, transform_name))
        
        # Apply noise augmentation (data only)
        if self.apply_noise:
            aug_data = add_gaussian_noise(data, self.noise_factor)
            augmented_pairs.append((aug_data, label.copy(), "noise"))
        
        return augmented_pairs

    def augment_pair(self, data: np.ndarray, label: np.ndarray) -> List[Tuple[np.ndarray, np.ndarray, str]]:
        """
        Apply augmentations to a data-label pair
        
        Args:
            data (np.ndarray): Input data array of shape (h, w, c)
            label (np.ndarray): Corresponding label array of shape (h, w, c) or (h, w, 1)
            
        Returns:
            List[Tuple[np.ndarray, np.ndarray, str]]: List of (augmented_data, augmented_label, transform_name) tuples
        """
        augmented_pairs = []
        spatial_transforms = self._get_spatial_transformations()
        
        # Apply spatial transformations (both data and labels)
        for transform_name, transform_func in spatial_transforms.items():
            if random.random() < self.augmentation_probability:
                aug_data = transform_func(data)
                aug_label = transform_func(label)
                augmented_pairs.append((aug_data, aug_label, transform_name))
        
        # Apply noise augmentation (data only)
        if self.apply_noise and random.random() < self.augmentation_probability:
            aug_data = add_gaussian_noise(data, self.noise_factor)
            augmented_pairs.append((aug_data, label.copy(), "noise"))
        
        return augmented_pairs
    
    def augment_data_only(self, data: np.ndarray) -> List[Tuple[np.ndarray, str]]:
        """
        Apply augmentations to data only (for inference/satellite processing)
        
        Args:
            data (np.ndarray): Input data array of shape (h, w, c)
            
        Returns:
            List[Tuple[np.ndarray, str]]: List of (augmented_data, transform_name) tuples
        """
        augmented_data = []
        all_transforms = self._get_spatial_transformations()
        
        # Apply spatial transformations
        for transform_name, transform_func in all_transforms.items():
            if random.random() < self.augmentation_probability:
                aug_data = transform_func(data)
                augmented_data.append((aug_data, transform_name))
        
        # Apply noise augmentation
        if self.apply_noise and random.random() < self.augmentation_probability:
            aug_data = add_gaussian_noise(data, self.noise_factor)
            augmented_data.append((aug_data, "noise"))
        
        return augmented_data

    def augment_data_only_all(self, data: np.ndarray) -> List[Tuple[np.ndarray, str]]:
        """
        Apply ALL augmentations to data only (for inference/satellite processing)
        
        Args:
            data (np.ndarray): Input data array of shape (h, w, c)
            
        Returns:
            List[Tuple[np.ndarray, str]]: List of (augmented_data, transform_name) tuples
        """
        augmented_data = []
        all_transforms = self._get_spatial_transformations()
        
        # Apply ALL spatial transformations
        for transform_name, transform_func in all_transforms.items():
            aug_data = transform_func(data)
            augmented_data.append((aug_data, transform_name))
        
        # Apply noise augmentation
        if self.apply_noise:
            aug_data = add_gaussian_noise(data, self.noise_factor)
            augmented_data.append((aug_data, "noise"))
        
        return augmented_data


def should_augment_tile(label: np.ndarray, 
                       high_risk_threshold: float = 0.6, 
                       min_percentage: float = 0.2) -> bool:
    """
    Determine if a tile should be augmented based on label values
    
    Args:
        label (np.ndarray): Label array of shape (h, w, c) or (h, w, 1)
        high_risk_threshold (float): Threshold value for high-risk pixels
        min_percentage (float): Minimum percentage of high-risk pixels required
        
    Returns:
        bool: True if tile should be augmented, False otherwise
    """
    # Count pixels above threshold
    high_risk_pixels = np.sum(label > high_risk_threshold)
    total_pixels = label.size
    
    # Calculate percentage
    percentage = high_risk_pixels / total_pixels
    
    return percentage > min_percentage


def should_keep_low_risk_tile(removal_probability: float = 0.5) -> bool:
    """
    Determine if a low-risk tile should be kept (with random removal for dataset balancing)
    
    Args:
        removal_probability (float): Probability of removing a low-risk tile (0.0-1.0)
        
    Returns:
        bool: True if tile should be kept, False if it should be removed
    """
    return random.random() > removal_probability


def data_augmentation(data_folder: str, 
                     labels_folder: Optional[str] = None,
                     output_data_folder: Optional[str] = None,
                     output_labels_folder: Optional[str] = None,
                     augmentation_config: Optional[Dict] = None) -> Dict:
    """
    Apply data augmentation to tiles in folders before NDVI/NDMI extraction.
    Only augments tiles where more than specified percentage of label pixels have values above threshold.
    Randomly removes low-risk tiles with specified probability for dataset balancing.
    
    Args:
        data_folder (str): Path to folder containing data .npy files
        labels_folder (str, optional): Path to folder containing label .npy files
        output_data_folder (str, optional): Output folder for augmented data. If None, saves to data_folder
        output_labels_folder (str, optional): Output folder for augmented labels. If None, saves to labels_folder
        augmentation_config (dict, optional): Configuration for augmentation parameters
        
    Returns:
        Dict: Summary of augmentation results including removal statistics
    """
    # Default augmentation configuration
    default_config = {
        'apply_rotations': True,
        'apply_flips': True,
        'apply_noise': True,
        'noise_factor': 0.05,
        'augmentation_probability': 0.7,
        'seed': 42,
        'high_risk_threshold': 0.6,
        'min_percentage': 0.15,
        'apply_all_to_high_risk': True,  # Apply all augmentations to high-risk tiles
        'remove_low_risk': True,  # Enable random removal of low-risk tiles
        'low_risk_removal_probability': 0.5  # Probability of removing low-risk tiles
    }
    
    if augmentation_config:
        default_config.update(augmentation_config)
    
    # Extract parameters for DataAugmentation class (filter out selective augmentation params)
    augmenter_params = {
        'apply_rotations': default_config['apply_rotations'],
        'apply_flips': default_config['apply_flips'],
        'apply_noise': default_config['apply_noise'],
        'noise_factor': default_config['noise_factor'],
        'augmentation_probability': default_config['augmentation_probability'],
        'seed': default_config['seed']
    }
    
    # Initialize augmentation
    augmenter = DataAugmentation(**augmenter_params)
    
    # Setup paths
    data_path = Path(data_folder)
    output_data_path = Path(output_data_folder) if output_data_folder else data_path
    
    if labels_folder:
        labels_path = Path(labels_folder)
        output_labels_path = Path(output_labels_folder) if output_labels_folder else labels_path
        process_labels = True
    else:
        process_labels = False
    
    # Create output directories
    output_data_path.mkdir(parents=True, exist_ok=True)
    if process_labels:
        output_labels_path.mkdir(parents=True, exist_ok=True)
    
    # Find data files
    data_files = list(data_path.glob("*.npy"))
    if not data_files:
        print(f"No .npy files found in {data_path}")
        return {'status': 'no_files', 'augmented_count': 0}
    
    print(f"Starting data augmentation on {len(data_files)} files...")
    augmented_count = 0
    skipped_count = 0
    removed_count = 0
    
    for data_file in data_files:
        try:
            # Load data
            data = np.load(data_file)
            
            if process_labels:
                # Find corresponding label file
                label_file = labels_path / data_file.name
                if not label_file.exists():
                    print(f"Warning: No corresponding label file for {data_file.name}")
                    continue
                
                label = np.load(label_file)
                
                # Check if tile should be augmented based on label values
                is_high_risk = should_augment_tile(label, 
                                                 default_config['high_risk_threshold'], 
                                                 default_config['min_percentage'])
                
                if is_high_risk:
                    # Apply ALL augmentations to high-risk tiles
                    if default_config.get('apply_all_to_high_risk', True):
                        augmented_pairs = augmenter.augment_pair_all(data, label)
                    else:
                        augmented_pairs = augmenter.augment_pair(data, label)
                    
                    for aug_data, aug_label, transform_name in augmented_pairs:
                        # Save augmented data
                        aug_data_filename = f"{transform_name}_{data_file.name}"
                        aug_label_filename = f"{transform_name}_{label_file.name}"
                        
                        np.save(output_data_path / aug_data_filename, aug_data)
                        np.save(output_labels_path / aug_label_filename, aug_label)
                        
                        augmented_count += 1
                else:
                    # Low-risk tile: decide whether to keep or remove for balancing
                    if default_config.get('remove_low_risk', True):
                        if not should_keep_low_risk_tile(default_config.get('low_risk_removal_probability', 0.5)):
                            # Remove this low-risk tile for dataset balancing
                            removed_count += 1
                            continue
                    
                    # Keep the original low-risk tile (no augmentation)
                    skipped_count += 1
                    # Optionally, copy the original tile to output (uncomment if needed)
                    # np.save(output_data_path / data_file.name, data)
                    # np.save(output_labels_path / label_file.name, label)
                    
            else:
                # Apply augmentations to data only (no selective filtering without labels)
                augmented_data = augmenter.augment_data_only(data)
                
                for aug_data, transform_name in augmented_data:
                    # Save augmented data
                    aug_data_filename = f"{transform_name}_{data_file.name}"
                    np.save(output_data_path / aug_data_filename, aug_data)
                    
                    augmented_count += 1
                    
        except Exception as e:
            print(f"Error processing {data_file.name}: {str(e)}")
            continue
    
    print(f"✓ Data augmentation completed! Generated {augmented_count} augmented samples")
    if process_labels:
        print(f"  Tiles processed: {len(data_files)}")
        high_risk_tiles = len(data_files) - skipped_count - removed_count
        print(f"  High-risk tiles augmented: {high_risk_tiles}")
        print(f"  Low-risk tiles kept: {skipped_count}")
        print(f"  Low-risk tiles removed for balancing: {removed_count}")
        print(f"  Dataset balance ratio (high-risk augmented : low-risk kept): {augmented_count}:{skipped_count}")
    
    return {
        'status': 'success',
        'original_files': len(data_files),
        'augmented_count': augmented_count,
        'skipped_count': skipped_count if process_labels else 0,
        'removed_count': removed_count if process_labels else 0,
        'high_risk_tiles': len(data_files) - skipped_count - removed_count if process_labels else 0,
        'output_data_folder': str(output_data_path),
        'output_labels_folder': str(output_labels_path) if process_labels else None,
        'config_used': default_config
    }


def augment_npy_files(folder_path):
    """
    Legacy function for backward compatibility.
    Apply data augmentation to all .npy files in the given folder.
    
    Args:
        folder_path (str): Path to folder containing .npy files
    """
    print("Using legacy augment_npy_files function. Consider using data_augmentation() for more features.")
    
    folder_path = Path(folder_path)
    
    # Find all .npy files in the folder
    npy_files = list(folder_path.glob("*.npy"))
    
    if not npy_files:
        print(f"No .npy files found in {folder_path}")
        return
    
    print(f"Found {len(npy_files)} .npy files to augment")
    
    for npy_file in npy_files:
        print(f"Processing: {npy_file.name}")
        
        try:
            # Load the numpy array
            data = np.load(npy_file)
            
            # Apply augmentations and save with appropriate prefixes
            augmentations = {
                "rot90": rotate_90(data),
                "rot180": rotate_180(data),
                "flipped_h": flip_horizontal(data),
                "flipped_v": flip_vertical(data),
                "noise": add_gaussian_noise(data)
            }
            
            for prefix, augmented_data in augmentations.items():
                # Create new filename with prefix
                new_filename = f"{prefix}_{npy_file.name}"
                new_filepath = folder_path / new_filename
                
                # Save augmented data
                np.save(new_filepath, augmented_data)
                print(f"  Saved: {new_filename}")
                
        except Exception as e:
            print(f"Error processing {npy_file.name}: {str(e)}")
    
    print("Data augmentation completed!")


def main():
    """Main function to run data augmentation"""
    print("=== Data Augmentation Tool ===")
    print("1. Legacy mode (simple file augmentation)")
    print("2. Advanced mode (data + labels with configuration)")
    
    choice = input("Select mode (1 or 2): ").strip()
    
    if choice == "1":
        # Legacy mode
        current_folder = "/home/dario/Desktop/sample_veg_detection_data/VEGETATION"
        folder_to_process = input(f"Enter folder path (or press Enter for {current_folder}): ").strip()
        if folder_to_process:
            current_folder = folder_to_process
        
        print(f"Applying data augmentation to .npy files in: {current_folder}")
        augment_npy_files(current_folder)
        
    elif choice == "2":
        # Advanced mode
        data_folder = input("Enter data folder path: ").strip()
        labels_folder = input("Enter labels folder path (or press Enter to skip): ").strip()
        
        if not labels_folder:
            labels_folder = None
            
        config = {
            'augmentation_probability': 0.7,
            'noise_factor': 0.05,
            'apply_rotations': True,
            'apply_flips': True,
            'apply_noise': True,
            'high_risk_threshold': 0.6,
            'min_percentage': 0.2,
            'apply_all_to_high_risk': True,
            'remove_low_risk': True,
            'low_risk_removal_probability': 0.5
        }
        
        result = data_augmentation(
            data_folder=data_folder,
            labels_folder=labels_folder,
            augmentation_config=config
        )
        
        print(f"Augmentation result: {result}")
        
    else:
        print("Invalid choice. Please run again and select 1 or 2.")


if __name__ == "__main__":
    main()