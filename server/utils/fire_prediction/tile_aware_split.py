"""
Tile-aware train-validation split for geospatial datasets.

Prevents data leakage by ensuring augmented versions of the same tile 
stay together in the same split (either all copies in train or all copies in val).
"""

import os
import re
from collections import defaultdict
from typing import List, Tuple, Dict
import torch
from torch.utils.data import Subset
import numpy as np


def extract_base_tile_identifier(filename: str) -> str:
    """Extract base tile ID from filename, removing augmentation prefixes."""
    basename = os.path.basename(filename).replace('.npy', '')
    pattern = r'^(?:noise_|rot180_|rot90_)?(.+_tile_\(.+\))$'
    match = re.match(pattern, basename)
    return match.group(1) if match else basename


def group_files_by_tile(file_paths: List[str]) -> Dict[str, List[str]]:
    """Group file paths by their base tile identifier."""
    tile_groups = defaultdict(list)
    for file_path in file_paths:
        base_id = extract_base_tile_identifier(file_path)
        tile_groups[base_id].append(file_path)
    return dict(tile_groups)


def create_tile_aware_split(dataset, val_ratio: float = 0.2, seed: int = 33) -> Tuple[Subset, Subset]:
    """Create train-validation split that keeps all versions of the same tile together."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # Get unique tiles and split them
    tile_ids = sorted(set(extract_base_tile_identifier(f) for f in dataset.input_list))
    np.random.shuffle(tile_ids)
    n_val_tiles = int(val_ratio * len(tile_ids))
    val_tiles = set(tile_ids[:n_val_tiles])
    
    # Assign indices based on tile membership
    train_indices, val_indices = [], []
    for idx, file_path in enumerate(dataset.input_list):
        base_id = extract_base_tile_identifier(file_path)
        (val_indices if base_id in val_tiles else train_indices).append(idx)
    
    # Print summary and verify no leakage
    print(f"\nTile-aware split: {len(tile_ids)} tiles → {len(train_indices)} train, {len(val_indices)} val samples")
    print(f"Validation ratio: {len(val_indices) / len(dataset.input_list):.3f}")
    
    train_tiles = {extract_base_tile_identifier(dataset.input_list[i]) for i in train_indices}
    val_tiles_check = {extract_base_tile_identifier(dataset.input_list[i]) for i in val_indices}
    overlap = train_tiles & val_tiles_check
    print("✓ No data leakage detected" if not overlap else f"ERROR: {len(overlap)} tiles in both splits!")
    
    return Subset(dataset, train_indices), Subset(dataset, val_indices)


def analyze_augmentation_distribution(dataset) -> None:
    """Analyze the distribution of original vs augmented files in the dataset."""
    prefixes = ['noise_', 'rot180_', 'rot90_']
    counts = {'original': 0, 'noise': 0, 'rot180': 0, 'rot90': 0}
    
    for file_path in dataset.input_list:
        basename = os.path.basename(file_path)
        key = next((prefix[:-1] for prefix in prefixes if basename.startswith(prefix)), 'original')
        counts[key] += 1
    
    total = len(dataset.input_list)
    print(f"\nAugmentation distribution:")
    for key, count in counts.items():
        print(f"{key.capitalize()} files: {count} ({count/total:.1%})")
    print(f"Total files: {total}")
    
    # Check completeness
    tile_groups = group_files_by_tile(dataset.input_list)
    complete = sum(1 for files in tile_groups.values() 
                  if len(files) == 4 and all(any(os.path.basename(f).startswith(p) or not p 
                  for f in files) for p in ['', 'noise_', 'rot180_', 'rot90_']))
    
    print(f"\nTile completeness: {complete} complete groups, {len(tile_groups) - complete} incomplete")


if __name__ == "__main__":
    import sys
    sys.path.append('/home/dario/Desktop/FirePrediction/server/utils/fire_prediction')
    from datasets import TrainingDataset
    
    dataset = TrainingDataset("/home/dario/Desktop/FirePrediction/inputs", 
                             "/home/dario/Desktop/FirePrediction/labels")
    train_subset, val_subset = create_tile_aware_split(dataset)
