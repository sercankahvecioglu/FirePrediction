# Tile-Aware Train-Validation Split Implementation

## Problem Solved

The original training code used PyTorch's `random_split()` function, which could place augmented versions of the same geographical tile in both training and validation sets. This created **data leakage** - the model could see very similar images (just with noise or rotation) in both splits, leading to artificially inflated validation performance.

## Solution Implementation

### New Files Created

1. **`tile_aware_split.py`** - Core implementation
   - `extract_base_tile_identifier()` - Extracts base tile ID from augmented filenames
   - `group_files_by_tile()` - Groups all versions of the same tile together
   - `create_tile_aware_split()` - Creates split ensuring tile groups stay together
   - `analyze_augmentation_distribution()` - Analyzes dataset composition

2. **`analyze_dataset.py`** - Utility script for dataset analysis

### Modified Files

1. **`train_test.py`** - Updated main training script
   - Added import for `create_tile_aware_split`
   - Replaced `random_split()` with `create_tile_aware_split()`
   - Added informative print statements

## Dataset Analysis Results

From the analysis of your dataset:

- **Total samples**: 7,549
- **Original files**: 6,715 (89.0%)
- **Augmented files**: 834 (11.0%)
  - Noise augmented: 278 (3.7%)
  - 180° rotated: 278 (3.7%)  
  - 90° rotated: 278 (3.7%)

**Tile groups**:
- **Complete groups**: 278 tiles (have all 4 versions: original + 3 augmentations)
- **Incomplete groups**: 6,437 tiles (only have original version)
- **Total unique tiles**: 6,715

## How It Works

### File Naming Pattern Recognition
```python
# Example files for the same tile:
"usa2_tile_(512, 3328).npy"           # Original
"noise_usa2_tile_(512, 3328).npy"     # Noise augmented
"rot180_usa2_tile_(512, 3328).npy"    # 180° rotated  
"rot90_usa2_tile_(512, 3328).npy"     # 90° rotated

# All grouped under base ID: "usa2_tile_(512, 3328)"
```

### Split Process
1. **Group by base tile ID** - All versions of same tile grouped together
2. **Split at tile level** - Randomly assign tile groups to train/validation
3. **Maintain proportions** - Achieves ~20% validation as requested
4. **Verify no leakage** - Confirms no tile appears in both splits

## Benefits

### Data Integrity
- ✅ **No data leakage** - Same geographical location never appears in both splits
- ✅ **Realistic evaluation** - Validation truly tests generalization to unseen locations
- ✅ **Consistent with geospatial ML best practices**

### Performance Impact
- **Training samples**: 6,065 (was ~6,039 with random split)
- **Validation samples**: 1,484 (was ~1,510 with random split)  
- **Validation ratio**: 19.7% (target was 20%)

## Usage

The implementation is now integrated into your training pipeline. When you run `train_test.py`, it will automatically:

1. Load the dataset
2. Create tile-aware split with informative output
3. Proceed with training using the properly split data

To analyze your dataset independently:
```bash
python analyze_dataset.py
```

## Technical Details

### Key Functions

- **`extract_base_tile_identifier()`**: Uses regex to remove augmentation prefixes
- **`create_tile_aware_split()`**: Maintains same interface as `random_split()` for easy integration
- **Reproducibility**: Uses same seed (33) as original implementation

### Error Handling
- Validates split integrity (no overlap between train/val)
- Reports warnings for unexpected file patterns
- Provides detailed statistics about the split

This implementation ensures your fire prediction model training follows geospatial ML best practices and provides more reliable validation metrics.
