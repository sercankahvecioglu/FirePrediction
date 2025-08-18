#!/usr/bin/env python3
"""
Utility script to calculate and manage dataset statistics for global normalization.
This script should be run only, in the beginning, on your training data, to generate the statistics file.
"""

import argparse
import os
import sys
from datasets import calculate_dataset_statistics, load_dataset_statistics


def main():
    parser = argparse.ArgumentParser(description='Calculate dataset statistics for global normalization')
    parser.add_argument('--input_dir', type=str, required=True,
                        help='Directory containing .npy input files')
    parser.add_argument('--output_path', type=str, default=None,
                        help='Output path for statistics file (default: input_dir/dataset_statistics.json)')
    parser.add_argument('--force', action='store_true',
                        help='Force recalculation even if statistics file exists')
    parser.add_argument('--band_indices', type=int, nargs='+', default=[7, 11, 12, 13, 14],
                        help='Band indices to use (default: [7, 11, 12, 13, 14])')
    parser.add_argument('--verify', action='store_true',
                        help='Verify existing statistics file')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input_dir):
        print(f"Error: Input directory {args.input_dir} does not exist")
        sys.exit(1)
    
    if args.output_path is None:
        args.output_path = os.path.join(args.input_dir, 'dataset_statistics.json')
    
    if args.verify:
        if os.path.exists(args.output_path):
            try:
                means, stds = load_dataset_statistics(args.output_path)
                print(f"Statistics file verified: {args.output_path}")
                print(f"Means: {means}")
                print(f"Stds: {stds}")
            except Exception as e:
                print(f"Error loading statistics file: {e}")
                sys.exit(1)
        else:
            print(f"Statistics file does not exist: {args.output_path}")
            sys.exit(1)
        return
    
    try:
        means, stds = calculate_dataset_statistics(
            input_data_dir=args.input_dir,
            band_indices=args.band_indices,
            save_path=args.output_path,
            force_recalculate=args.force
        )
        
        print("\nCalculation completed successfully!")
        print(f"Statistics saved to: {args.output_path}")
        print(f"Means: {means}")
        print(f"Standard deviations: {stds}")
        print(f"Band indices: {args.band_indices}")
        
    except Exception as e:
        print(f"Error calculating statistics: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
