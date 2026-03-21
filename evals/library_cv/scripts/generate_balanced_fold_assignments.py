#!/usr/bin/env python3
"""
Generate Balanced Library Fold Assignments with Positive Rate Constraints.

This script creates fold assignments that ensure each test fold has a positive
rate within specified bounds, enabling meaningful statistical evaluation while
avoiding unrealistic positive rates.

Two modes of operation:
1. **Minimum rate only** (default): Ensures each fold has at least the specified
   minimum positive rate. Uses `create_positive_balanced_library_kfold_splits`.

2. **Bounded rate** (when --max_positive_rate is specified): Ensures each fold
   has a positive rate within [min, max] bounds. Uses the more sophisticated
   `create_bounded_positive_rate_library_kfold_splits` which intelligently
   pairs high-positive and low-positive libraries.

The bounded mode is particularly useful for drug discovery datasets where:
- Some chemical libraries have very high hit rates (>30%)
- Others have very low hit rates (<1%)
- You want each CV fold to represent realistic screening conditions (e.g., 3-20%)

Usage Examples:
    # Minimum rate only (at least 4% positives per fold)
    python evals/library_cv/scripts/generate_balanced_fold_assignments.py \
        --dataset wanglab/delbert-data \
        --dataset_name LRRK2 \
        --n_folds 20 \
        --min_positive_rate 0.04 \
        --seed 37 \
        --output evals/library_cv/fold_assignments/LRRK2_20fold_lib_balanced.yaml

    # Bounded rate (3-15% positives per fold, at least 1000 samples)
    python evals/library_cv/scripts/generate_balanced_fold_assignments.py \
        --dataset wanglab/delbert-data \
        --dataset_name DCAF7 \
        --n_folds 20 \
        --min_positive_rate 0.03 \
        --max_positive_rate 0.15 \
        --min_fold_size 1000 \
        --seed 37 \
        --output evals/library_cv/fold_assignments/DCAF7_18fold_bounded.yaml

Note: When constraints cannot be fully satisfied, the script may return fewer
folds than requested and will print detailed warnings about any violations.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
from datasets import load_dataset, load_from_disk

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent

from delbert.data.cv_utils import (
    create_positive_balanced_library_kfold_splits,
    create_bounded_positive_rate_library_kfold_splits,
    save_fold_assignments,
    print_fold_summary
)


def main():
    parser = argparse.ArgumentParser(
        description="Generate balanced library fold assignments with positive rate constraints",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Minimum rate only (at least 4%% positives per fold)
  %(prog)s --dataset wanglab/delbert-data --dataset_name LRRK2 --n_folds 20 \\
      --min_positive_rate 0.04 --output LRRK2_20fold_balanced.yaml

  # Bounded rate (3-20%% positives, at least 1000 samples per fold)
  %(prog)s --dataset wanglab/delbert-data --dataset_name DCAF7 --n_folds 20 \\
      --min_positive_rate 0.03 --max_positive_rate 0.20 --min_fold_size 1000 \\
      --output DCAF7_bounded.yaml
        """
    )
    parser.add_argument(
        '--dataset', type=str, required=True,
        help='HuggingFace dataset path (e.g., wanglab/delbert-data) or local path'
    )
    parser.add_argument(
        '--dataset_name', type=str, required=True,
        help='Dataset name/config (e.g., LRRK2, WDR91, DCAF7)'
    )
    parser.add_argument(
        '--n_folds', type=int, default=20,
        help='Target number of CV folds. May return fewer if constraints cannot be met. (default: 20)'
    )
    parser.add_argument(
        '--min_positive_rate', type=float, default=0.04,
        help='Minimum positive rate per fold (default: 0.04 = 4%%)'
    )
    parser.add_argument(
        '--max_positive_rate', type=float, default=None,
        help='Maximum positive rate per fold. When specified, enables bounded mode which '
             'intelligently pairs high-pos and low-pos libraries. (default: None = no upper bound)'
    )
    parser.add_argument(
        '--min_fold_size', type=int, default=1000,
        help='Minimum number of samples per test fold. Only used in bounded mode. (default: 1000)'
    )
    parser.add_argument(
        '--min_positive_count', type=int, default=0,
        help='Minimum number of positives per fold. Only used in non-bounded mode. (default: 0 = no constraint)'
    )
    parser.add_argument(
        '--min_library_size', type=int, default=1000,
        help='Minimum library size to be considered "major" for reporting. (default: 1000)'
    )
    parser.add_argument(
        '--train_only_libraries', type=str, nargs='+', default=None,
        help='Library prefixes to always keep in train (never in test folds). '
             'Example: --train_only_libraries L05 L06'
    )
    parser.add_argument(
        '--seed', type=int, default=37,
        help='Random seed for reproducibility (default: 37)'
    )
    parser.add_argument(
        '--library_column', type=str, default='LIBRARY_ID',
        help='Column name for library IDs (default: LIBRARY_ID)'
    )
    parser.add_argument(
        '--output', type=str, required=True,
        help='Output path for fold assignments YAML'
    )
    args = parser.parse_args()

    # Determine mode
    bounded_mode = args.max_positive_rate is not None

    print("=" * 70)
    if bounded_mode:
        print("Bounded Positive Rate Library Fold Assignment Generator")
    else:
        print("Positive-Balanced Library Fold Assignment Generator")
    print("=" * 70)
    print(f"Dataset: {args.dataset}/{args.dataset_name}")
    print(f"Target folds: {args.n_folds}")
    print(f"Min positive rate: {args.min_positive_rate*100:.1f}%")
    if bounded_mode:
        print(f"Max positive rate: {args.max_positive_rate*100:.1f}%")
        print(f"Min fold size: {args.min_fold_size:,}")
        if args.train_only_libraries:
            print(f"Train-only libraries: {args.train_only_libraries}")
        print(f"Mode: BOUNDED (enforces both min and max positive rate)")
    else:
        if args.min_positive_count > 0:
            print(f"Min positive count: {args.min_positive_count}")
        print(f"Mode: MIN-ONLY (enforces minimum positive rate only)")
    print(f"Seed: {args.seed}")
    print("=" * 70)

    # Load dataset
    print("\nLoading dataset...")
    try:
        # Try HuggingFace Hub first
        dataset = load_dataset(args.dataset, args.dataset_name, split='train')
    except Exception as e:
        print(f"Could not load from HuggingFace Hub: {e}")
        # Try loading from disk
        dataset = load_from_disk(args.dataset)

    n_samples = len(dataset)
    print(f"Loaded {n_samples:,} samples")

    # Extract required columns
    library_ids = np.array(dataset[args.library_column])
    labels = np.array(dataset['LABEL'])
    compound_ids = np.array(dataset['COMPOUND_ID'])

    # Calculate overall stats
    n_positives = int(np.sum(labels))
    overall_pos_rate = n_positives / n_samples
    print(f"Total positives: {n_positives:,} ({overall_pos_rate*100:.2f}%)")

    # Compute expected positives per fold
    expected_pos_per_fold = n_positives / args.n_folds
    expected_samples_per_fold = n_samples / args.n_folds
    min_positives_needed = int(expected_samples_per_fold * args.min_positive_rate)

    print(f"\nExpected per fold:")
    print(f"  Samples: ~{int(expected_samples_per_fold):,}")
    print(f"  Positives (if balanced): ~{int(expected_pos_per_fold)}")
    print(f"  Min positives for {args.min_positive_rate*100:.1f}% threshold: ~{min_positives_needed}")

    # Create fold assignments using appropriate algorithm
    if bounded_mode:
        print("\nCreating bounded positive-rate fold assignments...")
        print(f"Constraints: {args.min_positive_rate*100:.1f}% <= pos_rate <= {args.max_positive_rate*100:.1f}%, size >= {args.min_fold_size}")
        splits, metadata = create_bounded_positive_rate_library_kfold_splits(
            library_ids=library_ids,
            labels=labels,
            n_folds=args.n_folds,
            min_positive_rate=args.min_positive_rate,
            max_positive_rate=args.max_positive_rate,
            min_fold_size=args.min_fold_size,
            min_library_size=args.min_library_size,
            train_only_libraries=args.train_only_libraries,
            seed=args.seed
        )
    else:
        print("\nCreating positive-balanced fold assignments...")
        splits, metadata = create_positive_balanced_library_kfold_splits(
            library_ids=library_ids,
            labels=labels,
            n_folds=args.n_folds,
            min_positive_rate=args.min_positive_rate,
            min_positive_count=args.min_positive_count,
            min_library_size=args.min_library_size,
            seed=args.seed
        )

    # Print summary
    print_fold_summary(metadata)

    # Save fold assignments
    output_path = args.output
    if not Path(output_path).is_absolute():
        output_path = str(project_root / output_path)

    print(f"\nSaving fold assignments to: {output_path}")
    save_fold_assignments(splits, metadata, compound_ids, output_path)

    # Print comparison summary
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)

    actual_n_folds = metadata['n_folds']

    if bounded_mode:
        # Bounded mode summary
        constraints_satisfied = metadata.get('constraints_satisfied', True)
        constraint_violations = metadata.get('constraint_violations', [])

        if constraints_satisfied:
            print(f"SUCCESS: All {actual_n_folds} folds meet constraints!")
            print(f"  Positive rate: {args.min_positive_rate*100:.1f}% - {args.max_positive_rate*100:.1f}%")
            print(f"  Minimum fold size: {args.min_fold_size:,}")
        else:
            print(f"WARNING: {len(constraint_violations)} fold(s) violate constraints:")
            for cv in constraint_violations:
                print(f"  Fold {cv['fold']}: {'; '.join(cv['violations'])}")
    else:
        # Min-only mode summary
        below_threshold = metadata.get('below_threshold_folds', [])
        if below_threshold:
            print(f"WARNING: {len(below_threshold)} fold(s) below {args.min_positive_rate*100:.1f}% threshold")
            print(f"  Fold indices: {below_threshold}")
        else:
            print(f"SUCCESS: All {actual_n_folds} folds meet the {args.min_positive_rate*100:.1f}% threshold!")

    if actual_n_folds < args.n_folds:
        print(f"\nNote: Reduced from {args.n_folds} to {actual_n_folds} folds due to library grouping constraints.")

    # Calculate statistics
    pos_rates = [fs['test_pos_rate'] for fs in metadata['fold_stats']]
    test_sizes = [fs['test_size'] for fs in metadata['fold_stats']]

    print(f"\nTest set positive rates:")
    print(f"  Min: {min(pos_rates)*100:.2f}%")
    print(f"  Max: {max(pos_rates)*100:.2f}%")
    print(f"  Mean: {np.mean(pos_rates)*100:.2f}%")
    print(f"  Std: {np.std(pos_rates)*100:.2f}%")

    print(f"\nTest set sizes:")
    print(f"  Min: {min(test_sizes):,}")
    print(f"  Max: {max(test_sizes):,}")
    print(f"  Mean: {np.mean(test_sizes):,.0f}")
    print(f"  Std: {np.std(test_sizes):,.0f}")

    # Show library groupings for bounded mode
    if bounded_mode and 'library_groups' in metadata:
        n_groups = len(metadata['library_groups'])
        print(f"\nLibrary groupings: {n_groups} groups created")
        print(f"  Too high (>{args.max_positive_rate*100:.0f}%): {metadata.get('n_too_high_libraries', 0)} libraries")
        print(f"  Too low (<{args.min_positive_rate*100:.0f}%): {metadata.get('n_too_low_libraries', 0)} libraries")
        print(f"  In range: {metadata.get('n_in_range_libraries', 0)} libraries")

    print(f"\nOutput saved to: {output_path}")


if __name__ == '__main__':
    main()
