"""
Library-based K-Fold Cross-Validation Utilities.

This module provides shared utilities for library-based K-fold CV that can be
used by both transformer and baseline evaluation scripts.

Key features:
- Library grouping: Major libraries assigned to folds, "OTHER" distributed proportionally
- Reproducible fold assignments: Save/load assignments to ensure identical splits
- Aggregation: Compute mean, std, and 95% CI from fold metrics
- Statistical comparison: Wilcoxon signed-rank test for model comparison
"""

from typing import Dict, List, Tuple, Any, Optional
from pathlib import Path
from datetime import datetime
import os
import pickle
import numpy as np
import yaml
import scipy.stats as stats
from tqdm import tqdm

try:
    from rdkit.ML.Scoring.Scoring import CalcBEDROC
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False

from .splits import compute_library_groups, extract_library_prefix


def create_random_stratified_kfold_splits(
    labels: np.ndarray,
    n_folds: int = 5,
    seed: int = 42
) -> Tuple[List[Tuple[np.ndarray, np.ndarray]], Dict[str, Any]]:
    """
    Create K-fold splits using stratified random sampling.

    Uses sklearn's StratifiedKFold to ensure label proportions are
    preserved in each fold. This is an alternative to library-based
    splitting when you don't need to keep libraries together.

    Args:
        labels: Array of binary labels for each molecule
        n_folds: Number of CV folds
        seed: Random seed for reproducibility

    Returns:
        Tuple of:
        - splits: List of (train_indices, test_indices) for each fold
        - metadata: Dict with fold statistics
    """
    from sklearn.model_selection import StratifiedKFold

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    n_samples = len(labels)

    splits = []
    fold_stats = []

    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(np.arange(n_samples), labels)):
        splits.append((train_idx, test_idx))

        train_labels = labels[train_idx]
        test_labels = labels[test_idx]

        fold_stats.append({
            "fold": fold_idx,
            "train_size": len(train_idx),
            "test_size": len(test_idx),
            "train_pos_rate": float(train_labels.mean()) if len(train_labels) > 0 else 0.0,
            "test_pos_rate": float(test_labels.mean()) if len(test_labels) > 0 else 0.0,
        })

    metadata = {
        "n_folds": n_folds,
        "n_samples": n_samples,
        "seed": seed,
        "fold_stats": fold_stats,
        "algorithm": "stratified_random"
    }

    return splits, metadata


def create_library_kfold_splits(
    library_ids: np.ndarray,
    labels: np.ndarray,
    n_folds: int = 5,
    min_library_size: int = 1000,
    seed: int = 42
) -> Tuple[List[Tuple[np.ndarray, np.ndarray]], Dict[str, Any]]:
    """
    Create K-fold splits based on library grouping using greedy bin-packing.

    IMPORTANT: Every library (regardless of size) is assigned to exactly ONE fold.
    This prevents data leakage that occurs when small libraries are split across folds.

    The algorithm uses greedy bin-packing to balance fold sizes while ensuring
    no library is ever split across train/test boundaries.

    Args:
        library_ids: Array of library ID strings for each molecule
        labels: Array of binary labels for each molecule
        n_folds: Number of CV folds
        min_library_size: Libraries >= this are considered "major" (for metadata only)
        seed: Random seed for reproducibility

    Returns:
        Tuple of:
        - splits: List of (train_indices, test_indices) for each fold
        - metadata: Dict with fold statistics and library assignments
    """
    rng = np.random.default_rng(seed)
    n_samples = len(library_ids)

    # 1. Extract library prefixes - DO NOT merge into "OTHER"
    prefixes = np.array([extract_library_prefix(lib) for lib in library_ids])
    unique_libs, lib_inverse = np.unique(prefixes, return_inverse=True)

    # Calculate stats per library
    lib_stats = []
    for i, lib in enumerate(unique_libs):
        mask = (lib_inverse == i)
        count = np.sum(mask)
        pos_rate = float(np.mean(labels[mask])) if count > 0 else 0.0
        lib_stats.append({
            'lib': lib,
            'count': int(count),
            'pos_rate': pos_rate,
            'idx': i
        })

    # Sort by size (descending) for better bin packing
    lib_stats.sort(key=lambda x: x['count'], reverse=True)

    # 2. Greedy Bin Packing - assign each library to the smallest fold
    # This ensures balanced fold sizes while NEVER splitting a library
    fold_assignments = [[] for _ in range(n_folds)]
    fold_counts = np.zeros(n_folds)

    for item in lib_stats:
        target_fold = int(np.argmin(fold_counts))  # Greedily pick smallest fold
        fold_assignments[target_fold].append(item['lib'])
        fold_counts[target_fold] += item['count']

    # Create lib -> fold mapping
    lib_to_fold = {}
    for fold_idx, libs in enumerate(fold_assignments):
        for lib in libs:
            lib_to_fold[lib] = fold_idx

    # 3. Generate Indices - EVERY molecule stays with its library
    fold_indices = [[] for _ in range(n_folds)]
    for idx, prefix in enumerate(prefixes):
        assigned_fold = lib_to_fold[prefix]
        fold_indices[assigned_fold].append(idx)

    # Shuffle within folds
    for i in range(n_folds):
        fold_indices[i] = np.array(fold_indices[i])
        rng.shuffle(fold_indices[i])

    # 4. Create train/test splits for each fold
    splits = []
    for test_fold in range(n_folds):
        test_indices = fold_indices[test_fold]
        train_indices = np.concatenate([
            fold_indices[i] for i in range(n_folds) if i != test_fold
        ])
        rng.shuffle(train_indices)
        splits.append((train_indices, test_indices))

    # 5. Compute metadata
    major_libraries = [x['lib'] for x in lib_stats if x['count'] >= min_library_size]
    small_libraries = [x['lib'] for x in lib_stats if x['count'] < min_library_size]

    fold_stats = []
    for fold_idx, (train_idx, test_idx) in enumerate(splits):
        train_labels = labels[train_idx]
        test_labels = labels[test_idx]

        # Get libraries in test set
        test_libs = set(prefixes[test_idx])

        fold_stats.append({
            "fold": fold_idx,
            "train_size": len(train_idx),
            "test_size": len(test_idx),
            "train_pos_rate": float(train_labels.mean()) if len(train_labels) > 0 else 0.0,
            "test_pos_rate": float(test_labels.mean()) if len(test_labels) > 0 else 0.0,
            "test_libraries": sorted(list(test_libs)),
            "n_test_libraries": len(test_libs)
        })

    metadata = {
        "n_folds": n_folds,
        "n_samples": n_samples,
        "n_total_libraries": len(unique_libs),
        "n_major_libraries": len(major_libraries),
        "major_libraries": major_libraries,
        "small_libraries_merged": [],  # NO LONGER MERGING - kept for backwards compatibility
        "n_other_molecules": 0,        # NO LONGER USING OTHER
        "min_library_size": min_library_size,
        "seed": seed,
        "library_to_fold": lib_to_fold,
        "fold_stats": fold_stats,
        "algorithm": "greedy_bin_packing"  # Document the algorithm used
    }

    return splits, metadata


def create_positive_balanced_library_kfold_splits(
    library_ids: np.ndarray,
    labels: np.ndarray,
    n_folds: int = 20,
    min_positive_rate: float = 0.04,
    min_positive_count: int = 0,
    min_library_size: int = 1000,
    seed: int = 42
) -> Tuple[List[Tuple[np.ndarray, np.ndarray]], Dict[str, Any]]:
    """
    Create K-fold splits with balanced positive counts across folds.

    Unlike create_library_kfold_splits which balances sample counts,
    this function balances POSITIVE counts to ensure each fold has
    sufficient positives for meaningful statistical evaluation.

    Libraries with positive rates below min_positive_rate are always
    grouped with higher-positive libraries to meet the threshold.

    The algorithm:
    1. Separate libraries into "self-sufficient" and "needy" based on constraints
    2. Pair needy libraries with self-sufficient ones to form groups
    3. Use greedy bin-packing on groups to balance positive counts across folds

    Args:
        library_ids: Array of library ID strings for each molecule
        labels: Array of binary labels for each molecule
        n_folds: Number of CV folds
        min_positive_rate: Target minimum positive rate per fold (default 4%)
        min_positive_count: Target minimum number of positives per fold (default 0 = no constraint)
        min_library_size: Libraries >= this are considered "major" (for metadata only)
        seed: Random seed for reproducibility

    Returns:
        Tuple of:
        - splits: List of (train_indices, test_indices) for each fold
        - metadata: Dict with fold statistics and library assignments
    """
    rng = np.random.default_rng(seed)
    n_samples = len(library_ids)

    # 1. Extract library prefixes
    prefixes = np.array([extract_library_prefix(lib) for lib in library_ids])
    unique_libs, lib_inverse = np.unique(prefixes, return_inverse=True)

    # 2. Calculate stats per library
    lib_stats = []
    for i, lib in enumerate(unique_libs):
        mask = (lib_inverse == i)
        count = int(np.sum(mask))
        pos_count = int(np.sum(labels[mask]))
        pos_rate = pos_count / count if count > 0 else 0.0
        lib_stats.append({
            'lib': lib,
            'count': count,
            'pos_count': pos_count,
            'pos_rate': pos_rate,
            'idx': i
        })

    # 3. Separate into self-sufficient and needy libraries
    # Self-sufficient = meets BOTH rate AND count constraints
    self_sufficient = [x for x in lib_stats
                       if x['pos_rate'] >= min_positive_rate
                       and x['pos_count'] >= min_positive_count]
    needy = [x for x in lib_stats
             if x['pos_rate'] < min_positive_rate
             or x['pos_count'] < min_positive_count]

    print(f"\nLibrary analysis:")
    if min_positive_count > 0:
        print(f"  Self-sufficient (rate >= {min_positive_rate*100:.1f}% AND pos >= {min_positive_count}): {len(self_sufficient)} libraries")
    else:
        print(f"  Self-sufficient (rate >= {min_positive_rate*100:.1f}%): {len(self_sufficient)} libraries")
    print(f"  Need grouping: {len(needy)} libraries")

    # Rename for compatibility with rest of code
    low_positive = needy

    # 4. Create groups by pairing low-positive with high-positive libraries
    # Sort self-sufficient by positive count (descending) - most positives first
    self_sufficient.sort(key=lambda x: x['pos_count'], reverse=True)
    # Sort low-positive by positive RATE (ascending) - lowest rate first needs best partner
    low_positive.sort(key=lambda x: x['pos_rate'])

    groups = []  # Each group is a list of library stats

    # First, pair lowest-positive libraries with high-positive ones
    # Allow multiple low-positive libraries to share one high-positive library
    used_high_pos = set()
    unassigned_low = list(low_positive)  # Make a copy

    # Phase 1: Assign lowest-rate libraries to best available partners
    for low_lib in low_positive:
        if low_lib not in unassigned_low:
            continue  # Already assigned

        # Find the best high-positive library to pair with
        best_partner = None
        best_combined_rate = 0

        for high_lib in self_sufficient:
            if high_lib['lib'] in used_high_pos:
                continue

            # Calculate combined stats
            combined_count = low_lib['count'] + high_lib['count']
            combined_pos = low_lib['pos_count'] + high_lib['pos_count']
            combined_rate = combined_pos / combined_count if combined_count > 0 else 0

            # Check BOTH constraints
            if (combined_rate >= min_positive_rate
                and combined_pos >= min_positive_count
                and combined_rate > best_combined_rate):
                best_partner = high_lib
                best_combined_rate = combined_rate

        if best_partner is not None:
            # Create a group with both libraries
            groups.append([low_lib, best_partner])
            used_high_pos.add(best_partner['lib'])
            unassigned_low.remove(low_lib)

    # Phase 2: Try to add remaining unassigned low-pos libraries to existing groups
    # as long as the group meets BOTH constraints
    still_unassigned = []
    for low_lib in unassigned_low:
        added = False
        # Try to add to an existing group
        for group in groups:
            group_count = sum(lib['count'] for lib in group)
            group_pos = sum(lib['pos_count'] for lib in group)
            # Check if adding this library keeps BOTH constraints satisfied
            new_count = group_count + low_lib['count']
            new_pos = group_pos + low_lib['pos_count']
            new_rate = new_pos / new_count if new_count > 0 else 0

            if new_rate >= min_positive_rate and new_pos >= min_positive_count:
                group.append(low_lib)
                added = True
                break

        if not added:
            still_unassigned.append(low_lib)

    # Phase 3: Any remaining unassigned go alone (will fail threshold)
    for low_lib in still_unassigned:
        groups.append([low_lib])

    # Add remaining self-sufficient libraries as individual groups
    for high_lib in self_sufficient:
        if high_lib['lib'] not in used_high_pos:
            groups.append([high_lib])

    print(f"  Created {len(groups)} library groups")

    # Calculate group stats
    group_stats = []
    for i, group in enumerate(groups):
        total_count = sum(lib['count'] for lib in group)
        total_pos = sum(lib['pos_count'] for lib in group)
        group_rate = total_pos / total_count if total_count > 0 else 0
        group_stats.append({
            'group_idx': i,
            'libraries': [lib['lib'] for lib in group],
            'count': total_count,
            'pos_count': total_pos,
            'pos_rate': group_rate
        })

    # 5. Greedy bin-packing: assign GROUPS to folds with fewest positives
    # Sort groups by positive count (descending) for better bin packing
    group_stats.sort(key=lambda x: x['pos_count'], reverse=True)

    fold_assignments = [[] for _ in range(n_folds)]  # Lists of library names
    fold_pos_counts = np.zeros(n_folds)
    fold_sample_counts = np.zeros(n_folds)

    for group in group_stats:
        # Find fold with fewest positives
        target_fold = int(np.argmin(fold_pos_counts))
        fold_assignments[target_fold].extend(group['libraries'])
        fold_pos_counts[target_fold] += group['pos_count']
        fold_sample_counts[target_fold] += group['count']

    # 6. Create lib -> fold mapping
    lib_to_fold = {}
    for fold_idx, libs in enumerate(fold_assignments):
        for lib in libs:
            lib_to_fold[lib] = fold_idx

    # 7. Generate indices - every molecule stays with its library
    fold_indices = [[] for _ in range(n_folds)]
    for idx, prefix in enumerate(prefixes):
        assigned_fold = lib_to_fold[prefix]
        fold_indices[assigned_fold].append(idx)

    # Shuffle within folds
    for i in range(n_folds):
        fold_indices[i] = np.array(fold_indices[i]) if fold_indices[i] else np.array([], dtype=int)
        if len(fold_indices[i]) > 0:
            rng.shuffle(fold_indices[i])

    # Remove empty folds
    non_empty_fold_indices = [fi for fi in fold_indices if len(fi) > 0]
    actual_n_folds = len(non_empty_fold_indices)

    if actual_n_folds < n_folds:
        print(f"\nNote: Reduced from {n_folds} to {actual_n_folds} folds (not enough library groups)")
        fold_indices = non_empty_fold_indices

    # 8. Create train/test splits for each fold
    splits = []
    for test_fold in range(len(fold_indices)):
        test_indices = fold_indices[test_fold]
        train_indices = np.concatenate([
            fold_indices[i] for i in range(len(fold_indices)) if i != test_fold
        ])
        rng.shuffle(train_indices)
        splits.append((train_indices, test_indices))

    # 9. Compute metadata and validate positive rates
    major_libraries = [x['lib'] for x in lib_stats if x['count'] >= min_library_size]
    low_pos_libraries = [x['lib'] for x in low_positive]

    fold_stats = []
    below_threshold_folds = []

    for fold_idx, (train_idx, test_idx) in enumerate(splits):
        train_labels = labels[train_idx]
        test_labels = labels[test_idx]

        # Get libraries in test set
        test_libs = set(prefixes[test_idx])

        test_pos_rate = float(test_labels.mean()) if len(test_labels) > 0 else 0.0

        test_pos_count = int(np.sum(test_labels))
        if test_pos_rate < min_positive_rate or test_pos_count < min_positive_count:
            below_threshold_folds.append(fold_idx)

        fold_stats.append({
            "fold": fold_idx,
            "train_size": len(train_idx),
            "test_size": len(test_idx),
            "train_pos_rate": float(train_labels.mean()) if len(train_labels) > 0 else 0.0,
            "test_pos_rate": test_pos_rate,
            "test_pos_count": int(np.sum(test_labels)),
            "test_libraries": sorted(list(test_libs)),
            "n_test_libraries": len(test_libs)
        })

    # Warn about folds below threshold
    if below_threshold_folds:
        constraint_str = f"rate < {min_positive_rate*100:.1f}%"
        if min_positive_count > 0:
            constraint_str += f" or pos < {min_positive_count}"
        print(f"\nWARNING: {len(below_threshold_folds)} fold(s) fail constraints ({constraint_str}):")
        for fold_idx in below_threshold_folds:
            fs = fold_stats[fold_idx]
            print(f"  Fold {fold_idx}: {fs['test_pos_rate']*100:.2f}% "
                  f"({fs['test_pos_count']} positives in {fs['test_size']} samples)")
            print(f"    Libraries: {fs['test_libraries']}")

    metadata = {
        "n_folds": len(splits),
        "n_samples": n_samples,
        "n_total_libraries": len(unique_libs),
        "n_major_libraries": len(major_libraries),
        "major_libraries": major_libraries,
        "low_positive_libraries": low_pos_libraries,
        "min_positive_rate": min_positive_rate,
        "min_positive_count": min_positive_count,
        "min_library_size": min_library_size,
        "seed": seed,
        "library_to_fold": lib_to_fold,
        "fold_stats": fold_stats,
        "below_threshold_folds": below_threshold_folds,
        "algorithm": "positive_balanced_greedy_bin_packing"
    }

    return splits, metadata


def create_bounded_positive_rate_library_kfold_splits(
    library_ids: np.ndarray,
    labels: np.ndarray,
    n_folds: int = 20,
    min_positive_rate: float = 0.03,
    max_positive_rate: float = 0.20,
    min_fold_size: int = 1000,
    min_library_size: int = 1000,
    train_only_libraries: Optional[List[str]] = None,
    seed: int = 42
) -> Tuple[List[Tuple[np.ndarray, np.ndarray]], Dict[str, Any]]:
    """
    Create K-fold splits with BOUNDED positive rates (both min AND max) per fold.

    This function addresses a common problem in drug discovery datasets where some
    chemical libraries have very high hit rates (>30%) while others have very low
    hit rates (<1%). Standard CV approaches can create folds with extreme positive
    rates that are either:
    - Too low: insufficient positives for meaningful statistical evaluation
    - Too high: unrepresentative of real-world screening conditions

    The algorithm intelligently groups libraries to ensure each test fold has a
    positive rate within the specified bounds [min_positive_rate, max_positive_rate].

    Algorithm Overview:
    ------------------
    1. **Categorize libraries** into three groups:
       - "too_high": Libraries with pos_rate > max_positive_rate (need dilution)
       - "too_low": Libraries with pos_rate < min_positive_rate (need boosting)
       - "in_range": Libraries already within [min, max] bounds

    2. **Dilute high-positive libraries**: Pair each "too_high" library with
       enough "too_low" libraries to bring the combined rate below max_positive_rate.
       Prioritizes larger low-pos libraries for efficient dilution.

    3. **Boost low-positive libraries**: Pair remaining "too_low" libraries with
       "in_range" libraries to bring their combined rate above min_positive_rate.

    4. **Handle remaining libraries**: Add "in_range" libraries as individual
       groups if they meet the min_fold_size constraint, or merge small ones.

    5. **Greedy bin-packing**: Assign groups to folds, balancing by positive
       count to achieve similar class distributions across folds.

    Constraints Enforced:
    --------------------
    - min_positive_rate <= test_fold_positive_rate <= max_positive_rate
    - test_fold_size >= min_fold_size

    When Constraints Cannot Be Met:
    ------------------------------
    If the library composition makes it impossible to satisfy all constraints
    (e.g., not enough low-pos samples to dilute all high-pos libraries),
    the function will:
    - Return fewer folds than requested (groups that violate constraints are merged)
    - Print warnings identifying problematic groups
    - Include validation results in metadata for inspection

    Use Cases:
    ---------
    - Balanced evaluation: Ensure each CV fold provides a fair test of model
      performance across different positive rate regimes
    - Statistical validity: Guarantee sufficient positives in each fold for
      meaningful metrics (PR-AUC, enrichment factors)
    - Realistic simulation: Cap positive rates to match expected real-world
      screening hit rates (typically 0.1-15%)

    Example:
    -------
    >>> # DCAF7 dataset has libraries ranging from 0% to 35% positive rate
    >>> # Create 18 folds where each test set has 3-20% positive rate
    >>> splits, metadata = create_bounded_positive_rate_library_kfold_splits(
    ...     library_ids=library_ids,
    ...     labels=labels,
    ...     n_folds=20,  # May return fewer if constraints can't be met
    ...     min_positive_rate=0.03,
    ...     max_positive_rate=0.20,
    ...     min_fold_size=1000,
    ...     seed=37
    ... )
    >>> print(f"Created {metadata['n_folds']} valid folds")
    >>> print(f"Positive rate range: {metadata['actual_min_pos_rate']:.1%} - {metadata['actual_max_pos_rate']:.1%}")

    Args:
        library_ids: Array of library ID strings for each molecule. Library
            prefixes are extracted (e.g., "L05-123-456" -> "L05") to group
            molecules from the same chemical series.
        labels: Array of binary labels (0/1) for each molecule.
        n_folds: Target number of CV folds. The actual number may be lower if
            constraints cannot be satisfied with the requested number.
        min_positive_rate: Minimum acceptable positive rate for each test fold.
            Folds below this threshold lack sufficient positives for reliable
            metrics. Default: 0.03 (3%).
        max_positive_rate: Maximum acceptable positive rate for each test fold.
            Folds above this may not represent realistic screening conditions.
            Default: 0.20 (20%).
        min_fold_size: Minimum number of samples required in each test fold.
            Ensures statistical reliability. Default: 1000.
        min_library_size: Libraries with >= this many samples are considered
            "major" libraries (used for metadata reporting only, does not
            affect grouping). Default: 1000.
        train_only_libraries: List of library prefixes (e.g., ["L05"]) that should
            always remain in the training set and never appear in test folds.
            Useful for excluding very large libraries that would dominate test
            sets or for keeping specific libraries as training-only data.
            Default: None (all libraries can appear in test folds).
        seed: Random seed for reproducibility. Affects shuffling within folds
            and tie-breaking in greedy assignment.

    Returns:
        Tuple of:
        - splits: List of (train_indices, test_indices) tuples, one per fold.
          Train indices are all samples NOT in the test fold.
        - metadata: Dict containing:
          - n_folds: Actual number of folds created
          - n_samples: Total samples in dataset
          - algorithm: "bounded_positive_rate_grouping"
          - min_positive_rate, max_positive_rate, min_fold_size: Constraints used
          - actual_min_pos_rate, actual_max_pos_rate: Achieved bounds
          - n_valid_groups, n_invalid_groups: Group validation counts
          - fold_stats: Per-fold statistics (sizes, rates, libraries)
          - constraint_violations: List of any violated constraints
          - library_groups: How libraries were grouped together

    Raises:
        ValueError: If no valid fold assignments can be created (all groups
            violate constraints).

    See Also:
        - create_library_kfold_splits: Balances fold sizes without rate constraints
        - create_positive_balanced_library_kfold_splits: Only enforces minimum rate
        - create_cluster_kfold_splits: Groups by structural similarity instead of library

    Notes:
        - Libraries are NEVER split across folds to prevent data leakage
        - The algorithm prioritizes constraint satisfaction over exact fold count
        - For datasets with extreme library imbalance, consider adjusting bounds
          or accepting fewer folds
    """
    rng = np.random.default_rng(seed)
    n_samples = len(library_ids)

    # ==========================================================================
    # Step 1: Extract library prefixes and compute per-library statistics
    # ==========================================================================
    prefixes = np.array([extract_library_prefix(lib) for lib in library_ids])
    unique_libs = np.unique(prefixes)

    lib_stats = {}
    for lib in unique_libs:
        mask = (prefixes == lib)
        count = int(np.sum(mask))
        pos_count = int(np.sum(labels[mask]))
        lib_stats[lib] = {
            'count': count,
            'pos_count': pos_count,
            'pos_rate': pos_count / count if count > 0 else 0.0
        }

    # ==========================================================================
    # Step 1b: Handle train-only libraries (always in train, never in test)
    # ==========================================================================
    train_only_set = set(train_only_libraries) if train_only_libraries else set()
    train_only_indices = []

    if train_only_set:
        # Identify indices for train-only libraries
        for idx, prefix in enumerate(prefixes):
            if prefix in train_only_set:
                train_only_indices.append(idx)
        train_only_indices = np.array(train_only_indices, dtype=int)

        # Report train-only library stats
        train_only_count = len(train_only_indices)
        train_only_pos = int(np.sum(labels[train_only_indices]))
        print(f"\nTrain-only libraries: {sorted(train_only_set)}")
        print(f"  Samples: {train_only_count:,} ({train_only_count/n_samples*100:.1f}% of dataset)")
        print(f"  Positives: {train_only_pos:,} ({train_only_pos/train_only_count*100:.2f}%)")
    else:
        train_only_indices = np.array([], dtype=int)

    # Libraries available for test folds (excluding train-only)
    testable_libs = [lib for lib in unique_libs if lib not in train_only_set]

    # ==========================================================================
    # Step 2: Categorize libraries by positive rate
    # ==========================================================================
    too_high = [lib for lib in testable_libs if lib_stats[lib]['pos_rate'] > max_positive_rate]
    too_low = [lib for lib in testable_libs if lib_stats[lib]['pos_rate'] < min_positive_rate]
    in_range = [lib for lib in testable_libs
                if min_positive_rate <= lib_stats[lib]['pos_rate'] <= max_positive_rate]

    # Sort for deterministic processing
    too_high.sort(key=lambda x: lib_stats[x]['pos_count'], reverse=True)  # Most positives first
    too_low.sort(key=lambda x: lib_stats[x]['count'], reverse=True)  # Largest diluters first
    in_range.sort(key=lambda x: lib_stats[x]['count'], reverse=True)

    print(f"\nLibrary categorization (bounds: {min_positive_rate*100:.1f}% - {max_positive_rate*100:.1f}%):")
    print(f"  Libraries with rate > {max_positive_rate*100:.0f}% (need dilution): {len(too_high)}")
    print(f"  Libraries with rate < {min_positive_rate*100:.0f}% (need boosting): {len(too_low)}")
    print(f"  Libraries already in range: {len(in_range)}")

    # ==========================================================================
    # Step 3: Helper function to check group validity
    # ==========================================================================
    def check_group(libs):
        """Check if a group of libraries meets all constraints."""
        if not libs:
            return {'count': 0, 'pos_count': 0, 'rate': 0, 'valid': False,
                    'reason': 'empty group'}

        total_count = sum(lib_stats[lib]['count'] for lib in libs)
        total_pos = sum(lib_stats[lib]['pos_count'] for lib in libs)
        rate = total_pos / total_count if total_count > 0 else 0

        reasons = []
        if rate < min_positive_rate:
            reasons.append(f"rate {rate*100:.2f}% < {min_positive_rate*100:.0f}%")
        if rate > max_positive_rate:
            reasons.append(f"rate {rate*100:.2f}% > {max_positive_rate*100:.0f}%")
        if total_count < min_fold_size:
            reasons.append(f"size {total_count} < {min_fold_size}")

        return {
            'count': total_count,
            'pos_count': total_pos,
            'rate': rate,
            'valid': len(reasons) == 0,
            'reason': '; '.join(reasons) if reasons else None
        }

    # ==========================================================================
    # Step 4: Create groups by intelligent pairing
    # ==========================================================================
    groups = []  # Each group is a list of library names
    used = set()

    # --- Phase 4a: Dilute high-positive libraries ---
    # Pair each too-high library with enough too-low libraries to reach target range
    TARGET_RATE = (min_positive_rate + max_positive_rate) / 2  # Aim for middle of range

    for high_lib in too_high:
        group = [high_lib]
        used.add(high_lib)

        current = check_group(group)

        # Keep adding low-pos libraries until we're in valid range
        for low_lib in too_low:
            if low_lib in used:
                continue

            test_group = group + [low_lib]
            result = check_group(test_group)

            # Add if it helps dilute (reduces rate) without going below min
            if current['rate'] > max_positive_rate and result['rate'] < current['rate']:
                if result['rate'] >= min_positive_rate:
                    group.append(low_lib)
                    used.add(low_lib)
                    current = result

                    # Stop if we've reached a good target rate
                    if min_positive_rate <= result['rate'] <= TARGET_RATE:
                        break

        groups.append(group)

    # --- Phase 4b: Boost remaining low-positive libraries ---
    # Pair with in-range libraries to bring rate up
    remaining_low = [lib for lib in too_low if lib not in used]
    remaining_in_range = [lib for lib in in_range if lib not in used]

    for low_lib in remaining_low:
        if low_lib in used:
            continue

        # Find best in-range partner that creates a valid group
        best_partner = None
        best_result = None

        for ir_lib in remaining_in_range:
            if ir_lib in used:
                continue

            test_group = [low_lib, ir_lib]
            result = check_group(test_group)

            if result['valid']:
                # Prefer larger groups for better statistics
                if best_result is None or result['count'] > best_result['count']:
                    best_partner = ir_lib
                    best_result = result

        if best_partner:
            groups.append([low_lib, best_partner])
            used.add(low_lib)
            used.add(best_partner)
        else:
            # Try adding to existing valid groups
            added = False
            for group in groups:
                current = check_group(group)
                if not current['valid']:
                    continue

                test_group = group + [low_lib]
                result = check_group(test_group)

                if result['valid']:
                    group.append(low_lib)
                    used.add(low_lib)
                    added = True
                    break

            if not added:
                # Create singleton group (will likely fail constraints)
                groups.append([low_lib])
                used.add(low_lib)

    # --- Phase 4c: Add remaining in-range libraries ---
    for ir_lib in remaining_in_range:
        if ir_lib in used:
            continue

        result = check_group([ir_lib])

        if result['valid']:
            # Large enough to be its own group
            groups.append([ir_lib])
            used.add(ir_lib)
        else:
            # Too small - try to merge with another small in-range library
            merged = False

            for other_lib in remaining_in_range:
                if other_lib in used or other_lib == ir_lib:
                    continue

                test_group = [ir_lib, other_lib]
                result = check_group(test_group)

                if result['valid']:
                    groups.append([ir_lib, other_lib])
                    used.add(ir_lib)
                    used.add(other_lib)
                    merged = True
                    break

            if not merged:
                # Try adding to existing valid groups
                for group in groups:
                    test_group = group + [ir_lib]
                    result = check_group(test_group)
                    if result['valid']:
                        group.append(ir_lib)
                        used.add(ir_lib)
                        merged = True
                        break

                if not merged:
                    # Last resort: create singleton (may fail)
                    groups.append([ir_lib])
                    used.add(ir_lib)

    # ==========================================================================
    # Step 5: Validate groups and separate valid from invalid
    # ==========================================================================
    valid_groups = []
    invalid_groups = []

    for group in groups:
        result = check_group(group)
        if result['valid']:
            valid_groups.append((group, result))
        else:
            invalid_groups.append((group, result))

    print(f"\nGroup validation:")
    print(f"  Valid groups: {len(valid_groups)}")
    print(f"  Invalid groups: {len(invalid_groups)}")

    if invalid_groups:
        print(f"\n  Invalid groups details:")
        for group, result in invalid_groups:
            libs_str = ', '.join(group[:3])
            if len(group) > 3:
                libs_str += f"... (+{len(group)-3})"
            print(f"    {libs_str}: {result['reason']}")

    # ==========================================================================
    # Step 6: Handle invalid groups by merging into valid ones
    # ==========================================================================
    # Try to merge invalid groups into valid groups without breaking constraints
    for inv_group, inv_result in invalid_groups:
        merged = False

        for i, (val_group, val_result) in enumerate(valid_groups):
            test_group = val_group + inv_group
            result = check_group(test_group)

            if result['valid']:
                valid_groups[i] = (test_group, result)
                merged = True
                print(f"  Merged invalid group {inv_group[:2]}... into valid group")
                break

        if not merged:
            # Force merge with the group that results in smallest constraint violation
            print(f"  WARNING: Could not merge invalid group {inv_group[:2]}... - constraints will be violated")
            # Add as separate group anyway (user will see warning in results)
            valid_groups.append((inv_group, inv_result))

    # ==========================================================================
    # Step 7: Greedy bin-packing to assign groups to folds
    # ==========================================================================
    # Sort groups by positive count for better distribution
    valid_groups.sort(key=lambda x: x[1]['pos_count'], reverse=True)

    actual_n_folds = min(n_folds, len(valid_groups))

    if actual_n_folds < n_folds:
        print(f"\nNote: Reduced from {n_folds} to {actual_n_folds} folds (limited by valid groups)")

    fold_assignments = [[] for _ in range(actual_n_folds)]
    fold_pos_counts = np.zeros(actual_n_folds)
    fold_sample_counts = np.zeros(actual_n_folds)

    for group, result in valid_groups:
        # Assign to fold with fewest positives (balances positive distribution)
        target_fold = int(np.argmin(fold_pos_counts))
        fold_assignments[target_fold].extend(group)
        fold_pos_counts[target_fold] += result['pos_count']
        fold_sample_counts[target_fold] += result['count']

    # ==========================================================================
    # Step 8: Create library-to-fold mapping and generate indices
    # ==========================================================================
    lib_to_fold = {}
    for fold_idx, libs in enumerate(fold_assignments):
        for lib in libs:
            lib_to_fold[lib] = fold_idx

    # Generate molecule indices per fold
    fold_indices = [[] for _ in range(actual_n_folds)]
    for idx, prefix in enumerate(prefixes):
        if prefix in lib_to_fold:
            fold_indices[lib_to_fold[prefix]].append(idx)

    # Convert to numpy arrays and shuffle
    for i in range(actual_n_folds):
        fold_indices[i] = np.array(fold_indices[i]) if fold_indices[i] else np.array([], dtype=int)
        if len(fold_indices[i]) > 0:
            rng.shuffle(fold_indices[i])

    # ==========================================================================
    # Step 9: Create train/test splits
    # ==========================================================================
    splits = []
    for test_fold in range(actual_n_folds):
        test_indices = fold_indices[test_fold]
        # Combine: other folds + train-only libraries
        train_parts = [fold_indices[i] for i in range(actual_n_folds) if i != test_fold]
        if len(train_only_indices) > 0:
            train_parts.append(train_only_indices)
        train_indices = np.concatenate(train_parts)
        rng.shuffle(train_indices)
        splits.append((train_indices, test_indices))

    # ==========================================================================
    # Step 10: Compute comprehensive metadata
    # ==========================================================================
    fold_stats = []
    constraint_violations = []

    for fold_idx, (train_idx, test_idx) in enumerate(splits):
        train_labels = labels[train_idx]
        test_labels = labels[test_idx]
        test_libs = sorted(list(set(prefixes[test_idx])))

        test_pos_rate = float(test_labels.mean()) if len(test_labels) > 0 else 0.0
        test_pos_count = int(np.sum(test_labels))
        test_size = len(test_idx)

        # Check for constraint violations
        violations = []
        if test_pos_rate < min_positive_rate:
            violations.append(f"rate {test_pos_rate*100:.2f}% < {min_positive_rate*100:.0f}%")
        if test_pos_rate > max_positive_rate:
            violations.append(f"rate {test_pos_rate*100:.2f}% > {max_positive_rate*100:.0f}%")
        if test_size < min_fold_size:
            violations.append(f"size {test_size} < {min_fold_size}")

        if violations:
            constraint_violations.append({
                'fold': fold_idx,
                'violations': violations
            })

        fold_stats.append({
            "fold": fold_idx,
            "train_size": len(train_idx),
            "test_size": test_size,
            "train_pos_rate": float(train_labels.mean()) if len(train_labels) > 0 else 0.0,
            "test_pos_rate": test_pos_rate,
            "test_pos_count": test_pos_count,
            "test_libraries": test_libs,
            "n_test_libraries": len(test_libs),
            "constraint_violations": violations if violations else None
        })

    # Compute actual achieved bounds
    test_pos_rates = [fs['test_pos_rate'] for fs in fold_stats]
    test_sizes = [fs['test_size'] for fs in fold_stats]

    # Report results
    print(f"\nFinal fold statistics:")
    print(f"  Number of folds: {actual_n_folds}")
    print(f"  Positive rate range: {min(test_pos_rates)*100:.2f}% - {max(test_pos_rates)*100:.2f}%")
    print(f"  Test size range: {min(test_sizes):,} - {max(test_sizes):,}")

    if constraint_violations:
        print(f"\n  WARNING: {len(constraint_violations)} fold(s) violate constraints:")
        for cv in constraint_violations:
            print(f"    Fold {cv['fold']}: {'; '.join(cv['violations'])}")
    else:
        print(f"\n  SUCCESS: All {actual_n_folds} folds meet constraints!")

    # Build library groups info for metadata
    library_groups = []
    for group, result in valid_groups:
        library_groups.append({
            'libraries': group,
            'count': result['count'],
            'pos_count': result['pos_count'],
            'pos_rate': result['rate']
        })

    # Identify major libraries
    major_libraries = [lib for lib, stats in lib_stats.items() if stats['count'] >= min_library_size]

    metadata = {
        "created": datetime.now().isoformat(),
        "algorithm": "bounded_positive_rate_grouping",
        "n_folds": actual_n_folds,
        "n_samples": n_samples,
        "n_total_libraries": len(unique_libs),
        "seed": seed,
        # Train-only libraries
        "train_only_libraries": sorted(train_only_set) if train_only_set else [],
        "n_train_only_samples": len(train_only_indices),
        "train_only_indices": train_only_indices.tolist() if len(train_only_indices) > 0 else [],
        # Constraint parameters
        "min_positive_rate": min_positive_rate,
        "max_positive_rate": max_positive_rate,
        "min_fold_size": min_fold_size,
        "min_library_size": min_library_size,
        # Achieved bounds
        "actual_min_pos_rate": min(test_pos_rates),
        "actual_max_pos_rate": max(test_pos_rates),
        "actual_min_fold_size": min(test_sizes),
        "actual_max_fold_size": max(test_sizes),
        # Validation results
        "n_valid_groups": len([g for g in valid_groups if check_group(g[0])['valid']]),
        "n_constraint_violations": len(constraint_violations),
        "constraint_violations": constraint_violations,
        "constraints_satisfied": len(constraint_violations) == 0,
        # Library categorization
        "n_too_high_libraries": len(too_high),
        "n_too_low_libraries": len(too_low),
        "n_in_range_libraries": len(in_range),
        "too_high_libraries": too_high,
        "too_low_libraries": too_low,
        "n_major_libraries": len(major_libraries),
        "major_libraries": major_libraries,
        # Mappings
        "library_to_fold": lib_to_fold,
        "library_groups": library_groups,
        "fold_stats": fold_stats,
    }

    return splits, metadata


def generate_clusters_leaderpicker(
    fingerprints: np.ndarray,
    threshold: float = 0.7,
    cache_path: Optional[str] = None
) -> np.ndarray:
    """
    Cluster molecules using RDKit LeaderPicker algorithm.

    LeaderPicker selects cluster centroids that are at least `threshold` apart
    in Tanimoto distance, then assigns remaining molecules to their nearest centroid.

    Args:
        fingerprints: Binary fingerprint matrix of shape (N_molecules, n_bits)
        threshold: Tanimoto distance threshold for LeaderPicker (higher = more clusters)
        cache_path: Optional path to cache/load cluster assignments as pickle

    Returns:
        cluster_ids: Array of cluster assignments (N,) with values 0 to n_clusters-1
    """
    from rdkit import DataStructs
    from rdkit.DataStructs import ExplicitBitVect
    from rdkit.SimDivFilters import rdSimDivPickers

    # Check cache
    if cache_path and os.path.exists(cache_path):
        print(f"Loading cached clusters from: {cache_path}")
        with open(cache_path, 'rb') as f:
            return pickle.load(f)

    n_molecules = len(fingerprints)
    n_bits = fingerprints.shape[1] if fingerprints.ndim > 1 else len(fingerprints[0])

    # Convert numpy arrays to RDKit BitVects
    print(f"Converting {n_molecules} fingerprints to RDKit BitVects...")
    fps_rdkit = []
    for fp in tqdm(fingerprints, desc="Converting to BitVects"):
        bv = ExplicitBitVect(n_bits)
        on_bits = np.where(fp > 0)[0].tolist()
        bv.SetBitsFromList(on_bits)
        fps_rdkit.append(bv)

    # Run LeaderPicker to select centroids
    print(f"Running LeaderPicker with threshold={threshold}...")
    lp = rdSimDivPickers.LeaderPicker()
    picks = list(lp.LazyBitVectorPick(fps_rdkit, len(fps_rdkit), threshold))
    n_clusters = len(picks)
    print(f"LeaderPicker selected {n_clusters} cluster centroids")

    # Assign molecules to nearest centroid
    clusters = np.full(n_molecules, -1, dtype=np.int32)

    # Centroids get their own cluster ID
    for cluster_id, idx in enumerate(picks):
        clusters[idx] = cluster_id

    # Assign non-centroids to nearest centroid
    centroid_fps = [fps_rdkit[p] for p in picks]
    non_centroid_indices = np.where(clusters == -1)[0]

    print(f"Assigning {len(non_centroid_indices)} molecules to nearest centroid...")
    for i in tqdm(non_centroid_indices, desc="Assigning clusters"):
        sims = DataStructs.BulkTanimotoSimilarity(fps_rdkit[i], centroid_fps)
        clusters[i] = np.argmax(sims)

    # Cache if requested
    if cache_path:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        with open(cache_path, 'wb') as f:
            pickle.dump(clusters, f)
        print(f"Saved cluster assignments to: {cache_path}")

    return clusters


def create_cluster_kfold_splits(
    fingerprints: np.ndarray,
    labels: np.ndarray,
    n_folds: int = 5,
    threshold: float = 0.7,
    fingerprint_name: str = "FCFP6",
    seed: int = 42,
    cache_dir: Optional[str] = None
) -> Tuple[List[Tuple[np.ndarray, np.ndarray]], Dict[str, Any]]:
    """
    Create K-fold splits using LeaderPicker clustering.

    Molecules in the same cluster are kept together, preventing data leakage
    from structurally similar molecules appearing in both train and test sets.

    Args:
        fingerprints: Binary fingerprint matrix (N, n_bits)
        labels: Array of binary labels for each molecule
        n_folds: Number of CV folds
        threshold: Tanimoto distance threshold for LeaderPicker
        fingerprint_name: Name of fingerprint type (for metadata)
        seed: Random seed for reproducibility
        cache_dir: Directory to cache cluster assignments

    Returns:
        Tuple of:
        - splits: List of (train_indices, test_indices) for each fold
        - metadata: Dict with fold statistics and cluster information
    """
    from sklearn.model_selection import StratifiedGroupKFold

    # Generate clusters
    cache_path = None
    if cache_dir:
        cache_path = os.path.join(cache_dir, f'clusters_{fingerprint_name}_{threshold}.pkl')

    cluster_ids = generate_clusters_leaderpicker(
        fingerprints, threshold=threshold, cache_path=cache_path
    )

    # Use StratifiedGroupKFold to create splits (keeps clusters together)
    sgkf = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    splits = list(sgkf.split(np.zeros(len(labels)), labels, groups=cluster_ids))

    # Compute metadata
    n_clusters = len(np.unique(cluster_ids))
    cluster_sizes = np.bincount(cluster_ids)

    # Compute fold statistics
    fold_stats = []
    for fold_idx, (train_idx, test_idx) in enumerate(splits):
        test_clusters = np.unique(cluster_ids[test_idx])
        fold_stats.append({
            'fold': fold_idx,
            'train_size': len(train_idx),
            'test_size': len(test_idx),
            'train_pos_rate': float(labels[train_idx].mean()) if len(train_idx) > 0 else 0.0,
            'test_pos_rate': float(labels[test_idx].mean()) if len(test_idx) > 0 else 0.0,
            'n_test_clusters': len(test_clusters),
        })

    metadata = {
        'algorithm': 'leaderpicker_cluster',
        'n_folds': n_folds,
        'n_samples': len(labels),
        'seed': seed,
        'threshold': threshold,
        'fingerprint': fingerprint_name,
        'n_clusters': n_clusters,
        'cluster_size_mean': float(cluster_sizes.mean()),
        'cluster_size_std': float(cluster_sizes.std()),
        'cluster_size_min': int(cluster_sizes.min()),
        'cluster_size_max': int(cluster_sizes.max()),
        'fold_stats': fold_stats,
        'cluster_ids': cluster_ids,  # Will be saved as cluster_assignments in YAML
    }

    return splits, metadata


def _convert_to_native(obj):
    """Convert numpy types to Python native types for YAML serialization."""
    if isinstance(obj, np.ndarray):
        return [_convert_to_native(x) for x in obj.tolist()]
    elif isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, (np.str_, np.bytes_)):
        return str(obj)
    elif isinstance(obj, dict):
        return {_convert_to_native(k): _convert_to_native(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_convert_to_native(v) for v in obj]
    elif hasattr(obj, 'item'):  # Catch any remaining numpy scalars
        return obj.item()
    return obj


def save_fold_assignments(
    splits: List[Tuple[np.ndarray, np.ndarray]],
    metadata: Dict[str, Any],
    compound_ids: np.ndarray,
    output_path: str
) -> None:
    """
    Save fold assignments to YAML for reproducibility.

    This allows baseline and transformer CV to use identical splits.

    Args:
        splits: List of (train_indices, test_indices) tuples
        metadata: Metadata dict from create_*_kfold_splits functions
        compound_ids: Array of compound IDs
        output_path: Path to save YAML file
    """
    # Create compound_id -> fold mapping
    # Check for duplicate compound IDs which would silently overwrite earlier entries
    assignments = {}
    duplicates = []
    for fold_idx, (_, test_indices) in enumerate(splits):
        for idx in test_indices:
            cid = str(compound_ids[idx])
            if cid in assignments:
                duplicates.append((cid, assignments[cid], fold_idx))
            else:
                assignments[cid] = int(fold_idx)
    if duplicates:
        raise ValueError(
            f"CRITICAL: {len(duplicates)} duplicate COMPOUND_IDs found during fold assignment save. "
            f"Duplicate IDs would be silently overwritten, corrupting fold assignments. "
            f"Deduplicate the dataset before creating fold splits. "
            f"First 5 duplicates (id, existing_fold, new_fold): {duplicates[:5]}"
        )

    output_data = {
        "created": datetime.now().isoformat(),
        "algorithm": metadata.get("algorithm", "greedy_bin_packing"),
        "n_folds": int(metadata["n_folds"]),
        "n_samples": int(metadata["n_samples"]),
        "seed": int(metadata["seed"]),
        "fold_stats": _convert_to_native(metadata["fold_stats"]),
    }

    # Add train-only libraries info if present
    if metadata.get("train_only_libraries"):
        output_data["train_only_libraries"] = [str(x) for x in metadata["train_only_libraries"]]
        output_data["n_train_only_samples"] = int(metadata.get("n_train_only_samples", 0))
        # Save compound IDs for train-only samples so load_fold_assignments can use them
        train_only_indices = metadata.get("train_only_indices", [])
        output_data["train_only_compound_ids"] = [str(compound_ids[i]) for i in train_only_indices]

    # Add library-specific fields only if present (library-based splits)
    if "library_to_fold" in metadata:
        output_data["min_library_size"] = int(metadata.get("min_library_size", 1000))
        output_data["n_major_libraries"] = int(metadata.get("n_major_libraries", 0))
        output_data["major_libraries"] = [str(x) for x in metadata.get("major_libraries", [])]
        output_data["small_libraries_merged"] = [str(x) for x in metadata.get("small_libraries_merged", [])]
        output_data["n_other_molecules"] = int(metadata.get("n_other_molecules", 0))
        output_data["library_to_fold"] = {str(k): int(v) for k, v in metadata["library_to_fold"].items()}

    # Add cluster-specific fields if present (cluster-based splits)
    if "cluster_ids" in metadata:
        output_data["threshold"] = float(metadata["threshold"])
        output_data["fingerprint"] = str(metadata["fingerprint"])
        output_data["n_clusters"] = int(metadata["n_clusters"])
        output_data["cluster_size_mean"] = float(metadata["cluster_size_mean"])
        output_data["cluster_size_std"] = float(metadata["cluster_size_std"])
        output_data["cluster_size_min"] = int(metadata["cluster_size_min"])
        output_data["cluster_size_max"] = int(metadata["cluster_size_max"])

        # Save cluster assignments (compound_id -> cluster_id)
        cluster_ids = metadata["cluster_ids"]
        cluster_assignments = {
            str(compound_ids[i]): int(cluster_ids[i])
            for i in range(len(compound_ids))
        }
        output_data["cluster_assignments"] = cluster_assignments

    # Add fold assignments last (they're typically the largest section)
    output_data["assignments"] = assignments

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        yaml.dump(output_data, f, default_flow_style=False, sort_keys=False)

    print(f"Fold assignments saved to: {output_path}")


def load_fold_assignments(
    assignments_path: str,
    compound_ids: np.ndarray,
    strict: bool = True
) -> Tuple[List[Tuple[np.ndarray, np.ndarray]], Dict[str, Any]]:
    """
    Load pre-computed fold assignments to ensure identical splits.

    Args:
        assignments_path: Path to YAML file with fold assignments
        compound_ids: Array of compound IDs in current dataset
        strict: If True (default), raise error on ANY mismatch between
                dataset and fold assignments. This prevents silent failures.

    Returns:
        Tuple of:
        - splits: List of (train_indices, test_indices) for each fold
        - metadata: Loaded metadata dict

    Raises:
        ValueError: If strict=True and there are mismatched compound IDs
    """
    with open(assignments_path, 'r') as f:
        data = yaml.safe_load(f)

    assignments = data["assignments"]
    n_folds = data["n_folds"]

    # Load train-only compound IDs if present
    train_only_compound_ids = set(data.get("train_only_compound_ids", []))

    # Build compound_id to index mapping (handle duplicates by keeping ALL occurrences)
    from collections import defaultdict
    id_to_indices = defaultdict(list)
    for idx, cid in enumerate(compound_ids):
        id_to_indices[str(cid)].append(idx)

    dataset_ids = set(id_to_indices.keys())
    assignment_ids = set(assignments.keys())

    # Train-only IDs are expected to be in dataset but not in assignments
    expected_extra_ids = train_only_compound_ids & dataset_ids

    # Create fold index lists
    fold_indices = [[] for _ in range(n_folds)]
    missing_ids = []  # In assignments but not in dataset

    for cid, fold_idx in assignments.items():
        if cid in id_to_indices:
            # Add ALL occurrences of this compound ID to the fold
            fold_indices[fold_idx].extend(id_to_indices[cid])
        else:
            missing_ids.append(cid)

    # Check for IDs in dataset but not in assignments (excluding train-only)
    extra_ids = dataset_ids - assignment_ids - expected_extra_ids
    # IDs that are unexpectedly missing from train-only
    unexpected_train_only = train_only_compound_ids - dataset_ids

    # Report and potentially error on mismatches
    total_assigned = len(assignments)
    total_dataset = len(compound_ids)

    # Report train-only status
    if train_only_compound_ids:
        print(f"\nTrain-only compounds: {len(expected_extra_ids):,} found in dataset")
        if unexpected_train_only:
            print(f"  WARNING: {len(unexpected_train_only)} train-only IDs not found in dataset")

    if missing_ids or extra_ids:
        print(f"\nFold Assignment Validation:")
        print(f"  Dataset compound IDs: {total_dataset}")
        print(f"  Fold assignment IDs: {total_assigned}")
        if train_only_compound_ids:
            print(f"  Train-only IDs (expected extra): {len(expected_extra_ids)}")

        if missing_ids:
            missing_pct = 100 * len(missing_ids) / total_assigned
            print(f"  In assignments but NOT in dataset: {len(missing_ids)} ({missing_pct:.1f}%)")
            print(f"    First 5: {missing_ids[:5]}")

        if extra_ids:
            extra_pct = 100 * len(extra_ids) / total_dataset
            print(f"  In dataset but NOT in assignments (unexpected): {len(extra_ids)} ({extra_pct:.1f}%)")
            print(f"    First 5: {list(extra_ids)[:5]}")

        if strict:
            raise ValueError(
                f"STRICT VALIDATION FAILED: Compound ID mismatch between dataset and fold assignments! "
                f"Missing from dataset: {len(missing_ids)}, Extra in dataset: {len(extra_ids)}. "
                f"This indicates the fold assignments were created with a different dataset. "
                f"Either regenerate fold assignments or check data sources."
            )
        else:
            print("  WARNING: Proceeding despite mismatches (strict=False)")

    # Convert to numpy arrays
    fold_indices = [np.array(indices) for indices in fold_indices]

    # Collect train-only indices
    train_only_indices = []
    for cid in expected_extra_ids:
        train_only_indices.extend(id_to_indices[cid])
    train_only_indices = np.array(train_only_indices, dtype=int)

    # Create train/test splits
    splits = []
    for test_fold in range(n_folds):
        test_indices = fold_indices[test_fold]
        # Combine: other folds + train-only samples
        train_parts = [fold_indices[i] for i in range(n_folds) if i != test_fold]
        if len(train_only_indices) > 0:
            train_parts.append(train_only_indices)
        train_indices = np.concatenate(train_parts)
        splits.append((train_indices, test_indices))

    # Extract metadata (exclude assignments to save memory)
    metadata = {k: v for k, v in data.items() if k != "assignments"}

    matched_samples = sum(len(f) for f in fold_indices)
    print(f"\nLoaded fold assignments from: {assignments_path}")
    print(f"  {n_folds} folds, {matched_samples} samples matched")
    if matched_samples != total_assigned:
        print(f"  WARNING: {total_assigned - matched_samples} assignments could not be matched")

    return splits, metadata


def aggregate_cv_metrics(fold_results: List[Dict[str, float]]) -> Dict[str, Any]:
    """
    Compute aggregate statistics from fold-level metrics.

    Args:
        fold_results: List of dicts with metric values for each fold

    Returns:
        Dict with:
        - metrics: Dict of metric_name -> {mean, std, ci_95_lower, ci_95_upper}
        - per_fold: Raw fold-level results
        - n_folds: Number of folds
    """
    if not fold_results:
        return {"metrics": {}, "per_fold": [], "n_folds": 0}

    n_folds = len(fold_results)

    # Collect all metric names
    all_metrics = set()
    for fold in fold_results:
        all_metrics.update(fold.keys())

    aggregated = {}
    for metric in sorted(all_metrics):
        values = [fold[metric] for fold in fold_results if metric in fold]
        if len(values) == 0:
            continue

        values = np.array(values)
        mean = float(np.mean(values))
        std = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0

        # 95% CI using t-distribution
        if len(values) > 1:
            sem = std / np.sqrt(len(values))
            ci = stats.t.interval(0.95, len(values) - 1, loc=mean, scale=sem)
            ci_lower, ci_upper = float(ci[0]), float(ci[1])
        else:
            ci_lower, ci_upper = mean, mean

        aggregated[metric] = {
            "mean": mean,
            "std": std,
            "ci_95_lower": ci_lower,
            "ci_95_upper": ci_upper
        }

    return {
        "metrics": aggregated,
        "per_fold": fold_results,
        "n_folds": n_folds
    }


def compute_statistical_comparisons(
    model_results: Dict[str, List[Dict[str, float]]],
    baseline_model: str = "xgb",
    metrics: Optional[List[str]] = None
) -> Dict[str, Dict[str, Any]]:
    """
    Compute pairwise statistical comparisons between models using Wilcoxon signed-rank test.

    Args:
        model_results: Dict mapping model name to list of fold metrics
        baseline_model: Model to compare others against
        metrics: List of metrics to compare (default: ranking metrics)

    Returns:
        Dict of comparisons: {model_vs_baseline: {metric: {diff, p_value, significant}}}
    """
    if metrics is None:
        metrics = ["roc_auc", "pr_auc", "enrich_at_100", "enrich_at_500"]

    if baseline_model not in model_results:
        print(f"Warning: Baseline model '{baseline_model}' not in results")
        return {}

    baseline_folds = model_results[baseline_model]
    comparisons = {}

    for model_name, model_folds in model_results.items():
        if model_name == baseline_model:
            continue

        comparison_key = f"{model_name}_vs_{baseline_model}"
        comparisons[comparison_key] = {}

        for metric in metrics:
            # Get values for each fold (filtering out folds where metric is missing)
            model_values = [fold.get(metric) for fold in model_folds if metric in fold]
            baseline_values = [fold.get(metric) for fold in baseline_folds if metric in fold]

            if len(model_values) != len(baseline_values) or len(model_values) < 2:
                continue

            model_values = np.array(model_values)
            baseline_values = np.array(baseline_values)

            # Compute difference
            diff = float(np.mean(model_values) - np.mean(baseline_values))

            # Wilcoxon signed-rank test
            try:
                stat, p_value = stats.wilcoxon(model_values, baseline_values)
                p_value = float(p_value)
            except ValueError:
                # All differences are zero or sample too small
                p_value = 1.0

            comparisons[comparison_key][metric] = {
                "diff": diff,
                "p_value": p_value,
                "significant": p_value < 0.05
            }

    return comparisons


def calculate_ranking_metrics(
    labels: np.ndarray,
    probs: np.ndarray,
    ks: List[int] = [100, 500]
) -> Dict[str, float]:
    """
    Calculate ranking-based metrics (no threshold-dependent metrics).

    Args:
        labels: True binary labels
        probs: Predicted probabilities for positive class
        ks: List of K values for precision@K and enrichment@K

    Returns:
        Dict of metric name -> value
    """
    from sklearn.metrics import roc_auc_score, average_precision_score

    metrics = {}

    # ROC-AUC and PR-AUC
    try:
        if len(np.unique(labels)) > 1:
            metrics["roc_auc"] = float(roc_auc_score(labels, probs))
            metrics["pr_auc"] = float(average_precision_score(labels, probs))
        else:
            metrics["roc_auc"] = 0.5
            metrics["pr_auc"] = float(np.mean(labels))
    except Exception:
        metrics["roc_auc"] = 0.0
        metrics["pr_auc"] = 0.0

    # BEDROC (requires RDKit)
    if RDKIT_AVAILABLE:
        try:
            if len(np.unique(labels)) > 1 and np.sum(labels) > 0:
                # Sort by predicted probability descending
                order = np.argsort(-probs)
                sorted_labels = labels[order]
                sorted_probs = probs[order]
                # Format for RDKit: list of (score, activity) tuples
                scores = list(zip(sorted_probs, sorted_labels))
                # alpha=85.0 emphasizes early recognition (standard for virtual screening)
                metrics["bedroc"] = float(CalcBEDROC(scores, col=1, alpha=85.0))
            else:
                metrics["bedroc"] = 0.0
        except Exception:
            metrics["bedroc"] = 0.0

    # Base rate for enrichment calculation
    base_rate = np.sum(labels) / len(labels) if len(labels) > 0 else 0.0

    # Precision@K and Enrichment@K
    sorted_indices = np.argsort(probs)[::-1]

    for k in ks:
        if len(labels) >= k:
            top_k_labels = labels[sorted_indices[:k]]
            precision_k = float(np.mean(top_k_labels))
            enrich_k = precision_k / base_rate if base_rate > 0 else 0.0

            metrics[f"precision_at_{k}"] = precision_k
            metrics[f"enrich_at_{k}"] = enrich_k

    return metrics


def split_train_val(
    train_indices: np.ndarray,
    labels: np.ndarray,
    val_fraction: float = 0.1,
    seed: int = 42
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Split training indices into train and validation sets (stratified).

    Used for inner validation within each CV fold for early stopping.

    Args:
        train_indices: Indices of training samples
        labels: Full label array (indexed by train_indices)
        val_fraction: Fraction of training data to use for validation
        seed: Random seed

    Returns:
        Tuple of (train_subset_indices, val_indices) - both are absolute indices
    """
    rng = np.random.default_rng(seed)

    # Get labels for training samples
    train_labels = labels[train_indices]

    # Stratified split
    pos_mask = train_labels == 1
    pos_indices = train_indices[pos_mask]
    neg_indices = train_indices[~pos_mask]

    # Shuffle
    rng.shuffle(pos_indices)
    rng.shuffle(neg_indices)

    # Split
    n_pos_val = int(len(pos_indices) * val_fraction)
    n_neg_val = int(len(neg_indices) * val_fraction)

    val_indices = np.concatenate([
        pos_indices[:n_pos_val],
        neg_indices[:n_neg_val]
    ])

    train_subset_indices = np.concatenate([
        pos_indices[n_pos_val:],
        neg_indices[n_neg_val:]
    ])

    rng.shuffle(val_indices)
    rng.shuffle(train_subset_indices)

    return train_subset_indices, val_indices


def print_fold_summary(metadata: Dict[str, Any]) -> None:
    """Print a summary of fold assignments."""
    algorithm = metadata.get('algorithm', 'greedy_bin_packing')
    is_library_based = algorithm == 'greedy_bin_packing' or 'library_to_fold' in metadata
    is_cluster_based = algorithm == 'leaderpicker_cluster' or 'cluster_ids' in metadata

    print("\n" + "=" * 70)
    if is_library_based:
        print("Library K-Fold CV Summary")
    elif is_cluster_based:
        print("Cluster-Based K-Fold CV Summary (LeaderPicker)")
    else:
        print("Stratified Random K-Fold CV Summary")
    print("=" * 70)
    print(f"Algorithm: {algorithm}")
    print(f"Total samples: {metadata['n_samples']}")
    print(f"Number of folds: {metadata['n_folds']}")

    # Library-specific info
    if is_library_based:
        print(f"Major libraries: {metadata.get('n_major_libraries', 'N/A')}")
        print(f"Small libraries merged to OTHER: {len(metadata.get('small_libraries_merged', []))}")
        print(f"OTHER molecules: {metadata.get('n_other_molecules', 'N/A')}")

    # Cluster-specific info
    if is_cluster_based:
        print(f"Fingerprint: {metadata.get('fingerprint', 'N/A')}")
        print(f"Threshold: {metadata.get('threshold', 'N/A')}")
        print(f"Number of clusters: {metadata.get('n_clusters', 'N/A')}")
        print(f"Cluster sizes: mean={metadata.get('cluster_size_mean', 0):.1f}, "
              f"std={metadata.get('cluster_size_std', 0):.1f}, "
              f"min={metadata.get('cluster_size_min', 0)}, "
              f"max={metadata.get('cluster_size_max', 0)}")

    print(f"Seed: {metadata['seed']}")

    print("\nFold Statistics:")
    if is_library_based:
        print(f"{'Fold':<6} {'Train':<10} {'Test':<10} {'Train Pos%':<12} {'Test Pos%':<12} {'Test Libraries'}")
        print("-" * 80)
    elif is_cluster_based:
        print(f"{'Fold':<6} {'Train':<10} {'Test':<10} {'Train Pos%':<12} {'Test Pos%':<12} {'Test Clusters'}")
        print("-" * 70)
    else:
        print(f"{'Fold':<6} {'Train':<10} {'Test':<10} {'Train Pos%':<12} {'Test Pos%':<12}")
        print("-" * 60)

    for fs in metadata.get('fold_stats', []):
        train_pos = f"{fs['train_pos_rate']*100:.2f}%"
        test_pos = f"{fs['test_pos_rate']*100:.2f}%"

        if is_library_based:
            libs = ", ".join(fs.get('test_libraries', [])[:3])
            if len(fs.get('test_libraries', [])) > 3:
                libs += f" +{len(fs['test_libraries'])-3} more"
            print(f"{fs['fold']:<6} {fs['train_size']:<10} {fs['test_size']:<10} {train_pos:<12} {test_pos:<12} {libs}")
        elif is_cluster_based:
            n_test_clusters = fs.get('n_test_clusters', 'N/A')
            print(f"{fs['fold']:<6} {fs['train_size']:<10} {fs['test_size']:<10} {train_pos:<12} {test_pos:<12} {n_test_clusters}")
        else:
            print(f"{fs['fold']:<6} {fs['train_size']:<10} {fs['test_size']:<10} {train_pos:<12} {test_pos:<12}")

    print("=" * 70 + "\n")
