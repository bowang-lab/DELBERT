"""
Split strategies for data preparation.

This module provides helper functions for creating train/val/test splits
with different strategies (random stratified, library-based OOD).

To add a new split strategy:
1. Create functions: create_{strategy}_test_split() and create_{strategy}_train_val_split()
2. Register them in SPLIT_STRATEGIES dict at the bottom of this file
3. The prepare_supervised_data.py script will automatically pick them up

All test split functions should return: (test_dataset, remaining_dataset)
All train/val split functions should return: (train_dataset, val_dataset)
"""

from typing import Tuple, Dict, Any, Callable, List
import re
from collections import Counter

import numpy as np
from datasets import Dataset


# =============================================================================
# Random Stratified Splits (preserves label proportions)
# =============================================================================

def create_random_test_split(
    dataset: Dataset,
    test_fraction: float,
    seed: int = 42,
    label_column: str = "LABEL",
    **kwargs
) -> Tuple[Dataset, Dataset]:
    """
    Create a stratified random test split. Test set is determined FIRST.

    Stratification ensures that label proportions are preserved in both
    the test set and remaining data.

    Args:
        dataset: Full dataset
        test_fraction: Fraction to hold out for test (e.g., 0.1 for 10%)
        seed: Random seed for reproducibility
        label_column: Column containing labels for stratification

    Returns:
        Tuple of (test_dataset, remaining_dataset)
    """
    labels = np.array(dataset[label_column])
    unique_labels = np.unique(labels)

    rng = np.random.default_rng(seed)

    test_indices = []
    remaining_indices = []

    # Stratify by label
    for label in unique_labels:
        label_indices = np.where(labels == label)[0]
        rng.shuffle(label_indices)

        n_test = int(len(label_indices) * test_fraction)

        test_indices.extend(label_indices[:n_test].tolist())
        remaining_indices.extend(label_indices[n_test:].tolist())

    # Shuffle within splits to mix labels
    rng.shuffle(test_indices)
    rng.shuffle(remaining_indices)

    print(f"  Stratified random test split:")
    print(f"    Test: {len(test_indices)} samples")
    print(f"    Remaining: {len(remaining_indices)} samples")

    return dataset.select(test_indices), dataset.select(remaining_indices)


# =============================================================================
# Library-based OOD Splits (entire libraries held out)
# =============================================================================


def extract_library_prefix(library_id: str) -> str:
    """
    Extract library prefix from library ID.

    Examples:
        "L34_001" -> "L34"
        "L34" -> "L34"
        "L34-something" -> "L34"
        "Lib01" -> "LIB01"
        "ABC-123" -> "ABC"
        "12345" -> "12345"

    Tries multiple patterns in order. If none match, returns the entire ID
    (ensuring the library stays together in cross-validation).
    """
    if library_id is None:
        return "UNKNOWN"

    lib_str = str(library_id).strip()
    if not lib_str:
        return "UNKNOWN"

    # Try multiple patterns in order of specificity
    patterns = [
        r'^([A-Za-z]\d{1,3})',       # L01, L123 (letter + 1-3 digits)
        r'^([A-Za-z]+\d+)',           # Lib01, ABC123 (letters + digits)
        r'^([A-Za-z]+)[-_]',          # ABC-, Lib_ (letters followed by separator)
        r'^(\d+)',                    # Pure numbers (12345)
    ]

    for pattern in patterns:
        match = re.match(pattern, lib_str)
        if match:
            return match.group(1).upper()

    # Fallback: use entire ID (ensures library stays together in CV)
    # This is critical - never let an unknown format cause data leakage
    return lib_str.upper()


def compute_library_groups(
    library_values: np.ndarray,
    min_library_size: int = 1000
) -> Tuple[np.ndarray, Dict[str, int], list]:
    """
    Compute library groups by extracting prefixes and merging small libraries.

    Args:
        library_values: Array of raw library IDs
        min_library_size: Libraries smaller than this are merged into "OTHER"

    Returns:
        Tuple of:
        - grouped_libraries: Array of library group for each molecule
        - library_counts: Dict mapping library group to molecule count
        - small_libraries: List of libraries merged into OTHER
    """
    # Extract prefixes
    prefixes = np.array([extract_library_prefix(lib) for lib in library_values])

    # Count molecules per library prefix
    library_counts = Counter(prefixes)

    # Identify small libraries to merge
    small_libraries = [lib for lib, count in library_counts.items() if count < min_library_size]

    # Create grouped library assignments
    grouped_libraries = np.array([
        "OTHER" if lib in small_libraries else lib
        for lib in prefixes
    ])

    # Recount after merging
    final_counts = Counter(grouped_libraries)

    return grouped_libraries, dict(final_counts), small_libraries


def create_library_ood_test_split(
    dataset: Dataset,
    test_fraction: float,
    seed: int = 42,
    library_column: str = "LIBRARY_ID",
    min_library_size: int = 1000,
    **kwargs
) -> Tuple[Dataset, Dataset]:
    """
    Create library-based OOD test split.

    Entire libraries are held out for the test set to evaluate
    out-of-distribution generalization. This ensures that no molecules
    from test libraries appear in training.

    Library IDs are grouped by prefix (e.g., "L34" from "L34_001").
    Libraries with fewer than min_library_size molecules are merged into "OTHER".
    The "OTHER" group is never used for test set.

    Note: This split is NOT stratified by label since we're splitting by library.

    Args:
        dataset: Full dataset
        test_fraction: Fraction of LIBRARIES to hold out (not molecules)
        seed: Random seed for reproducibility
        library_column: Column containing library IDs
        min_library_size: Libraries smaller than this are merged into "OTHER"

    Returns:
        Tuple of (test_dataset, remaining_dataset)
    """
    # Get library values and compute groups
    library_values = np.array(dataset[library_column])
    grouped_libraries, library_counts, small_libraries = compute_library_groups(
        library_values, min_library_size
    )

    print(f"  Library grouping:")
    print(f"    Total libraries (by prefix): {len(library_counts)}")
    print(f"    Small libraries merged into OTHER: {len(small_libraries)}")
    if small_libraries:
        print(f"      Merged: {small_libraries}")
    for lib, count in sorted(library_counts.items(), key=lambda x: -x[1]):
        print(f"      {lib}: {count:,} molecules")

    # Get unique libraries (excluding OTHER from test candidates)
    unique_libraries = [lib for lib in library_counts.keys() if lib != "OTHER"]
    n_libs = len(unique_libraries)
    n_test_libs = max(1, int(n_libs * test_fraction))

    # Shuffle and split libraries
    rng = np.random.default_rng(seed)
    shuffled_libs = list(rng.permutation(unique_libraries))

    test_libs = set(shuffled_libs[:n_test_libs])
    remaining_libs = set(shuffled_libs[n_test_libs:])
    remaining_libs.add("OTHER")  # OTHER always goes to train/val

    # Get indices for each split
    test_indices = [i for i, lib in enumerate(grouped_libraries) if lib in test_libs]
    remaining_indices = [i for i, lib in enumerate(grouped_libraries) if lib in remaining_libs]

    print(f"  Library OOD test split:")
    print(f"    Test libraries: {test_libs}")
    print(f"    Remaining libraries: {remaining_libs}")
    print(f"    Test molecules: {len(test_indices):,}, Remaining molecules: {len(remaining_indices):,}")

    return dataset.select(test_indices), dataset.select(remaining_indices)


def create_random_train_val_split(
    dataset: Dataset,
    train_fraction: float,
    seed: int = 42,
    label_column: str = "LABEL",
    **kwargs
) -> Tuple[Dataset, Dataset]:
    """
    Split data into train/val with stratification.

    Stratification ensures that label proportions are preserved in both
    training and validation sets.

    Args:
        dataset: Dataset (typically after test removal)
        train_fraction: Fraction for training (e.g., 0.8 means 80% train, 20% val)
        seed: Random seed (uses seed+1 internally to ensure independence from test split)
        label_column: Column containing labels for stratification

    Returns:
        Tuple of (train_dataset, val_dataset)
    """
    labels = np.array(dataset[label_column])
    unique_labels = np.unique(labels)

    # Use seed+1 to ensure independence from test split
    rng = np.random.default_rng(seed + 1)

    train_indices = []
    val_indices = []

    # Stratify by label
    for label in unique_labels:
        label_indices = np.where(labels == label)[0]
        rng.shuffle(label_indices)

        n_train = int(len(label_indices) * train_fraction)

        train_indices.extend(label_indices[:n_train].tolist())
        val_indices.extend(label_indices[n_train:].tolist())

    # Shuffle within splits to mix labels
    rng.shuffle(train_indices)
    rng.shuffle(val_indices)

    print(f"  Stratified random train/val split:")
    print(f"    Train: {len(train_indices)} samples")
    print(f"    Val: {len(val_indices)} samples")

    return dataset.select(train_indices), dataset.select(val_indices)


def create_library_ood_train_val_split(
    dataset: Dataset,
    train_fraction: float,
    seed: int = 42,
    library_column: str = "LIBRARY_ID",
    min_library_size: int = 1000,
    **kwargs
) -> Tuple[Dataset, Dataset]:
    """
    Split remaining data into train/val by library.

    Library IDs are grouped by prefix (e.g., "L34" from "L34_001").
    Libraries with fewer than min_library_size molecules are merged into "OTHER".
    The "OTHER" group is always included in training.

    Args:
        dataset: Dataset after test removal (only contains train/val libraries)
        train_fraction: Fraction of remaining libraries for training
        seed: Random seed (uses seed+1 internally)
        library_column: Column containing library IDs
        min_library_size: Libraries smaller than this are merged into "OTHER"

    Returns:
        Tuple of (train_dataset, val_dataset)
    """
    # Get library values and compute groups
    library_values = np.array(dataset[library_column])
    grouped_libraries, library_counts, small_libraries = compute_library_groups(
        library_values, min_library_size
    )

    # Get unique libraries (excluding OTHER from split candidates)
    unique_libraries = [lib for lib in library_counts.keys() if lib != "OTHER"]
    n_libs = len(unique_libraries)
    n_train_libs = int(n_libs * train_fraction)

    # Use seed+1 to ensure independence from test split
    rng = np.random.default_rng(seed + 1)
    shuffled_libs = list(rng.permutation(unique_libraries))

    train_libs = set(shuffled_libs[:n_train_libs])
    train_libs.add("OTHER")  # OTHER always goes to train
    val_libs = set(shuffled_libs[n_train_libs:])

    # Get indices for each split
    train_indices = [i for i, lib in enumerate(grouped_libraries) if lib in train_libs]
    val_indices = [i for i, lib in enumerate(grouped_libraries) if lib in val_libs]

    print(f"  Library OOD train/val split:")
    print(f"    Train libraries: {train_libs}")
    print(f"    Val libraries: {val_libs}")
    print(f"    Train molecules: {len(train_indices):,}, Val molecules: {len(val_indices):,}")

    return dataset.select(train_indices), dataset.select(val_indices)


def get_label_distribution(dataset: Dataset, label_column: str = "LABEL") -> Dict[int, int]:
    """
    Get label distribution statistics.

    Args:
        dataset: Dataset with labels
        label_column: Column containing labels

    Returns:
        Dictionary mapping label values to counts
    """
    labels = np.array(dataset[label_column])
    unique, counts = np.unique(labels, return_counts=True)
    return {int(k): int(v) for k, v in zip(unique, counts)}


def compute_split_statistics(
    train_dataset: Dataset,
    val_dataset: Dataset,
    test_dataset: Dataset,
    label_column: str = "LABEL"
) -> Dict[str, Any]:
    """
    Compute comprehensive statistics for all splits.

    Args:
        train_dataset: Training dataset
        val_dataset: Validation dataset
        test_dataset: Test dataset
        label_column: Column containing labels

    Returns:
        Dictionary with statistics for each split
    """
    stats = {
        "sizes": {
            "train": len(train_dataset),
            "val": len(val_dataset),
            "test": len(test_dataset),
            "total": len(train_dataset) + len(val_dataset) + len(test_dataset)
        },
        "label_distribution": {
            "train": get_label_distribution(train_dataset, label_column),
            "val": get_label_distribution(val_dataset, label_column),
            "test": get_label_distribution(test_dataset, label_column)
        }
    }

    # Compute class ratios
    for split in ["train", "val", "test"]:
        dist = stats["label_distribution"][split]
        total = sum(dist.values())
        stats["label_distribution"][f"{split}_ratios"] = {
            k: round(v / total, 4) for k, v in dist.items()
        }

    return stats


# =============================================================================
# Split Strategy Registry
# =============================================================================
# To add a new split strategy:
# 1. Create create_{name}_test_split() and create_{name}_train_val_split() functions above
# 2. Add an entry to SPLIT_STRATEGIES dict below
# 3. The prepare_supervised_data.py script will automatically use it via get_split_functions()

SPLIT_STRATEGIES: Dict[str, Dict[str, Callable]] = {
    "random": {
        "test_split": create_random_test_split,
        "train_val_split": create_random_train_val_split,
        "description": "Stratified random split (preserves label proportions)"
    },
    "library_ood": {
        "test_split": create_library_ood_test_split,
        "train_val_split": create_library_ood_train_val_split,
        "description": "Library-based OOD split (entire libraries held out)"
    },
}


def get_split_functions(strategy: str) -> Tuple[Callable, Callable]:
    """
    Get the split functions for a given strategy.

    Args:
        strategy: Name of the split strategy (e.g., "random", "library_ood")

    Returns:
        Tuple of (test_split_fn, train_val_split_fn)

    Raises:
        ValueError: If strategy is not registered

    Example:
        test_split_fn, train_val_split_fn = get_split_functions("random")
        test_data, remaining = test_split_fn(dataset, test_fraction=0.1, seed=42, label_column="LABEL")
        train_data, val_data = train_val_split_fn(remaining, train_fraction=0.8, seed=42, label_column="LABEL")
    """
    if strategy not in SPLIT_STRATEGIES:
        available = list(SPLIT_STRATEGIES.keys())
        raise ValueError(
            f"Unknown split strategy: '{strategy}'. "
            f"Available strategies: {available}"
        )

    return (
        SPLIT_STRATEGIES[strategy]["test_split"],
        SPLIT_STRATEGIES[strategy]["train_val_split"]
    )


def list_split_strategies() -> Dict[str, str]:
    """
    List all available split strategies with descriptions.

    Returns:
        Dictionary mapping strategy names to descriptions
    """
    return {
        name: info["description"]
        for name, info in SPLIT_STRATEGIES.items()
    }
