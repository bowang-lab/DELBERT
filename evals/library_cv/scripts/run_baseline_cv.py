#!/usr/bin/env python3
"""
Library-based K-Fold Cross-Validation for Baseline Models.

Uses the SAME fold assignments as transformer CV for fair comparison.

Supports: XGBoost, LightGBM, RandomForest, CatBoost
Fingerprints: ECFP4, FCFP6, ATOMPAIR, TOPTOR, MACCS, etc.

Inner ensemble (per fold): StratifiedGroupKFold with n_members splits,
matching the ensembling strategy in compare_baseline.py.

Usage:
    python evals/library_cv/scripts/run_baseline_cv.py \\
        --config evals/library_cv/configs/baseline_cv.yaml
"""

import argparse
import os
import sys
from pathlib import Path
from datetime import datetime
from time import time
from typing import Dict, List, Any, Optional, Tuple
from copy import deepcopy

# Suppress warnings and set env BEFORE importing ML libraries
import warnings
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['PYOPENCL_COMPILER_OUTPUT'] = '0'

import numpy as np
import pandas as pd
import polars as pl
import yaml
from tqdm import tqdm
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

# Import model libraries BEFORE cv_utils (which imports scipy)
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent

from delbert.data.cv_utils import (
    create_library_kfold_splits,
    create_random_stratified_kfold_splits,
    create_cluster_kfold_splits,
    save_fold_assignments,
    load_fold_assignments,
    aggregate_cv_metrics,
    compute_statistical_comparisons,
    calculate_ranking_metrics,
    print_fold_summary
)

try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

try:
    from catboost import CatBoostClassifier
    HAS_CATBOOST = True
except ImportError:
    HAS_CATBOOST = False

# RDKit imports for LeaderPicker clustering (Wellnitz method)
import pickle
from rdkit import DataStructs
from rdkit.DataStructs import ExplicitBitVect
from rdkit.SimDivFilters import rdSimDivPickers


# =============================================================================
# Model Factory (from compare_baseline.py)
# =============================================================================

def _get_models():
    """Build model factory dict with available models."""
    def make_lgbm(n_estimators, n_jobs, model_config):
        lgbm_config = model_config.get('lgbm', {})
        device = lgbm_config.get('device', 'cpu')
        scale_pos_weight = lgbm_config.get('scale_pos_weight', 1.0)
        kwargs = {
            'n_estimators': n_estimators,
            'n_jobs': n_jobs,
            'verbose': -1,
            'scale_pos_weight': scale_pos_weight,
        }
        if device == 'gpu':
            kwargs['device'] = 'gpu'
            kwargs['gpu_platform_id'] = lgbm_config.get('gpu_platform_id', 0)
            kwargs['gpu_device_id'] = lgbm_config.get('gpu_device_id', 0)
        return LGBMClassifier(**kwargs)

    def make_xgb(n_estimators, n_jobs, model_config):
        xgb_config = model_config.get('xgb', {})
        tree_method = xgb_config.get('tree_method', 'hist')
        device = xgb_config.get('device', 'cpu')
        scale_pos_weight = xgb_config.get('scale_pos_weight', 1.0)
        return XGBClassifier(
            n_estimators=n_estimators, n_jobs=n_jobs, verbosity=0,
            tree_method=tree_method, device=device,
            scale_pos_weight=scale_pos_weight
        )

    models = {
        'lgbm': make_lgbm,
        'rf': lambda n_estimators, n_jobs, model_config: RandomForestClassifier(
            n_estimators=n_estimators, n_jobs=n_jobs
        ),
    }
    if HAS_XGBOOST:
        models['xgb'] = make_xgb
    if HAS_CATBOOST:
        models['catboost'] = lambda n_estimators, n_jobs, model_config: CatBoostClassifier(
            n_estimators=n_estimators, verbose=0, thread_count=n_jobs if n_jobs > 0 else None
        )
    return models

MODELS = _get_models()

# Fingerprint column mapping (b-prefix = binarized version of same column)
FINGERPRINT_MAPPING = {
    'bECFP4': 'ECFP4', 'bECFP6': 'ECFP6', 'bFCFP4': 'FCFP4', 'bFCFP6': 'FCFP6',
    'ECFP4': 'ECFP4', 'ECFP6': 'ECFP6', 'FCFP4': 'FCFP4', 'FCFP6': 'FCFP6',
    'MACCS': 'MACCS', 'RDK': 'RDK', 'AVALON': 'AVALON',
    'ATOMPAIR': 'ATOMPAIR', 'TOPTOR': 'TOPTOR',
    'bATOMPAIR': 'ATOMPAIR', 'bTOPTOR': 'TOPTOR',
}


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file with OmegaConf interpolation (like Hydra)."""
    from omegaconf import OmegaConf
    config = OmegaConf.load(config_path)
    # Resolve all interpolations and convert to plain dict
    return OmegaConf.to_container(config, resolve=True)


def load_fingerprints_from_parquet(
    parquet_path: str,
    compound_ids: np.ndarray,
    fingerprint_cols: List[str]
) -> Tuple[Dict[str, np.ndarray], np.ndarray, np.ndarray]:
    """Load fingerprints and labels from parquet, filtered to matching COMPOUND_IDs."""
    columns = ['COMPOUND_ID', 'LABEL', 'LIBRARY_ID'] + fingerprint_cols

    print(f"  Loading columns: {columns}")
    df = pl.read_parquet(parquet_path, columns=columns)

    # Drop duplicates BEFORE filtering to ensure 1:1 mapping with compound_ids
    n_before = len(df)
    df = df.unique(subset=['COMPOUND_ID'], keep='first')
    n_after = len(df)
    if n_before != n_after:
        print(f"  WARNING: Removed {n_before - n_after} duplicate COMPOUND_IDs from parquet")

    # Filter to matching COMPOUND_IDs
    # Handle both string and int COMPOUND_ID columns by converting to match parquet type
    parquet_id_dtype = df['COMPOUND_ID'].dtype
    if parquet_id_dtype == pl.Utf8 or parquet_id_dtype == pl.String:
        # Parquet has string IDs - keep as strings
        compound_ids_list = [str(cid) for cid in compound_ids.tolist()]
    else:
        # Parquet has numeric IDs - convert to int
        compound_ids_list = [int(cid) for cid in compound_ids.tolist()]
    df_filtered = df.filter(pl.col('COMPOUND_ID').is_in(compound_ids_list))

    # Reorder to match input compound_ids order
    id_order = {cid: i for i, cid in enumerate(compound_ids_list)}
    df_filtered = df_filtered.with_columns(
        pl.col('COMPOUND_ID').replace_strict(id_order, default=None).alias('_order')
    ).sort('_order').drop('_order')

    # Validate row count matches expected (ensures 1:1 mapping with input compound_ids)
    if len(df_filtered) != len(compound_ids_list):
        raise ValueError(
            f"Row count mismatch after filtering: got {len(df_filtered)} rows "
            f"but expected {len(compound_ids_list)}. This may indicate duplicate "
            f"COMPOUND_IDs in the parquet file or missing compounds."
        )

    # Extract fingerprints as numpy arrays
    fingerprints = {}
    for fp_col in fingerprint_cols:
        fp_series = df_filtered[fp_col]
        fp_array = np.vstack(fp_series.to_list())
        fingerprints[fp_col] = fp_array
        print(f"  Loaded {fp_col}: shape={fp_array.shape}")

    labels = df_filtered['LABEL'].to_numpy().astype(np.int32)
    library_ids = df_filtered['LIBRARY_ID'].to_numpy()

    print(f"  Labels: {len(labels)} samples, {labels.sum()} positives ({100*labels.mean():.2f}%)")

    return fingerprints, labels, library_ids


def binarize_fingerprints(fingerprints: Dict[str, np.ndarray], fp_types: List[str]) -> Dict[str, np.ndarray]:
    """Convert count-based fingerprints to binary if requested."""
    result = {}
    for fp_type in fp_types:
        parquet_col = FINGERPRINT_MAPPING.get(fp_type, fp_type)
        fp_array = fingerprints[parquet_col]

        if fp_type.startswith('b'):
            fp_array = (fp_array > 0).astype(np.int32)

        result[fp_type] = fp_array

    return result


def library_ids_to_groups(library_ids: np.ndarray) -> np.ndarray:
    """Convert string LIBRARY_IDs to numeric group IDs.

    Uses extract_library_prefix() for consistency with the transformer pipeline
    and fold assignment creation (which both use the same function).
    """
    from delbert.data.splits import extract_library_prefix
    prefixes = np.array([extract_library_prefix(lid) for lid in library_ids])
    unique_prefixes = np.unique(prefixes)
    prefix_to_id = {prefix: i for i, prefix in enumerate(unique_prefixes)}
    groups = np.array([prefix_to_id[p] for p in prefixes], dtype=np.int32)
    return groups


def load_fingerprints_for_clustering(fp_name: str, parquet_path: str) -> np.ndarray:
    """
    Load a single fingerprint column from parquet for clustering.

    Args:
        fp_name: Fingerprint name (e.g., 'FCFP6', 'ECFP4')
        parquet_path: Path to parquet file

    Returns:
        Binary fingerprint matrix of shape (N, n_bits)
    """
    print(f"Loading {fp_name} fingerprints from parquet for clustering...")
    df = pl.read_parquet(parquet_path, columns=[fp_name])
    fingerprints = np.vstack(df[fp_name].to_list())
    print(f"  Loaded {len(fingerprints)} fingerprints, shape: {fingerprints.shape}")

    # Binarize if needed (convert counts to binary)
    if fingerprints.max() > 1:
        print(f"  Binarizing {fp_name} fingerprints (max value={fingerprints.max()})")
        fingerprints = (fingerprints > 0).astype(np.int32)

    return fingerprints


def generate_clusters_leaderpicker(
    fingerprints: np.ndarray,
    threshold: float = 0.7,
    cache_path: Optional[str] = None
) -> np.ndarray:
    """
    Generate cluster assignments using LeaderPicker algorithm (Wellnitz method).

    Args:
        fingerprints: Binary fingerprints array (N, dim)
        threshold: Tanimoto similarity threshold for clustering
        cache_path: Optional path to cache/load clusters

    Returns:
        clusters: (N,) array of cluster IDs
    """
    # Check cache
    if cache_path and os.path.exists(cache_path):
        print(f"  Loading clusters from cache: {cache_path}")
        with open(cache_path, 'rb') as f:
            return pickle.load(f)

    n_molecules = len(fingerprints)
    print(f"  Generating clusters for {n_molecules} molecules...")

    # Convert numpy arrays to RDKit BitVects
    fps_rdkit = []
    for fp in tqdm(fingerprints, desc="  Converting to BitVects"):
        bv = ExplicitBitVect(len(fp))
        on_bits = np.where(fp > 0)[0].tolist()
        bv.SetBitsFromList(on_bits)
        fps_rdkit.append(bv)

    # Run LeaderPicker
    print(f"  Running LeaderPicker with threshold {threshold}...")
    lp = rdSimDivPickers.LeaderPicker()
    picks = list(lp.LazyBitVectorPick(fps_rdkit, len(fps_rdkit), threshold))
    print(f"  Found {len(picks)} cluster centroids")

    # Initialize cluster assignments
    clusters = np.full(n_molecules, -1, dtype=np.int32)
    for cluster_id, idx in enumerate(picks):
        clusters[idx] = cluster_id

    # Assign non-centroids to nearest centroid
    centroid_fps = [fps_rdkit[p] for p in picks]
    non_centroid_indices = np.where(clusters == -1)[0]
    print(f"  Assigning {len(non_centroid_indices)} molecules to clusters...")

    for i in tqdm(non_centroid_indices, desc="  Assigning clusters"):
        sims = DataStructs.BulkTanimotoSimilarity(fps_rdkit[i], centroid_fps)
        clusters[i] = np.argmax(sims)

    n_clusters = len(np.unique(clusters))
    print(f"  Total clusters: {n_clusters}")

    # Cache results
    if cache_path:
        cache_dir = os.path.dirname(cache_path)
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)
        print(f"  Caching clusters to: {cache_path}")
        with open(cache_path, 'wb') as f:
            pickle.dump(clusters, f)

    return clusters


def train_inner_ensemble(
    train_data: Dict[str, np.ndarray],
    labels: np.ndarray,
    groups: Optional[np.ndarray],
    config: dict,
    model_type: str
) -> List[List]:
    """
    Train ensemble of models using StratifiedGroupKFold (same as compare_baseline.py).

    Returns:
        models: List of lists [[model_fp1, model_fp2, ...], ...]
                Outer list = ensemble members, inner list = fingerprint models
    """
    n_ensemble = config['ensemble']['n_members']
    n_estimators = config['model']['n_estimators']
    n_jobs = config['model']['n_jobs']
    model_config = config['model']
    shuffle = config['ensemble'].get('shuffle', True)

    fp_names = list(train_data.keys())
    first_fp = train_data[fp_names[0]]

    models = []

    # Choose splitter
    if groups is not None and len(np.unique(groups)) >= n_ensemble:
        splitter = StratifiedGroupKFold(n_splits=n_ensemble, shuffle=shuffle, random_state=42)
        split_iter = list(splitter.split(first_fp, labels, groups))
    else:
        splitter = StratifiedKFold(n_splits=n_ensemble, shuffle=shuffle, random_state=42)
        split_iter = list(splitter.split(first_fp, labels))

    for fold_idx, (train_idx, _) in enumerate(split_iter):
        fold_models = []
        y_train = labels[train_idx]

        for fp_name in fp_names:
            X_train = train_data[fp_name][train_idx]
            clf = MODELS[model_type](n_estimators, n_jobs, model_config)
            clf.fit(X_train, y_train)
            fold_models.append(deepcopy(clf))

        models.append(fold_models)

    return models


def predict_ensemble(models: List[List], test_data: Dict[str, np.ndarray]) -> np.ndarray:
    """Generate predictions from ensemble model (mean across all models)."""
    fp_names = list(test_data.keys())
    all_preds = []

    for fold_models in models:
        for clf, fp_name in zip(fold_models, fp_names):
            X_test = test_data[fp_name]
            preds = clf.predict_proba(X_test)[:, 1]
            all_preds.append(preds)

    all_preds = np.array(all_preds).T  # Shape: (n_samples, n_models)
    mean_preds = all_preds.mean(axis=1)

    return mean_preds


def train_fold_baseline(
    fold_idx: int,
    train_data: Dict[str, np.ndarray],
    train_labels: np.ndarray,
    train_groups: Optional[np.ndarray],
    test_data: Dict[str, np.ndarray],
    test_labels: np.ndarray,
    test_compound_ids: np.ndarray,
    config: dict,
    model_type: str
) -> Tuple[Dict[str, float], np.ndarray, np.ndarray, np.ndarray]:
    """Train and evaluate baseline model for a single fold."""
    # Validate alignment at function entry
    n_train = len(train_labels)
    n_test = len(test_labels)

    for fp_name, fp_array in train_data.items():
        if len(fp_array) != n_train:
            raise ValueError(f"train_data[{fp_name}] has {len(fp_array)} rows, expected {n_train}")

    for fp_name, fp_array in test_data.items():
        if len(fp_array) != n_test:
            raise ValueError(f"test_data[{fp_name}] has {len(fp_array)} rows, expected {n_test}")

    if len(test_compound_ids) != n_test:
        raise ValueError(f"test_compound_ids has {len(test_compound_ids)} rows, expected {n_test}")

    print(f"  Training {model_type} ensemble...")

    t0 = time()
    models = train_inner_ensemble(train_data, train_labels, train_groups, config, model_type)
    train_time = time() - t0

    # Predict
    mean_preds = predict_ensemble(models, test_data)

    # Calculate metrics (ranking-based only)
    metrics = calculate_ranking_metrics(test_labels, mean_preds)
    metrics['train_time'] = train_time

    return metrics, mean_preds, test_compound_ids, test_labels


def run_baseline_cv(config: dict):
    """Main CV orchestrator for baseline models."""
    print("=" * 70)
    print("Library K-Fold Cross-Validation for Baseline Models")
    print("=" * 70)

    seed = config.get('seed', 42)
    np.random.seed(seed)

    # Setup output directory
    output_dir = Path(config['output']['base_dir'])
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_dir = output_dir / f"{config['experiment_name']}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")

    # W&B config (initialization happens per fold)
    wandb_enabled = config['wandb'].get('enabled', False)
    wandb_dir = output_dir / 'wandb'
    if wandb_enabled:
        wandb_dir.mkdir(exist_ok=True)
        os.environ['WANDB_DIR'] = str(wandb_dir)

    # Get paths
    fold_path = config['cv']['fold_assignments_path']
    if not os.path.isabs(fold_path):
        fold_path = str(project_root / fold_path)

    parquet_path = config['data']['parquet_path']
    if not os.path.isabs(parquet_path):
        parquet_path = str(project_root / parquet_path)

    fp_types = config['fingerprints']['types']
    parquet_cols = list(set(FINGERPRINT_MAPPING.get(fp, fp) for fp in fp_types))

    fold_strategy = config['cv'].get('fold_strategy', 'library')  # 'library' or 'random'

    # Load or create fold assignments
    if os.path.exists(fold_path):
        # Load existing fold assignments
        print(f"\nLoading existing fold assignments from: {fold_path}")
        with open(fold_path, 'r') as f:
            fold_data = yaml.safe_load(f)

        assignments = fold_data['assignments']
        n_folds = fold_data['n_folds']
        # Extract cluster assignments if available (for reuse in inner splitting)
        cluster_assignments_map = fold_data.get('cluster_assignments', {})
        compound_ids_in_assignments = list(assignments.keys())

        print(f"  {n_folds} folds, {len(assignments)} samples")
        print_fold_summary(fold_data)

        # Load fingerprint data filtered to assignment compound IDs
        print(f"\nLoading fingerprints from: {parquet_path}")
        print(f"Fingerprint types: {fp_types}")

        compound_ids = np.array(compound_ids_in_assignments)
        fingerprints_raw, labels, library_ids = load_fingerprints_from_parquet(
            parquet_path, compound_ids, parquet_cols
        )

        # Convert cluster assignments to numpy array (matching compound_ids order)
        if cluster_assignments_map:
            precomputed_clusters = np.array([
                cluster_assignments_map.get(str(cid), -1)
                for cid in compound_ids
            ], dtype=np.int32)
            print(f"  Loaded pre-computed clusters: {len(np.unique(precomputed_clusters))} clusters")
        else:
            precomputed_clusters = None
    else:
        # Create new fold assignments - need to load all data first
        print(f"\nLoading all data from: {parquet_path}")
        print(f"Fingerprint types: {fp_types}")

        # Load all data without filtering
        columns = ['COMPOUND_ID', 'LABEL', 'LIBRARY_ID'] + parquet_cols
        df = pl.read_parquet(parquet_path, columns=columns)

        # Check for and remove duplicates
        n_before = len(df)
        df = df.unique(subset=['COMPOUND_ID'], keep='first')
        n_after = len(df)
        if n_before != n_after:
            print(f"  WARNING: Removed {n_before - n_after} duplicate COMPOUND_IDs from parquet")

        compound_ids = df['COMPOUND_ID'].to_numpy()
        labels = df['LABEL'].to_numpy().astype(np.int32)
        library_ids = df['LIBRARY_ID'].to_numpy()

        fingerprints_raw = {}
        for fp_col in parquet_cols:
            fp_array = np.vstack(df[fp_col].to_list())
            fingerprints_raw[fp_col] = fp_array
            print(f"  Loaded {fp_col}: shape={fp_array.shape}")

        print(f"  Labels: {len(labels)} samples, {labels.sum()} positives ({100*labels.mean():.2f}%)")

        # Create fold assignments
        print(f"\nCreating new fold assignments (strategy: {fold_strategy})...")
        n_folds = config['cv']['n_folds']

        if fold_strategy == 'library':
            splits, metadata = create_library_kfold_splits(
                library_ids=library_ids,
                labels=labels,
                n_folds=n_folds,
                min_library_size=config['library']['min_library_size'],
                seed=seed
            )
        elif fold_strategy == 'random':
            splits, metadata = create_random_stratified_kfold_splits(
                labels=labels,
                n_folds=n_folds,
                seed=seed
            )
        elif fold_strategy == 'cluster':
            # Cluster-based splitting using LeaderPicker
            cluster_cfg = config['cv'].get('cluster', {})
            if not cluster_cfg:
                raise ValueError("fold_strategy='cluster' requires cv.cluster config section")

            threshold = cluster_cfg.get('threshold', 0.7)
            fp_name = cluster_cfg.get('fingerprint')
            if not fp_name:
                raise ValueError("cv.cluster.fingerprint is required (e.g., 'FCFP6', 'ECFP4')")

            # Load fingerprints from parquet for clustering
            fingerprints_for_clustering = load_fingerprints_for_clustering(fp_name, parquet_path)

            splits, metadata = create_cluster_kfold_splits(
                fingerprints=fingerprints_for_clustering,
                labels=labels,
                n_folds=n_folds,
                threshold=threshold,
                fingerprint_name=fp_name,
                seed=seed,
                cache_dir=None
            )
        else:
            raise ValueError(f"Unknown fold_strategy: {fold_strategy}. Use 'library', 'random', or 'cluster'.")

        save_fold_assignments(splits, metadata, compound_ids, fold_path)
        print_fold_summary(metadata)

        # Build fold_data dict for consistency with loaded case
        with open(fold_path, 'r') as f:
            fold_data = yaml.safe_load(f)
        assignments = fold_data['assignments']

        # Extract cluster assignments if they were created
        cluster_assignments_map = fold_data.get('cluster_assignments', {})
        if cluster_assignments_map:
            precomputed_clusters = np.array([
                cluster_assignments_map.get(str(cid), -1)
                for cid in compound_ids
            ], dtype=np.int32)
            print(f"  Created clusters: {len(np.unique(precomputed_clusters))} clusters")
        else:
            precomputed_clusters = None

    # Process fingerprints
    if config['fingerprints'].get('binarize', False):
        fingerprints = {fp: (fingerprints_raw[FINGERPRINT_MAPPING.get(fp, fp)] > 0).astype(np.int32)
                       for fp in fp_types}
    else:
        fingerprints = {fp: fingerprints_raw[FINGERPRINT_MAPPING.get(fp, fp)] for fp in fp_types}

    # Build label lookup from parquet for validation during prediction saving
    df_labels = pl.read_parquet(parquet_path, columns=['COMPOUND_ID', 'LABEL'])
    label_lookup = dict(zip(
        [str(cid) for cid in df_labels['COMPOUND_ID'].to_list()],
        df_labels['LABEL'].to_list()
    ))

    # Get groups for inner ensemble splitting (based on config method)
    splitting_method = config.get('splitting', {}).get('method', 'library')
    print(f"\nInner ensemble splitting method: {splitting_method}")

    if splitting_method == 'library':
        train_groups_all = library_ids_to_groups(library_ids)
        print(f"  Using library-based groups: {len(np.unique(train_groups_all))} unique groups")
    elif splitting_method == 'cluster':
        if precomputed_clusters is not None:
            # Use pre-computed clusters from fold assignments
            train_groups_all = precomputed_clusters
            print(f"  Using pre-computed clusters from fold assignments: {len(np.unique(train_groups_all))} clusters")
        else:
            # Clusters will be computed per-fold from training data only (Wellnitz method)
            train_groups_all = None
            cluster_config = config.get('splitting', {}).get('cluster', {})
            print(f"  Cluster-based splitting enabled (computed per-fold)")
            print(f"    Threshold: {cluster_config.get('threshold', 0.7)}")
            print(f"    Fingerprint: {cluster_config.get('fingerprint', 'FCFP6')}")
    else:  # 'random' or other
        train_groups_all = None
        print(f"  Using random splitting (no groups)")

    # Recreate fold splits from assignments
    fold_indices = [[] for _ in range(n_folds)]
    for i, cid in enumerate(compound_ids):
        fold_idx = assignments.get(str(cid))
        if fold_idx is not None:
            fold_indices[fold_idx].append(i)

    fold_indices = [np.array(indices) for indices in fold_indices]

    # Validate fold assignments cover all compounds exactly once
    total_assigned = sum(len(indices) for indices in fold_indices)
    if total_assigned != len(compound_ids):
        raise ValueError(
            f"Fold assignments cover {total_assigned} compounds but dataset has {len(compound_ids)}. "
            f"Missing: {len(compound_ids) - total_assigned}"
        )

    assigned_indices = set()
    for fold_idx, indices in enumerate(fold_indices):
        for idx in indices:
            if idx in assigned_indices:
                raise ValueError(f"Index {idx} (compound {compound_ids[idx]}) assigned to multiple folds")
            assigned_indices.add(idx)

    # Create train/test splits
    splits = []
    for test_fold in range(n_folds):
        test_indices = fold_indices[test_fold]
        train_indices = np.concatenate([
            fold_indices[i] for i in range(n_folds) if i != test_fold
        ])
        splits.append((train_indices, test_indices))

    # Run CV for each model type
    model_types = config['model']['types']
    all_results = {}

    for model_type in model_types:
        if model_type not in MODELS:
            print(f"\nWarning: Model type '{model_type}' not available, skipping.")
            continue

        print(f"\n{'='*60}")
        print(f"Model: {model_type.upper()}")
        print(f"{'='*60}")

        fold_results = []

        for fold_idx, (train_indices, test_indices) in enumerate(splits):
            print(f"\nFold {fold_idx}: Train={len(train_indices)}, Test={len(test_indices)}")

            # Initialize W&B for this fold (separate run per fold, like transformer)
            if wandb_enabled:
                try:
                    import wandb
                    wandb.init(
                        project=config['wandb']['project'],
                        entity=config['wandb'].get('entity'),
                        name=f"{config['experiment_name']}_{model_type}_fold{fold_idx}",
                        group=config['wandb'].get('group', config['experiment_name']),
                        tags=list(config['wandb'].get('tags', [])) + [model_type, f'fold_{fold_idx}'],
                        config=config,
                        dir=str(wandb_dir),
                        reinit=True
                    )
                    print(f"  W&B run initialized: {config['experiment_name']}_{model_type}_fold{fold_idx}")
                except Exception as e:
                    print(f"  W&B initialization failed: {e}")

            # Create fold data
            train_data = {fp: fingerprints[fp][train_indices] for fp in fp_types}
            test_data = {fp: fingerprints[fp][test_indices] for fp in fp_types}
            train_labels = labels[train_indices]
            test_labels = labels[test_indices]

            # Get groups for this fold's inner ensemble splitting
            if splitting_method == 'cluster':
                if train_groups_all is not None:
                    # Use pre-computed clusters (subset for this fold's train indices)
                    train_groups = train_groups_all[train_indices]
                else:
                    # Compute clusters from THIS FOLD's training data only (Wellnitz method)
                    cluster_config = config.get('splitting', {}).get('cluster', {})
                    threshold = cluster_config.get('threshold', 0.7)
                    cluster_fp = cluster_config.get('fingerprint', 'FCFP6')

                    # Get the fingerprint to use for clustering
                    if cluster_fp in fingerprints:
                        fp_for_cluster = cluster_fp
                    elif FINGERPRINT_MAPPING.get(cluster_fp, cluster_fp) in fingerprints:
                        fp_for_cluster = FINGERPRINT_MAPPING.get(cluster_fp, cluster_fp)
                    else:
                        fp_for_cluster = fp_types[0]  # Default to first available
                        print(f"  Warning: Fingerprint '{cluster_fp}' not available, using '{fp_for_cluster}'")

                    # Get fingerprint array and binarize if needed (Tanimoto requires binary)
                    fp_array = fingerprints[fp_for_cluster][train_indices]
                    if fp_array.max() > 1:
                        fp_array = (fp_array > 0).astype(np.int32)

                    print(f"  Computing clusters for fold {fold_idx} (n={len(train_indices)}, fp={fp_for_cluster})...")
                    train_groups = generate_clusters_leaderpicker(fp_array, threshold)
            elif train_groups_all is not None:
                train_groups = train_groups_all[train_indices]
            else:
                train_groups = None

            # Train and evaluate
            test_compound_ids = compound_ids[test_indices]
            metrics, predictions, test_cids, test_lbls = train_fold_baseline(
                fold_idx, train_data, train_labels, train_groups,
                test_data, test_labels, test_compound_ids, config, model_type
            )

            print(f"  Results: ROC-AUC={metrics['roc_auc']:.4f}, PR-AUC={metrics['pr_auc']:.4f}, "
                  f"EF@100={metrics.get('enrich_at_100', 0):.2f}")

            fold_results.append(metrics)

            # Log to W&B (metrics go to this fold's run)
            if wandb_enabled:
                try:
                    import wandb
                    # Prefix with "test/" for W&B grouping (baselines have no validation set, only test)
                    wandb_metrics = {f"test/{k}": v for k, v in metrics.items()}
                    wandb_metrics['fold'] = fold_idx
                    wandb.log(wandb_metrics)
                except Exception:
                    pass

            # Save fold predictions
            if config['output'].get('save_predictions', True):
                pred_dir = output_dir / model_type / f"fold_{fold_idx}"
                pred_dir.mkdir(parents=True, exist_ok=True)

                pred_df = pd.DataFrame({
                    'compound_id': test_cids,
                    'label': test_lbls,
                    'prediction': predictions
                })

                # Verify saved labels match parquet ground truth
                for _, row in pred_df.iterrows():
                    cid = str(row['compound_id'])
                    saved_label = int(row['label'])
                    true_label = label_lookup.get(cid)
                    if true_label is not None and saved_label != int(true_label):
                        raise ValueError(
                            f"Label mismatch for compound {cid}: saved {saved_label}, parquet has {true_label}"
                        )

                pred_df.to_csv(pred_dir / 'predictions.csv', index=False)

            # Finish W&B run for this fold
            if wandb_enabled:
                try:
                    import wandb
                    wandb.finish()
                except Exception:
                    pass

        # Store results
        all_results[model_type] = fold_results

        # Aggregate for this model
        aggregated = aggregate_cv_metrics(fold_results)

        print(f"\n{model_type.upper()} Aggregate Results (mean +/- std, 95% CI):")
        for metric, stats in aggregated['metrics'].items():
            print(f"  {metric:<20} {stats['mean']:.4f} +/- {stats['std']:.4f}  "
                  f"[{stats['ci_95_lower']:.4f}, {stats['ci_95_upper']:.4f}]")

        # Note: Aggregate metrics are saved to files; per-fold W&B runs are separate

        # Save per-model results
        fold_df = pd.DataFrame(fold_results)
        fold_df.insert(0, 'fold', range(n_folds))
        fold_df.to_csv(output_dir / f'{model_type}_fold_results.csv', index=False)

    # Statistical comparisons between models
    if len(all_results) > 1:
        print("\n" + "=" * 60)
        print("STATISTICAL COMPARISONS (Wilcoxon signed-rank)")
        print("=" * 60)

        # Compare first model against others
        baseline_model = model_types[0]
        comparisons = compute_statistical_comparisons(all_results, baseline_model)

        for comparison_key, comparison_metrics in comparisons.items():
            print(f"\n{comparison_key}:")
            for metric, stats in comparison_metrics.items():
                sig = "*" if stats['significant'] else ""
                print(f"  {metric:<20} diff={stats['diff']:+.4f}, p={stats['p_value']:.4f}{sig}")

        # Save comparisons
        with open(output_dir / 'comparisons.yaml', 'w') as f:
            yaml.dump(comparisons, f, default_flow_style=False)

    # Save aggregate results for all models
    all_aggregated = {}
    for model_type, fold_results in all_results.items():
        all_aggregated[model_type] = aggregate_cv_metrics(fold_results)

    aggregate_data = {
        'experiment_name': config['experiment_name'],
        'timestamp': timestamp,
        'n_folds': n_folds,
        'seed': seed,
        'models': {
            model_type: agg['metrics']
            for model_type, agg in all_aggregated.items()
        },
        'fold_assignments_path': fold_path
    }

    with open(output_dir / 'aggregate.yaml', 'w') as f:
        yaml.dump(aggregate_data, f, default_flow_style=False)

    # Save config
    with open(output_dir / 'config_used.yaml', 'w') as f:
        yaml.dump(config, f, default_flow_style=False)

    print("\n" + "=" * 70)
    print("Baseline cross-validation complete!")
    print(f"Results saved to: {output_dir}")
    print("=" * 70)

    return all_aggregated


def main():
    parser = argparse.ArgumentParser(
        description="Library K-Fold CV for Baseline Models"
    )
    parser.add_argument(
        '--config', type=str, required=True,
        help='Path to configuration YAML file'
    )
    args = parser.parse_args()

    config = load_config(args.config)
    run_baseline_cv(config)


if __name__ == '__main__':
    main()
