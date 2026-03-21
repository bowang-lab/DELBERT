#!/usr/bin/env python3
"""
Orchestrator for Library K-Fold Cross-Validation.

This script:
1. Prepares the dataset once
2. Creates/loads fold assignments
3. Runs each fold as a SEPARATE PROCESS (bulletproof DDP handling)
4. Aggregates results at the end

This approach is 100% reliable because each fold runs in a fresh Python process
with no shared DDP state, avoiding NCCL cleanup issues entirely.

Usage:
    python evals/library_cv/scripts/run_cv_orchestrator.py \
        --config evals/library_cv/configs/transformer_cv.yaml
"""

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import yaml
from omegaconf import OmegaConf
from datasets import load_dataset, load_from_disk

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent

from delbert.data.tokenizer import MolecularTokenizer
from delbert.data.cv_utils import (
    create_library_kfold_splits,
    create_random_stratified_kfold_splits,
    create_cluster_kfold_splits,
    generate_clusters_leaderpicker,
    save_fold_assignments,
    load_fold_assignments,
    aggregate_cv_metrics,
    print_fold_summary
)
from delbert.data.splits import extract_library_prefix


def load_fingerprints_for_clustering(
    fp_name: str,
    dataset_name: str,
    project_root: Path
) -> np.ndarray:
    """
    Load fingerprints from the parquet file for clustering.

    Args:
        fp_name: Fingerprint name (e.g., 'FCFP6', 'ECFP4')
        dataset_name: Dataset name (e.g., 'WDR91')
        project_root: Project root directory

    Returns:
        Binary fingerprint matrix of shape (N, n_bits)
    """
    import pandas as pd

    # Load from parquet file (has dense fingerprints)
    parquet_path = project_root / "data" / dataset_name / f"{dataset_name}.parquet"

    if not parquet_path.exists():
        raise FileNotFoundError(
            f"Parquet file not found: {parquet_path}. "
            f"Expected dense fingerprints in parquet file."
        )

    print(f"Loading {fp_name} fingerprints from parquet: {parquet_path}")

    # Read only the fingerprint column we need
    df = pd.read_parquet(parquet_path, columns=[fp_name])

    if fp_name not in df.columns:
        raise ValueError(
            f"Fingerprint '{fp_name}' not found in parquet file. "
            f"Available columns can be checked with: pd.read_parquet('{parquet_path}').columns"
        )

    # Stack into numpy array
    fingerprints = np.stack(df[fp_name].values)
    print(f"Loaded {len(fingerprints)} fingerprints, shape: {fingerprints.shape}")

    # Binarize if needed (convert counts to binary)
    if fingerprints.max() > 1:
        print(f"Binarizing {fp_name} fingerprints (max value={fingerprints.max()})")
        fingerprints = (fingerprints > 0).astype(np.int32)

    return fingerprints


def prepare_dataset(cfg: dict, cache_dir: Path, project_root: Path):
    """Prepare and cache the tokenized dataset. Runs once before any folds.

    Includes STRICT cache validation to ensure cached data matches current tokenizer config.
    """
    print("=" * 70)
    print("Preparing Dataset")
    print("=" * 70)

    # Load tokenizer
    tokenizer_dir = cfg['data']['tokenizer_dir']
    if not os.path.isabs(tokenizer_dir):
        tokenizer_dir = str(project_root / tokenizer_dir)

    print(f"\nLoading tokenizer from: {tokenizer_dir}")
    tokenizer = MolecularTokenizer.from_pretrained(tokenizer_dir)
    print(f"Vocabulary size: {tokenizer.vocab_size}")

    # STRICT: Validate tokenizer loaded correctly
    if tokenizer.vocab_size == 0:
        raise ValueError(
            f"CRITICAL: Tokenizer vocab_size is 0! This indicates the tokenizer failed to load. "
            f"Check that tokenizer_dir points to a directory containing vocab.json. "
            f"Current tokenizer_dir: {tokenizer_dir}. "
            f"If tokenizer files are in a subdirectory (e.g., 'tokenizer/'), update the path accordingly."
        )

    # Get fingerprint types and token format from tokenizer config
    tokenizer_config_path = Path(tokenizer_dir) / "tokenizer_config.json"
    if tokenizer_config_path.exists():
        with open(tokenizer_config_path, 'r') as f:
            tk_config = json.load(f)
        fingerprint_types = tk_config.get('fingerprint_types')
        token_format = tk_config.get('token_format')
        fingerprint_nbits = tk_config.get('fingerprint_nbits')

        if fingerprint_types is None:
            print("WARNING: 'fingerprint_types' not set in tokenizer config, using default")
            fingerprint_types = ['ECFP4', 'FCFP6', 'ATOMPAIR', 'TOPTOR']
        if token_format is None:
            print("NOTE: 'token_format' not set in tokenizer config (only needed for new tokenization)")
            token_format = 'count'  # Most common format, but won't be used if data already tokenized
        if fingerprint_nbits is None:
            fingerprint_nbits = 2048  # Default, only used for binary format
    else:
        print("WARNING: tokenizer_config.json not found, using defaults")
        fingerprint_types = ['ECFP4', 'FCFP6', 'ATOMPAIR', 'TOPTOR']
        token_format = 'count'
        fingerprint_nbits = 2048

    print(f"Token format: {token_format}")
    if token_format == 'binary' and fingerprint_nbits:
        print(f"Fingerprint bits: {fingerprint_nbits}")

    max_length = cfg['data'].get('max_length', 1024)

    # Create cache fingerprint for validation
    cache_fingerprint = {
        'tokenizer_vocab_size': tokenizer.vocab_size,
        'fingerprint_types': sorted(fingerprint_types),
        'token_format': token_format,
        'fingerprint_nbits': fingerprint_nbits,
        'max_length': max_length,
        'return_segment_ids': cfg['data'].get('return_segment_ids', False),
        'dataset_name': cfg['data']['dataset_name'],
        'dataset_path': cfg['data']['dataset_path'],
        'tokenizer_dir': str(tokenizer_dir)
    }
    fingerprint_str = json.dumps(cache_fingerprint, sort_keys=True)
    fingerprint_hash = hashlib.md5(fingerprint_str.encode()).hexdigest()[:8]

    # Check cache with validation
    processed_path = cache_dir / f"processed_{cfg['data']['dataset_name']}"
    metadata_path = processed_path / "cache_metadata.json"

    # Check cache mode from config
    cache_mode = cfg['data'].get('cache_mode', 'use_if_valid')
    if cache_mode == 'force_regenerate' and processed_path.exists():
        print(f"\nForce regenerate mode: deleting existing cache at {processed_path}")
        shutil.rmtree(processed_path)

    if processed_path.exists():
        print(f"\nFound cached dataset at: {processed_path}")

        # STRICT: Validate cache metadata matches current config
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                cached_meta = json.load(f)

            # Compare fingerprints
            mismatches = []
            for key, expected in cache_fingerprint.items():
                cached_val = cached_meta.get(key)
                if cached_val != expected:
                    mismatches.append(f"  {key}: cached={cached_val}, current={expected}")

            if mismatches:
                print("\nCRITICAL: Cache metadata does not match current config!")
                print("Mismatches:")
                for m in mismatches:
                    print(m)
                print("\nDeleting stale cache and rebuilding...")
                shutil.rmtree(processed_path)
            else:
                print("Cache validation passed - metadata matches current config")
                full_dataset = load_from_disk(str(processed_path))

                # STRICT: Verify LABEL column exists
                if 'LABEL' not in full_dataset.column_names:
                    raise ValueError(
                        f"CRITICAL: Cached dataset missing 'LABEL' column! "
                        f"Available columns: {full_dataset.column_names}. "
                        f"Delete cache and re-run: rm -rf {processed_path}"
                    )

                return str(processed_path), tokenizer_dir, full_dataset
        else:
            print("\nWARNING: Cache has no metadata.json - cannot verify it matches current config")
            print("Deleting unverified cache and rebuilding...")
            shutil.rmtree(processed_path)

    # Load and tokenize (cache miss or invalidated)
    print(f"\nLoading dataset: {cfg['data']['dataset_path']}/{cfg['data']['dataset_name']}")
    full_dataset = load_dataset(
        cfg['data']['dataset_path'],
        cfg['data']['dataset_name'],
        split='train'
    )
    print(f"Total molecules: {len(full_dataset)}")

    # STRICT: Verify LABEL column exists in source dataset
    if 'LABEL' not in full_dataset.column_names:
        raise ValueError(
            f"CRITICAL: Source dataset missing 'LABEL' column! "
            f"Available columns: {full_dataset.column_names}. "
            f"Cannot proceed without labels."
        )

    # Report label distribution
    labels = full_dataset['LABEL']
    pos_count = sum(1 for l in labels if l == 1)
    print(f"Label distribution: {pos_count:,} positives ({100*pos_count/len(labels):.2f}%), "
          f"{len(labels)-pos_count:,} negatives ({100*(len(labels)-pos_count)/len(labels):.2f}%)")

    if 'input_ids' not in full_dataset.column_names:
        print("\nTokenizing dataset...")
        from delbert.data.transforms import molecule_to_tokens

        # Check if segment_ids should be generated (for segment-based attention)
        return_segment_ids = cfg['data'].get('return_segment_ids', False)
        if return_segment_ids:
            print("Segment IDs will be generated for segment-based attention")

        def tokenize_molecules(examples):
            batch_input_ids = []
            batch_segment_ids = [] if return_segment_ids else None
            batch_size = len(examples[list(examples.keys())[0]])

            for i in range(batch_size):
                example = {key: examples[key][i] for key in examples.keys()}

                if return_segment_ids:
                    tokens, segment_ids = molecule_to_tokens(
                        example, fingerprint_types, return_segment_ids=True,
                        token_format=token_format, nbits=fingerprint_nbits
                    )
                else:
                    tokens = molecule_to_tokens(
                        example, fingerprint_types,
                        token_format=token_format, nbits=fingerprint_nbits
                    )

                token_ids = tokenizer.convert_tokens_to_ids(tokens)

                if len(token_ids) > max_length:
                    token_ids = token_ids[:max_length]
                    if return_segment_ids:
                        segment_ids = segment_ids[:max_length]

                batch_input_ids.append(token_ids)
                if return_segment_ids:
                    batch_segment_ids.append(segment_ids)

            result = {
                'input_ids': batch_input_ids,
                'attention_mask': [[1] * len(ids) for ids in batch_input_ids]
            }
            if return_segment_ids:
                result['segment_ids'] = batch_segment_ids
            return result

        full_dataset = full_dataset.map(
            tokenize_molecules,
            batched=True,
            batch_size=1000,
            desc="Tokenizing",
            num_proc=cfg['data'].get('num_workers', 4)
        )
        print("Tokenization complete.")

    # Save to disk WITH metadata for future validation
    print(f"\nSaving processed dataset to: {processed_path}")
    full_dataset.save_to_disk(str(processed_path))

    # Save cache metadata
    with open(metadata_path, 'w') as f:
        json.dump(cache_fingerprint, f, indent=2)
    print(f"Saved cache metadata (fingerprint: {fingerprint_hash})")

    return str(processed_path), tokenizer_dir, full_dataset


def validate_data_source_consistency(
    full_dataset,
    cfg: dict,
    project_root: Path
) -> None:
    """
    Validate HuggingFace dataset matches parquet file used by baselines.

    This ensures transformer and baseline evaluations use identical data.
    Compares compound IDs and labels between data sources.

    Args:
        full_dataset: HuggingFace dataset loaded for transformer
        cfg: Configuration dict
        project_root: Project root path

    Raises:
        ValueError: If labels mismatch between data sources
    """
    # Check if baseline parquet path is specified
    baseline_parquet = cfg.get('data', {}).get('baseline_parquet_path')
    if not baseline_parquet:
        # Try default WDR91 path for comparison
        default_parquet = project_root / f"data/{cfg['data']['dataset_name']}/{cfg['data']['dataset_name']}.parquet"
        if default_parquet.exists():
            baseline_parquet = str(default_parquet)
        else:
            print("\nData source validation: No baseline parquet path configured, skipping validation")
            return

    if not os.path.isabs(baseline_parquet):
        baseline_parquet = str(project_root / baseline_parquet)

    if not os.path.exists(baseline_parquet):
        print(f"\nData source validation: Parquet not found at {baseline_parquet}, skipping")
        return

    print("\n" + "=" * 60)
    print("Data Source Consistency Validation")
    print("=" * 60)

    try:
        import polars as pl

        # Load parquet compound IDs and labels
        df = pl.read_parquet(baseline_parquet, columns=['COMPOUND_ID', 'LABEL'])
        parquet_ids = set(str(x) for x in df['COMPOUND_ID'].to_list())
        parquet_labels = {str(k): int(v) for k, v in zip(df['COMPOUND_ID'].to_list(), df['LABEL'].to_list())}

        print(f"Parquet file: {baseline_parquet}")
        print(f"  Compounds: {len(parquet_ids)}")
        print(f"  Positives: {sum(1 for v in parquet_labels.values() if v == 1)}")

        # Get HuggingFace dataset IDs and labels
        hf_ids = set(str(x) for x in full_dataset['COMPOUND_ID'])
        hf_labels = {str(k): int(v) for k, v in zip(full_dataset['COMPOUND_ID'], full_dataset['LABEL'])}

        print(f"\nHuggingFace dataset: {cfg['data']['dataset_path']}/{cfg['data']['dataset_name']}")
        print(f"  Compounds: {len(hf_ids)}")
        print(f"  Positives: {sum(1 for v in hf_labels.values() if v == 1)}")

        # Check ID overlap
        only_in_parquet = parquet_ids - hf_ids
        only_in_hf = hf_ids - parquet_ids
        shared_ids = parquet_ids & hf_ids

        print(f"\nID Comparison:")
        print(f"  Shared: {len(shared_ids)}")
        if only_in_parquet:
            print(f"  Only in parquet: {len(only_in_parquet)}")
        if only_in_hf:
            print(f"  Only in HuggingFace: {len(only_in_hf)}")

        # Check label consistency for shared IDs
        mismatched_labels = []
        for cid in shared_ids:
            if parquet_labels.get(cid) != hf_labels.get(cid):
                mismatched_labels.append(cid)

        if mismatched_labels:
            raise ValueError(
                f"CRITICAL: {len(mismatched_labels)} compounds have different labels "
                f"between parquet and HuggingFace dataset! "
                f"First 5 mismatched: {mismatched_labels[:5]}. "
                f"This will cause inconsistent evaluation results between transformer and baselines!"
            )

        if len(shared_ids) == len(parquet_ids) == len(hf_ids):
            print("\n✓ Data source validation PASSED: All compound IDs and labels match")
        else:
            print(f"\n⚠ Data source validation WARNING: ID counts differ but no label mismatches")
            print(f"  Proceeding with {len(shared_ids)} shared compounds")

    except ImportError:
        print("WARNING: polars not available, skipping data source validation")
    except ValueError:
        raise  # Label mismatches are critical — never swallow these
    except Exception as e:
        raise RuntimeError(
            f"Data source validation failed unexpectedly: {e}. "
            f"Fix the underlying issue or remove the validation call."
        ) from e


def create_fold_indices(
    fold_idx: int,
    train_indices: np.ndarray,
    test_indices: np.ndarray,
    labels: np.ndarray,
    library_ids: np.ndarray,
    inner_val_strategy: str,
    inner_val_fraction: float,
    seed: int,
    major_libraries: list = None,
    cluster_ids: np.ndarray = None
) -> tuple:
    """
    Create train/val/test indices for a fold.

    Inner validation strategy options:
    - 'cluster': select clusters from lower Q3 (<=75th percentile size)
    - 'library': hold out 1 library
    - 'random': stratified random split

    Args:
        fold_idx: Fold index (for logging)
        train_indices: Indices for training (from outer CV split)
        test_indices: Indices for testing (held-out fold)
        labels: All labels
        library_ids: All library IDs
        inner_val_strategy: 'cluster', 'library', or 'random' (independent of fold_strategy)
        inner_val_fraction: Target fraction for validation
        seed: Random seed
        major_libraries: List of major library prefixes (>= min_library_size)
        cluster_ids: Array of cluster IDs (required if inner_val_strategy='cluster')
    """
    from sklearn.model_selection import train_test_split, StratifiedGroupKFold

    if inner_val_strategy == 'library':
        # Library-based inner split: hold out 1 library for validation
        train_library_ids = library_ids[train_indices]
        train_library_prefixes = np.array([extract_library_prefix(lid) for lid in train_library_ids])
        unique_libraries = np.unique(train_library_prefixes)

        # Filter to major libraries only
        if major_libraries is not None:
            major_set = set(major_libraries)
            unique_libraries = np.array([lib for lib in unique_libraries if lib in major_set])
            if len(unique_libraries) == 0:
                raise ValueError(f"Fold {fold_idx}: No major libraries available for inner validation")

        # Shuffle and select 1 library for validation
        rng = np.random.RandomState(seed)
        rng.shuffle(unique_libraries)
        val_library = unique_libraries[0]

        # Split indices
        val_mask = train_library_prefixes == val_library
        val_subset_indices = np.where(val_mask)[0]
        train_subset_indices = np.where(~val_mask)[0]

        val_size = len(val_subset_indices)
        val_frac = val_size / len(train_indices) * 100
        print(f"  Fold {fold_idx}: inner val = {val_library} ({val_size:,} samples, {val_frac:.1f}%)")

    elif inner_val_strategy == 'cluster':
        # Cluster-based inner split: select from small clusters (≤ Q3)
        if cluster_ids is None:
            raise ValueError("inner_val_strategy='cluster' requires cluster_ids")

        train_labels = labels[train_indices]
        train_cluster_ids = cluster_ids[train_indices]

        # Calculate cluster sizes and Q3 threshold
        unique_clusters, cluster_counts = np.unique(train_cluster_ids, return_counts=True)
        q3_size = int(np.percentile(cluster_counts, 75))

        # Filter to small clusters (≤ Q3)
        small_clusters = set(c for c, size in zip(unique_clusters, cluster_counts) if size <= q3_size)
        small_cluster_mask = np.isin(train_cluster_ids, list(small_clusters))
        small_cluster_indices = np.where(small_cluster_mask)[0]

        if len(small_cluster_indices) == 0:
            raise ValueError(f"Fold {fold_idx}: No small clusters found (Q3={q3_size})")

        # Calculate n_splits to achieve target fraction of TOTAL training data
        # (not just fraction of small cluster pool)
        target_val_samples = inner_val_fraction * len(train_indices)
        fraction_of_small_pool = min(0.5, target_val_samples / len(small_cluster_indices))
        n_splits = max(2, round(1.0 / fraction_of_small_pool))

        sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)

        for _, small_val_sub in sgkf.split(
            np.zeros(len(small_cluster_indices)),
            train_labels[small_cluster_indices],
            groups=train_cluster_ids[small_cluster_indices]
        ):
            break

        val_subset_indices = small_cluster_indices[small_val_sub]
        train_subset_indices = np.setdiff1d(np.arange(len(train_indices)), val_subset_indices)

        n_val_clusters = len(np.unique(train_cluster_ids[val_subset_indices]))
        val_frac = len(val_subset_indices) / len(train_indices) * 100
        print(f"  Fold {fold_idx}: inner val = {n_val_clusters} clusters (≤{q3_size}/cluster), "
              f"{len(val_subset_indices):,} samples ({val_frac:.1f}%)")

    else:  # 'random' - stratified random split
        train_labels = labels[train_indices]
        train_subset_indices, val_subset_indices = train_test_split(
            np.arange(len(train_indices)),
            test_size=inner_val_fraction,
            stratify=train_labels,
            random_state=seed
        )
        val_frac = len(val_subset_indices) / len(train_indices) * 100
        print(f"  Fold {fold_idx}: inner val = random stratified, {len(val_subset_indices):,} samples ({val_frac:.1f}%)")

    final_train_indices = train_indices[train_subset_indices]
    final_val_indices = train_indices[val_subset_indices]

    return final_train_indices, final_val_indices, test_indices


def run_fold_subprocess(
    fold_idx: int,
    train_indices: np.ndarray,
    val_indices: np.ndarray,
    test_indices: np.ndarray,
    dataset_path: str,
    tokenizer_path: str,
    config_path: str,
    output_dir: Path,
    script_path: Path
) -> Dict[str, float]:
    """Run a single fold as a subprocess."""
    # Create fold directory and save indices
    fold_dir = output_dir / f"fold_{fold_idx}"
    fold_dir.mkdir(parents=True, exist_ok=True)

    indices_file = fold_dir / "indices.json"
    with open(indices_file, 'w') as f:
        json.dump({
            'train_indices': train_indices.tolist(),
            'val_indices': val_indices.tolist(),
            'test_indices': test_indices.tolist()
        }, f)

    # Copy config to fold directory for robustness
    # (prevents issues if parent config.yaml gets deleted by external processes)
    fold_config_path = fold_dir / "config.yaml"
    if not Path(config_path).exists():
        raise FileNotFoundError(
            f"Parent config file missing: {config_path}\n"
            f"This should not happen - the config should have been created at the start of the run.\n"
            f"Check for external cleanup scripts or filesystem issues."
        )
    shutil.copy2(config_path, fold_config_path)
    print(f"  Config copied to: {fold_config_path}")

    # Build command - use fold-local config for robustness
    cmd = [
        sys.executable,  # Use same Python interpreter
        str(script_path),
        '--fold_idx', str(fold_idx),
        '--output_dir', str(output_dir),
        '--config', str(fold_config_path),  # Use fold-local config copy
        '--indices_file', str(indices_file),
        '--dataset_path', dataset_path,
        '--tokenizer_path', tokenizer_path
    ]

    print(f"\n{'='*60}")
    print(f"Starting fold {fold_idx} as subprocess...")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*60}\n")

    # Run subprocess
    result = subprocess.run(
        cmd,
        cwd=str(project_root),
        env={**os.environ, 'HYDRA_FULL_ERROR': '1'}
    )

    if result.returncode != 0:
        raise RuntimeError(f"Fold {fold_idx} failed with return code {result.returncode}")

    # Load metrics from fold output
    metrics_file = output_dir / f"fold_{fold_idx}" / "metrics.json"
    with open(metrics_file, 'r') as f:
        metrics = json.load(f)

    return metrics


def main():
    parser = argparse.ArgumentParser(description="CV Orchestrator")
    parser.add_argument('--config', type=str, required=True, help='Config YAML path')
    parser.add_argument('--folds', type=str, default=None,
                        help='Comma-separated fold indices to run (e.g., "0" or "0,1,2"). Default: all folds')
    args = parser.parse_args()

    # Parse folds to run
    folds_to_run = None
    if args.folds:
        folds_to_run = [int(f.strip()) for f in args.folds.split(',')]

    print("=" * 70)
    print("Library K-Fold Cross-Validation Orchestrator")
    print("=" * 70)
    print("Each fold runs as a SEPARATE PROCESS for reliable DDP handling")
    print("=" * 70)

    # Load config
    cfg = OmegaConf.load(args.config)
    cfg = OmegaConf.to_container(cfg, resolve=True)
    print(f"\nExperiment: {cfg['experiment_name']}")

    # Setup directories
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    base_dir = cfg['output'].get('base_dir', 'evals/library_cv/results/transformer')
    if not os.path.isabs(base_dir):
        base_dir = str(project_root / base_dir)
    output_dir = Path(base_dir) / f"{cfg['experiment_name']}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")

    # Save config to output dir so it's frozen for this run
    # (prevents issues if user modifies the original config while job is running)
    frozen_config_path = output_dir / "config.yaml"
    with open(frozen_config_path, 'w') as f:
        yaml.dump(cfg, f, default_flow_style=False)

    # Cache dir includes tokenizer parent name to avoid conflicts between different tokenizers
    # e.g., "processed_data/supervised/wdr91_library_ood/tokenizer" -> "wdr91_library_ood"
    tokenizer_parent = Path(cfg['data']['tokenizer_dir']).parent.name
    cache_dir = project_root / "evals/library_cv/cache" / tokenizer_parent
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Prepare dataset
    dataset_path, tokenizer_path, full_dataset = prepare_dataset(cfg, cache_dir, project_root)

    # Validate data source consistency with baseline parquet (if available)
    validate_data_source_consistency(full_dataset, cfg, project_root)

    # STRICT: Validate segment embedding consistency between pretrained model and data config
    # Do this early (before any folds) to fail fast on config mismatch
    pretrained_checkpoint = cfg.get('pretrained_checkpoint')
    if pretrained_checkpoint:
        if not os.path.isabs(pretrained_checkpoint):
            pretrained_checkpoint = str(project_root / pretrained_checkpoint)

        if os.path.exists(pretrained_checkpoint):
            import torch
            ckpt = torch.load(pretrained_checkpoint, map_location='cpu')
            model_config = ckpt.get('model_config', {})

            model_uses_segment_embeds = model_config.get('use_segment_embeddings', False)
            model_uses_segment_attention = model_config.get('use_segment_attention', False)
            data_returns_segment_ids = cfg['data'].get('return_segment_ids', False)

            print("\n" + "=" * 60)
            print("Segment Embedding Consistency Check")
            print("=" * 60)
            print(f"  Pretrained model use_segment_embeddings: {model_uses_segment_embeds}")
            print(f"  Pretrained model use_segment_attention: {model_uses_segment_attention}")
            print(f"  Data config return_segment_ids: {data_returns_segment_ids}")

            if model_uses_segment_embeds or model_uses_segment_attention:
                if not data_returns_segment_ids:
                    raise ValueError(
                        f"CRITICAL: Pretrained model requires segment_ids but data config has "
                        f"'return_segment_ids: {data_returns_segment_ids}'!\n"
                        f"  Model use_segment_embeddings: {model_uses_segment_embeds}\n"
                        f"  Model use_segment_attention: {model_uses_segment_attention}\n"
                        f"Fix: Set 'return_segment_ids: true' in data config to match pretrained model."
                    )
                print("  ✓ Segment config validation PASSED")
            elif data_returns_segment_ids:
                print("  ⚠ WARNING: Data returns segment_ids but model doesn't use them (wasted computation)")
            else:
                print("  ✓ No segment features configured")

            del ckpt  # Free memory

    # Deduplicate compound IDs (baselines deduplicate via polars .unique(),
    # so the transformer must also deduplicate to ensure identical evaluation populations)
    n_total = len(full_dataset)
    compound_ids_list = full_dataset['COMPOUND_ID']
    seen = set()
    unique_indices = []
    for i, cid in enumerate(compound_ids_list):
        if cid not in seen:
            seen.add(cid)
            unique_indices.append(i)
    n_unique = len(unique_indices)
    if n_unique < n_total:
        n_dupes = n_total - n_unique
        print(f"WARNING: Removed {n_dupes} duplicate COMPOUND_IDs from dataset "
              f"({n_total} total -> {n_unique} unique)")
        full_dataset = full_dataset.select(unique_indices)

    # Get data for fold creation
    library_ids = np.array(full_dataset[cfg['library']['column']])
    labels = np.array(full_dataset['LABEL'])

    # Load or create fold assignments
    fold_path = cfg['cv']['fold_assignments_path']
    if not os.path.isabs(fold_path):
        fold_path = str(project_root / fold_path)

    fold_strategy = cfg['cv'].get('fold_strategy', 'library')  # 'library' or 'random'

    if os.path.exists(fold_path):
        print(f"\nLoading existing fold assignments from: {fold_path}")
        compound_ids = np.array(full_dataset['COMPOUND_ID'])
        splits, metadata = load_fold_assignments(fold_path, compound_ids)
    else:
        print(f"\nCreating new fold assignments (strategy: {fold_strategy})...")
        compound_ids = np.array(full_dataset['COMPOUND_ID'])

        if fold_strategy == 'library':
            splits, metadata = create_library_kfold_splits(
                library_ids=library_ids,
                labels=labels,
                n_folds=cfg['cv']['n_folds'],
                min_library_size=cfg['library']['min_library_size'],
                seed=cfg.get('seed', 42)
            )
        elif fold_strategy == 'random':
            splits, metadata = create_random_stratified_kfold_splits(
                labels=labels,
                n_folds=cfg['cv']['n_folds'],
                seed=cfg.get('seed', 42)
            )
        elif fold_strategy == 'cluster':
            # Cluster-based splitting using LeaderPicker
            cluster_cfg = cfg['cv'].get('cluster', {})
            if not cluster_cfg:
                raise ValueError("fold_strategy='cluster' requires cv.cluster config section")

            threshold = cluster_cfg.get('threshold', 0.7)
            fp_name = cluster_cfg.get('fingerprint')
            if not fp_name:
                raise ValueError("cv.cluster.fingerprint is required (e.g., 'FCFP6', 'ECFP4')")

            cache = cluster_cfg.get('cache', True)

            # Load fingerprints from parquet file
            fingerprints = load_fingerprints_for_clustering(
                fp_name=fp_name,
                dataset_name=cfg['data']['dataset_name'],
                project_root=project_root
            )

            cluster_cache_dir = str(cache_dir) if cache else None

            splits, metadata = create_cluster_kfold_splits(
                fingerprints=fingerprints,
                labels=labels,
                n_folds=cfg['cv']['n_folds'],
                threshold=threshold,
                fingerprint_name=fp_name,
                seed=cfg.get('seed', 42),
                cache_dir=cluster_cache_dir
            )
        else:
            raise ValueError(f"Unknown fold_strategy: {fold_strategy}. Use 'library', 'random', or 'cluster'.")

        save_fold_assignments(splits, metadata, compound_ids, fold_path)

    print_fold_summary(metadata)

    # Run each fold as a separate process
    n_folds = cfg['cv']['n_folds']
    inner_val_fraction = cfg['cv'].get('inner_val_fraction', 0.1)
    inner_val_strategy = cfg['cv'].get('inner_val_strategy', fold_strategy)  # Default to fold_strategy for backward compat
    seed = cfg.get('seed', 42)

    print(f"\nInner validation: {inner_val_strategy}-based, target {inner_val_fraction*100:.0f}%")

    # Get major libraries from metadata (for library-based inner val)
    major_libraries = metadata.get('major_libraries', None)
    if inner_val_strategy == 'library':
        if major_libraries:
            print(f"  Major libraries available: {len(major_libraries)}")
        else:
            # Extract major_libraries from library_ids if not in metadata
            # (needed when inner_val_strategy='library' but fold_strategy != 'library')
            library_prefixes = np.array([extract_library_prefix(lid) for lid in library_ids])
            min_library_size = cfg.get('library', {}).get('min_library_size', 1000)
            unique_libs, lib_counts = np.unique(library_prefixes, return_counts=True)
            major_libraries = [lib for lib, cnt in zip(unique_libs, lib_counts) if cnt >= min_library_size and lib != 'OTHER']
            print(f"  Extracted major libraries for inner val: {len(major_libraries)}")

    # Get cluster IDs for cluster-based inner validation
    cluster_ids = None
    if inner_val_strategy == 'cluster' or fold_strategy == 'cluster':
        if 'cluster_ids' in metadata:
            cluster_ids = metadata['cluster_ids']
            print(f"  Using cluster IDs from fold assignments: {metadata.get('n_clusters', 'N/A')} clusters")
        elif 'cluster_assignments' in metadata:
            cluster_assignments = metadata['cluster_assignments']
            cluster_ids = np.array([
                cluster_assignments.get(str(cid), -1) for cid in compound_ids
            ], dtype=np.int32)
            n_clusters = len(np.unique(cluster_ids[cluster_ids >= 0]))
            print(f"  Reconstructed cluster IDs from fold assignments: {n_clusters} clusters")
        else:
            raise ValueError("fold_strategy='cluster' requires cluster_assignments in fold file")

    script_path = Path(__file__).parent / "train_transformer_fold.py"
    fold_results = []

    for fold_idx, (train_indices, test_indices) in enumerate(splits):
        # Skip folds not in the list (if specified)
        if folds_to_run is not None and fold_idx not in folds_to_run:
            continue

        # Create train/val/test split
        train_final, val_indices, test_final = create_fold_indices(
            fold_idx=fold_idx,
            train_indices=train_indices,
            test_indices=test_indices,
            labels=labels,
            library_ids=library_ids,
            inner_val_strategy=inner_val_strategy,
            inner_val_fraction=inner_val_fraction,
            seed=seed + fold_idx,
            major_libraries=major_libraries,
            cluster_ids=cluster_ids
        )

        # Run fold as subprocess (use frozen config so edits to original don't affect running job)
        metrics = run_fold_subprocess(
            fold_idx=fold_idx,
            train_indices=train_final,
            val_indices=val_indices,
            test_indices=test_final,
            dataset_path=dataset_path,
            tokenizer_path=tokenizer_path,
            config_path=str(frozen_config_path),
            output_dir=output_dir,
            script_path=script_path
        )

        fold_results.append(metrics)
        print(f"\nFold {fold_idx} complete: ROC-AUC={metrics['roc_auc']:.4f}, PR-AUC={metrics['pr_auc']:.4f}")

        # Force cleanup between folds to prevent CUDA memory accumulation
        import gc
        import time
        gc.collect()
        time.sleep(5)  # Give GPU time to release resources

    # Aggregate results
    print("\n" + "=" * 70)
    print("AGGREGATE RESULTS")
    print("=" * 70)

    aggregated = aggregate_cv_metrics(fold_results)
    metrics_dict = aggregated['metrics']

    print(f"\nMetrics (mean +/- std, 95% CI):")
    for metric_name, stats in metrics_dict.items():
        mean = stats['mean']
        std = stats['std']
        ci_low = stats['ci_95_lower']
        ci_high = stats['ci_95_upper']
        print(f"  {metric_name}: {mean:.4f} +/- {std:.4f} (95% CI: [{ci_low:.4f}, {ci_high:.4f}])")

    # Save results
    results_df = pd.DataFrame(fold_results)
    results_df.to_csv(output_dir / 'fold_results.csv', index=False)

    aggregate_output = {
        'experiment': cfg['experiment_name'],
        'n_folds': n_folds,
        'metrics': {}
    }
    for metric_name, stats in metrics_dict.items():
        aggregate_output['metrics'][metric_name] = {
            'mean': float(stats['mean']),
            'std': float(stats['std']),
            'ci_95': [float(stats['ci_95_lower']), float(stats['ci_95_upper'])]
        }
    with open(output_dir / 'aggregate.yaml', 'w') as f:
        yaml.dump(aggregate_output, f, default_flow_style=False)

    # Save config used
    with open(output_dir / 'config_used.yaml', 'w') as f:
        yaml.dump(cfg, f, default_flow_style=False)

    print(f"\nResults saved to: {output_dir}")
    print("\n" + "=" * 70)
    print("Cross-validation complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
