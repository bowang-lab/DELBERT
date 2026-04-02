#!/usr/bin/env python3
"""
Pretrain data preparation script for DELBERT.

This script prepares data for unsupervised pretraining (MLM):
- Loads datasets from HuggingFace
- Builds vocabulary from all data
- Tokenizes molecules
- Creates train/val split (no test set for pretraining)

Output structure:
    processed_data/pretrain/{experiment_id}/
    ├── dataset/           # HuggingFace DatasetDict with train/val
    ├── tokenizer/         # HuggingFace tokenizer
    └── metadata.yaml      # Processing metadata and statistics

Usage:
    python scripts/prepare_pretrain_data.py --config-name pretrain/wdr91
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import sys

import json
import yaml
import hydra
from omegaconf import DictConfig, OmegaConf
from pathlib import Path
from datetime import datetime
import numpy as np
from datasets import load_dataset, concatenate_datasets, DatasetDict, Features, Sequence

from delbert.data.transforms import molecule_to_tokens, build_vocabulary, build_binary_vocabulary
from delbert.data.tokenizer import create_molecular_tokenizer


def process_molecules(examples, fingerprint_types, tokenizer, max_length, return_segment_ids=False, token_format="binary", nbits=2048):
    """Process a batch of molecules to token IDs."""
    batch_input_ids = []
    batch_num_tokens = []
    batch_segment_ids = [] if return_segment_ids else None

    batch_size = len(examples[list(examples.keys())[0]])

    for i in range(batch_size):
        example = {key: examples[key][i] for key in examples.keys()}

        if return_segment_ids:
            tokens, segment_ids = molecule_to_tokens(
                example, fingerprint_types, return_segment_ids=True,
                token_format=token_format, nbits=nbits
            )
        else:
            tokens = molecule_to_tokens(
                example, fingerprint_types,
                token_format=token_format, nbits=nbits
            )

        token_ids = tokenizer.convert_tokens_to_ids(tokens)

        if len(token_ids) > max_length:
            token_ids = token_ids[:max_length]
            if return_segment_ids:
                segment_ids = segment_ids[:max_length]

        batch_input_ids.append(token_ids)
        batch_num_tokens.append(len(token_ids))
        if return_segment_ids:
            batch_segment_ids.append(segment_ids)

    result = {
        'input_ids': batch_input_ids,
        'num_tokens': batch_num_tokens,
        'attention_mask': [[1] * len(ids) for ids in batch_input_ids]
    }

    if return_segment_ids:
        result['segment_ids'] = batch_segment_ids

    return result


def load_and_combine_datasets(dataset_path: str, dataset_names: list):
    """Load and combine multiple datasets from HuggingFace."""
    print("\n" + "=" * 40)
    print("Loading datasets...")
    print("=" * 40)

    loaded_datasets = []
    for dataset_name in dataset_names:
        print(f"Loading {dataset_name}...")
        dataset = load_dataset(
            path=dataset_path,
            name=dataset_name,
            split="train"
        )
        loaded_datasets.append(dataset)
        print(f"  Loaded {len(dataset):,} molecules")

    if len(loaded_datasets) == 1:
        return loaded_datasets[0]

    # Combine multiple datasets with schema alignment
    print("\nCombining datasets...")

    all_columns = set()
    for ds in loaded_datasets:
        all_columns.update(ds.column_names)
    all_columns = sorted(all_columns)

    shared_columns = set(loaded_datasets[0].column_names)
    for ds in loaded_datasets[1:]:
        shared_columns &= set(ds.column_names)

    print(f"\n  Column Analysis:")
    print(f"    Total unique columns: {len(all_columns)}")
    print(f"    Shared columns: {len(shared_columns)}")

    for i, ds in enumerate(loaded_datasets):
        unique_to_this = set(ds.column_names) - shared_columns
        if unique_to_this:
            print(f"    Only in {dataset_names[i]}: {sorted(unique_to_this)}")

    # Build unified schema
    unified_features = {}
    for col in all_columns:
        for ds in loaded_datasets:
            if col in ds.features:
                unified_features[col] = ds.features[col]
                break

    # Align datasets
    print(f"\n  Aligning datasets to {len(all_columns)} columns...")
    aligned_datasets = []

    for i, ds in enumerate(loaded_datasets):
        missing_cols = set(all_columns) - set(ds.column_names)

        if missing_cols:
            print(f"    {dataset_names[i]}: adding {len(missing_cols)} missing columns")

            null_values = {}
            for col in missing_cols:
                feature = unified_features[col]
                if isinstance(feature, Sequence):
                    null_values[col] = []
                else:
                    null_values[col] = None

            def add_null_columns(example, null_vals=null_values):
                for col, null_val in null_vals.items():
                    example[col] = null_val
                return example

            ds = ds.map(add_null_columns, desc=f"Adding columns to {dataset_names[i]}")

        ds = ds.select_columns(all_columns)
        aligned_datasets.append(ds)
        print(f"    {dataset_names[i]}: {len(ds):,} molecules, {len(all_columns)} columns")

    # Cast and concatenate
    print(f"  Casting and concatenating...")
    unified_features_obj = Features(unified_features)
    cast_datasets = []
    for i, ds in enumerate(aligned_datasets):
        try:
            ds_cast = ds.cast(unified_features_obj)
            cast_datasets.append(ds_cast)
        except Exception as e:
            print(f"    Warning: Could not cast {dataset_names[i]}: {e}")
            cast_datasets.append(ds)

    combined_dataset = concatenate_datasets(cast_datasets)
    print(f"\n  Combined dataset: {len(combined_dataset):,} molecules")

    return combined_dataset


def create_train_val_split(dataset, train_ratio: float, val_ratio: float, seed: int):
    """Create train/val split for pretraining."""
    print("\n" + "=" * 40)
    print("Creating train/val splits...")
    print("=" * 40)

    # Normalize ratios
    total_ratio = train_ratio + val_ratio
    train_ratio /= total_ratio
    val_ratio /= total_ratio

    n_samples = len(dataset)
    n_train = int(n_samples * train_ratio)

    rng = np.random.default_rng(seed)
    indices = rng.permutation(n_samples)

    train_indices = indices[:n_train].tolist()
    val_indices = indices[n_train:].tolist()

    train_dataset = dataset.select(train_indices)
    val_dataset = dataset.select(val_indices)

    print(f"  Train: {len(train_dataset):,} molecules ({train_ratio*100:.1f}%)")
    print(f"  Val: {len(val_dataset):,} molecules ({val_ratio*100:.1f}%)")

    return train_dataset, val_dataset


@hydra.main(version_base=None, config_path="../configs/data_prep", config_name=None)
def main(cfg: DictConfig):
    """Main pretrain data preparation function."""

    print("=" * 60)
    print("DELBERT Pretrain Data Preparation")
    print("=" * 60)

    # Extract configuration
    experiment_id = cfg.experiment_id
    output_base = cfg.output_base
    output_dir = Path(output_base) / experiment_id

    datasets_config = OmegaConf.to_container(cfg.datasets)
    if isinstance(datasets_config, str):
        datasets_config = [datasets_config]
    fingerprint_types = OmegaConf.to_container(cfg.fingerprint_types)
    seed = cfg.get('seed', 42)

    # Token format configuration
    token_format = cfg.get('token_format', 'binary')
    fingerprint_nbits = cfg.get('fingerprint_nbits', 2048)

    print(f"\nConfiguration:")
    print(f"  Experiment ID: {experiment_id}")
    print(f"  Output: {output_dir}")
    print(f"  Datasets: {datasets_config}")
    print(f"  Fingerprints: {fingerprint_types}")
    print(f"  Max length: {cfg.max_length}")
    print(f"  Token format: {token_format}")
    if token_format == 'binary':
        print(f"  Fingerprint bits: {fingerprint_nbits}")
    else:
        print(f"  Min token frequency: {cfg.min_token_frequency}")

    # Check output directory
    if output_dir.exists():
        print(f"\nWarning: Output directory already exists: {output_dir}")
        print("         Previous data will be overwritten!")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load datasets
    combined_dataset = load_and_combine_datasets(cfg.dataset_path, datasets_config)
    print(f"\nTotal molecules: {len(combined_dataset):,}")

    # Create train/val split
    train_ratio = cfg.splits.get('train', 0.95)
    val_ratio = cfg.splits.get('val', 0.05)
    train_dataset, val_dataset = create_train_val_split(
        combined_dataset, train_ratio, val_ratio, seed
    )

    # Build vocabulary based on token format
    print("\n" + "=" * 40)
    if token_format == 'binary':
        print("Building binary vocabulary (fixed, deterministic)...")
        print("=" * 40)
        vocab_data = build_binary_vocabulary(fingerprint_types, nbits=fingerprint_nbits)
    else:
        vocab_sample_fraction = cfg.get('vocab_sample_fraction', 0.25)
        print(f"Building count vocabulary (sampling {vocab_sample_fraction*100:.0f}% of data)...")
        print("=" * 40)
        vocab_data = build_vocabulary(
            combined_dataset,
            fingerprint_types,
            cfg.min_token_frequency,
            num_proc=cfg.get('num_workers', 4),
            sample_fraction=vocab_sample_fraction,
            seed=seed
        )

    print(f"Vocabulary size: {len(vocab_data['token_to_id']):,}")

    # Create and save tokenizer
    print("\nSaving tokenizer...")
    tokenizer_dir = output_dir / "tokenizer"
    tokenizer_dir.mkdir(exist_ok=True)

    tokenizer = create_molecular_tokenizer(
        vocabulary=vocab_data['token_to_id'],
        fingerprint_types=fingerprint_types,
        min_frequency=cfg.get('min_token_frequency') if token_format == 'count' else None,
        token_format=token_format,
        fingerprint_nbits=fingerprint_nbits
    )

    tokenizer.dataset_names = datasets_config
    if vocab_data['token_counts']:
        tokenizer.total_tokens = sum(vocab_data['token_counts'].values())
        tokenizer.unique_tokens = len(vocab_data['token_counts'])
    else:
        tokenizer.total_tokens = None
        tokenizer.unique_tokens = vocab_data['vocab_size']
    tokenizer.creation_date = datetime.now().isoformat()
    tokenizer.max_length = cfg.max_length

    tokenizer.save_pretrained(str(tokenizer_dir))

    # Save token counts (only for count mode)
    if vocab_data['token_counts']:
        token_counts_path = tokenizer_dir / "token_counts.json"
        top_tokens = {
            k: int(v) for k, v in sorted(
                vocab_data['token_counts'].items(),
                key=lambda x: x[1], reverse=True
            )[:10000]
        }
        with open(token_counts_path, 'w') as f:
            json.dump(top_tokens, f, indent=2)

    # Tokenize splits
    print("\n" + "=" * 40)
    print("Tokenizing molecules...")
    print("=" * 40)

    num_workers = cfg.get('num_workers', 4)
    return_segment_ids = cfg.get('return_segment_ids', False)

    if return_segment_ids:
        print("Segment IDs will be generated for segment-based attention")

    print("Tokenizing training data...")
    train_tokenized = train_dataset.map(
        lambda x: process_molecules(x, fingerprint_types, tokenizer, cfg.max_length, return_segment_ids, token_format, fingerprint_nbits),
        batched=True,
        batch_size=1000,
        desc="Tokenizing train",
        num_proc=num_workers
    )

    print("Tokenizing validation data...")
    val_tokenized = val_dataset.map(
        lambda x: process_molecules(x, fingerprint_types, tokenizer, cfg.max_length, return_segment_ids, token_format, fingerprint_nbits),
        batched=True,
        batch_size=1000,
        desc="Tokenizing val",
        num_proc=num_workers
    )

    # Save as DatasetDict
    print("\nSaving processed dataset...")
    dataset_dict = DatasetDict({
        'train': train_tokenized,
        'validation': val_tokenized
    })
    dataset_dir = output_dir / "dataset"
    dataset_dict.save_to_disk(str(dataset_dir))

    # Calculate statistics
    print("\n" + "=" * 40)
    print("Dataset Statistics")
    print("=" * 40)

    sample_size = min(10000, len(train_tokenized))
    sample_indices = np.random.choice(len(train_tokenized), sample_size, replace=False)
    sample_lengths = [len(train_tokenized[int(i)]['input_ids']) for i in sample_indices]

    stats = {
        'num_molecules': {
            'total': len(train_tokenized) + len(val_tokenized),
            'train': len(train_tokenized),
            'validation': len(val_tokenized)
        },
        'vocab_size': len(vocab_data['token_to_id']),
        'datasets': datasets_config,
        'fingerprint_types': fingerprint_types,
        'max_length': cfg.max_length,
        'token_format': token_format,
        'fingerprint_nbits': fingerprint_nbits if token_format == 'binary' else None,
        'min_token_frequency': cfg.get('min_token_frequency') if token_format == 'count' else None,
        'vocab_sample_fraction': cfg.get('vocab_sample_fraction') if token_format == 'count' else None,
        'splits': {
            'train': train_ratio,
            'val': val_ratio
        },
        'token_statistics': {
            'mean_length': float(np.mean(sample_lengths)),
            'std_length': float(np.std(sample_lengths)),
            'min_length': int(np.min(sample_lengths)),
            'max_length': int(np.max(sample_lengths)),
            'median_length': float(np.median(sample_lengths))
        },
        'processing_date': datetime.now().isoformat(),
        'seed': seed
    }

    print(f"  Total molecules: {stats['num_molecules']['total']:,}")
    print(f"    Train: {stats['num_molecules']['train']:,}")
    print(f"    Val: {stats['num_molecules']['validation']:,}")
    print(f"  Vocabulary size: {stats['vocab_size']:,}")
    print(f"  Mean sequence length: {stats['token_statistics']['mean_length']:.1f}")

    # Save metadata as YAML
    metadata_path = output_dir / "metadata.yaml"
    with open(metadata_path, 'w') as f:
        yaml.dump(stats, f, default_flow_style=False, sort_keys=False)

    # Save config snapshot
    config_path = output_dir / "config.yaml"
    with open(config_path, 'w') as f:
        OmegaConf.save(cfg, f)

    print("\n" + "=" * 60)
    print("Pretrain data preparation complete!")
    print("=" * 60)
    print(f"\nOutput saved to: {output_dir}")
    print(f"\nTo use for pretraining:")
    print(f"  python scripts/pretrain.py data.processed_dir={output_dir}")


if __name__ == "__main__":
    main()
