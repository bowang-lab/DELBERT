#!/usr/bin/env python3
"""
Supervised data preparation script for DELBERT.

This script prepares labeled data for supervised classification:
- Loads labeled dataset from HuggingFace
- Reuses tokenizer from pretraining (must exist)
- Creates TEST SPLIT FIRST (locked, never touched)
- Then splits remaining into train/val
- Supports random and library-based OOD splits

Output structure:
    processed_data/supervised/{output_folder}/
    ├── train/             # HuggingFace Dataset (separate directory)
    ├── val/               # HuggingFace Dataset (separate directory)
    ├── test/              # HuggingFace Dataset (LOCKED - separate directory)
    ├── tokenizer/         # Copied from pretrain for self-containment
    └── metadata.yaml      # Processing metadata with label distributions

IMPORTANT: Test set is created FIRST before any train/val split to ensure
complete isolation. The test set should never be modified after creation.

Usage:
    python scripts/prepare_supervised_data.py --config-name supervised/wdr91_random
    python scripts/prepare_supervised_data.py --config-name supervised/wdr91_library_ood
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'


import shutil
import yaml
import hydra
from omegaconf import DictConfig, OmegaConf
from pathlib import Path
from datetime import datetime
import numpy as np
from datasets import load_dataset

from delbert.data.tokenizer import MolecularTokenizer
from delbert.data.transforms import molecule_to_tokens
from delbert.data.splits import (
    get_split_functions,
    list_split_strategies,
    get_label_distribution,
    compute_split_statistics
)


def process_molecules(examples, fingerprint_types, tokenizer, max_length, return_segment_ids=False, token_format="binary", nbits=2048):
    """Process a batch of molecules to token IDs, preserving labels."""
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


@hydra.main(version_base=None, config_path="../configs/data_prep", config_name=None)
def main(cfg: DictConfig):
    """Main supervised data preparation function."""

    print("=" * 60)
    print("DELBERT Supervised Data Preparation")
    print("=" * 60)

    # Extract configuration
    output_folder = cfg.output_folder
    split_strategy = cfg.split_strategy
    seed = cfg.get('seed', 42)

    # Output directory
    output_dir = Path("processed_data/supervised") / output_folder

    print(f"\nConfiguration:")
    print(f"  Output: {output_dir}")
    print(f"  Split strategy: {split_strategy}")

    # Validate pretrain tokenizer exists
    tokenizer_path = Path(cfg.pretrain_tokenizer_path)
    if not tokenizer_path.exists():
        raise ValueError(
            f"\nPretrain tokenizer not found at: {tokenizer_path}\n"
            f"Please run prepare_pretrain_data.py first to create the tokenizer.\n"
            f"Example: python scripts/prepare_pretrain_data.py --config-name data_prep/pretrain/wdr91"
        )

    print(f"\nLoading tokenizer from: {tokenizer_path}")
    tokenizer = MolecularTokenizer.from_pretrained(str(tokenizer_path))
    print(f"  Vocabulary size: {tokenizer.vocab_size:,}")

    # STRICT: Validate tokenizer loaded correctly
    if tokenizer.vocab_size == 0:
        raise ValueError(
            f"CRITICAL: Tokenizer vocab_size is 0! This indicates the tokenizer failed to load. "
            f"Check that tokenizer_path points to a directory containing vocab.json. "
            f"Current tokenizer_path: {tokenizer_path}."
        )

    # Get fingerprint types and token format from tokenizer config
    tokenizer_config_path = tokenizer_path / "tokenizer_config.json"
    if tokenizer_config_path.exists():
        import json
        with open(tokenizer_config_path, 'r') as f:
            tokenizer_config = json.load(f)
        fingerprint_types = tokenizer_config.get('fingerprint_types', ['ECFP4', 'ATOMPAIR', 'TOPTOR'])
        token_format = tokenizer_config.get('token_format', 'binary')
        fingerprint_nbits = tokenizer_config.get('fingerprint_nbits', 2048)
    else:
        fingerprint_types = ['ECFP4', 'FCFP6', 'ATOMPAIR', 'TOPTOR']
        token_format = 'binary'
        fingerprint_nbits = 2048
    print(f"  Fingerprint types: {fingerprint_types}")
    print(f"  Token format: {token_format}")
    if token_format == 'binary':
        print(f"  Fingerprint bits: {fingerprint_nbits}")

    # Check output directory
    if output_dir.exists():
        print(f"\nWarning: Output directory already exists: {output_dir}")
        print("         Previous data will be overwritten!")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load labeled dataset
    print("\n" + "=" * 40)
    print("Loading labeled dataset...")
    print("=" * 40)

    dataset = load_dataset(
        path=cfg.dataset_path,
        name=cfg.dataset_name,
        split="train"
    )
    print(f"Loaded {len(dataset):,} labeled molecules")

    # Show label distribution
    label_column = cfg.label_column
    initial_dist = get_label_distribution(dataset, label_column)
    print(f"Label distribution: {initial_dist}")

    # Get split functions from registry
    print(f"\nAvailable split strategies: {list(list_split_strategies().keys())}")
    test_split_fn, train_val_split_fn = get_split_functions(split_strategy)

    # Common kwargs for split functions
    split_kwargs = {
        "seed": seed,
        "label_column": cfg.label_column,
        "library_column": cfg.get("library_id_column", "LIBRARY_ID"),
        "min_library_size": cfg.get("min_library_size", 1000),
    }

    # Create TEST SPLIT FIRST
    print("\n" + "=" * 40)
    print("Creating TEST split FIRST (this will be locked)...")
    print("=" * 40)

    test_fraction = cfg.splits.test
    test_dataset, remaining_dataset = test_split_fn(
        dataset,
        test_fraction=test_fraction,
        **split_kwargs
    )

    print(f"\nTest set: {len(test_dataset):,} molecules (LOCKED)")
    print(f"Remaining: {len(remaining_dataset):,} molecules")

    # Create train/val split from remaining
    print("\n" + "=" * 40)
    print("Creating train/val splits from remaining data...")
    print("=" * 40)

    train_fraction = cfg.splits.train
    train_dataset, val_dataset = train_val_split_fn(
        remaining_dataset,
        train_fraction=train_fraction,
        **split_kwargs
    )

    print(f"\nFinal splits:")
    print(f"  Train: {len(train_dataset):,} molecules")
    print(f"  Val: {len(val_dataset):,} molecules")
    print(f"  Test: {len(test_dataset):,} molecules (LOCKED)")

    # Compute split statistics
    split_stats = compute_split_statistics(
        train_dataset, val_dataset, test_dataset, label_column
    )

    print(f"\nLabel distributions:")
    for split_name in ['train', 'val', 'test']:
        dist = split_stats['label_distribution'][split_name]
        ratios = split_stats['label_distribution'][f'{split_name}_ratios']
        print(f"  {split_name}: {dist} (ratios: {ratios})")

    # Tokenize all splits
    print("\n" + "=" * 40)
    print("Tokenizing molecules...")
    print("=" * 40)

    num_workers = cfg.get('num_workers', 4)
    max_length = cfg.max_length
    return_segment_ids = cfg.get('return_segment_ids', False)

    if return_segment_ids:
        print("Segment IDs will be generated for segment-based attention")

    print("Tokenizing training data...")
    train_tokenized = train_dataset.map(
        lambda x: process_molecules(x, fingerprint_types, tokenizer, max_length, return_segment_ids, token_format, fingerprint_nbits),
        batched=True,
        batch_size=1000,
        desc="Tokenizing train",
        num_proc=num_workers
    )

    print("Tokenizing validation data...")
    val_tokenized = val_dataset.map(
        lambda x: process_molecules(x, fingerprint_types, tokenizer, max_length, return_segment_ids, token_format, fingerprint_nbits),
        batched=True,
        batch_size=1000,
        desc="Tokenizing val",
        num_proc=num_workers
    )

    print("Tokenizing test data...")
    test_tokenized = test_dataset.map(
        lambda x: process_molecules(x, fingerprint_types, tokenizer, max_length, return_segment_ids, token_format, fingerprint_nbits),
        batched=True,
        batch_size=1000,
        desc="Tokenizing test",
        num_proc=num_workers
    )

    # Save as SEPARATE DIRECTORIES (physical isolation)
    print("\n" + "=" * 40)
    print("Saving processed datasets (separate directories)...")
    print("=" * 40)

    train_dir = output_dir / "train"
    val_dir = output_dir / "val"
    test_dir = output_dir / "test"

    print(f"  Saving train to: {train_dir}")
    train_tokenized.save_to_disk(str(train_dir))

    print(f"  Saving val to: {val_dir}")
    val_tokenized.save_to_disk(str(val_dir))

    print(f"  Saving test to: {test_dir} (LOCKED)")
    test_tokenized.save_to_disk(str(test_dir))

    # Copy tokenizer for self-containment
    print("\nCopying tokenizer for self-containment...")
    tokenizer_dest = output_dir / "tokenizer"
    if tokenizer_dest.exists():
        shutil.rmtree(tokenizer_dest)
    shutil.copytree(tokenizer_path, tokenizer_dest)

    # Calculate sequence length statistics
    sample_size = min(5000, len(train_tokenized))
    sample_indices = np.random.choice(len(train_tokenized), sample_size, replace=False)
    sample_lengths = [len(train_tokenized[int(i)]['input_ids']) for i in sample_indices]

    # Save metadata as YAML
    metadata = {
        'output_folder': output_folder,
        'split_strategy': split_strategy,
        'created': datetime.now().isoformat(),
        'pretrain_tokenizer_source': str(tokenizer_path.absolute()),
        'dataset': cfg.dataset_name,
        'vocab_size': tokenizer.vocab_size,
        'max_length': max_length,
        'fingerprint_types': fingerprint_types,
        'token_format': token_format,
        'fingerprint_nbits': fingerprint_nbits if token_format == 'binary' else None,
        'seed': seed,
        'splits': {
            'test_fraction': float(test_fraction),
            'train_fraction': float(train_fraction),
            'val_fraction': float(1 - train_fraction)
        },
        'sizes': split_stats['sizes'],
        'label_distribution': {
            'train': split_stats['label_distribution']['train'],
            'val': split_stats['label_distribution']['val'],
            'test': split_stats['label_distribution']['test']
        },
        'label_ratios': {
            'train': split_stats['label_distribution']['train_ratios'],
            'val': split_stats['label_distribution']['val_ratios'],
            'test': split_stats['label_distribution']['test_ratios']
        },
        'token_statistics': {
            'mean_length': float(np.mean(sample_lengths)),
            'std_length': float(np.std(sample_lengths)),
            'min_length': int(np.min(sample_lengths)),
            'max_length': int(np.max(sample_lengths)),
            'median_length': float(np.median(sample_lengths))
        },
        'config': OmegaConf.to_container(cfg, resolve=True)
    }

    metadata_path = output_dir / "metadata.yaml"
    with open(metadata_path, 'w') as f:
        yaml.dump(metadata, f, default_flow_style=False, sort_keys=False)

    print("\n" + "=" * 60)
    print("Supervised data preparation complete!")
    print("=" * 60)
    print(f"\nOutput saved to: {output_dir}")
    print(f"\nDirectory structure:")
    print(f"  {output_dir}/")
    print(f"  ├── train/        ({len(train_tokenized):,} molecules)")
    print(f"  ├── val/          ({len(val_tokenized):,} molecules)")
    print(f"  ├── test/         ({len(test_tokenized):,} molecules) [LOCKED]")
    print(f"  ├── tokenizer/    (copied from pretrain)")
    print(f"  └── metadata.yaml")
    print(f"\nTo use for finetuning:")
    print(f"  python scripts/finetune.py data.processed_dir={output_dir}")
    print(f"\nIMPORTANT: The test set at {test_dir} should NOT be modified!")


if __name__ == "__main__":
    main()
