"""
Simplified PyTorch Lightning DataModule for pre-processed molecular datasets.

This module loads pre-tokenized data, avoiding all the complexity of 
vocabulary building and tokenization during training.
"""

import os
import json
from pathlib import Path
from typing import Optional, Union

import pytorch_lightning as pl
from torch.utils.data import DataLoader
from datasets import load_from_disk, DatasetDict

from .tokenizer import MolecularTokenizer
from .transforms import MolecularCollator


class ProcessedDataModule(pl.LightningDataModule):
    """
    Lightning DataModule for loading pre-processed molecular data.

    Supports two modes:
    1. Pretrain mode: Uses processed_dir containing dataset/ and tokenizer/
    2. Finetune mode: Uses separate train_dir, val_dir, tokenizer_dir paths
       (Test data is NOT loaded - use separately for final evaluation)
    """

    def __init__(
        self,
        # Pretrain mode: single directory
        processed_dir: Optional[Union[str, Path]] = None,
        # Finetune mode: separate directories
        train_dir: Optional[Union[str, Path]] = None,
        val_dir: Optional[Union[str, Path]] = None,
        tokenizer_dir: Optional[Union[str, Path]] = None,
        # Common settings
        batch_size: int = 32,
        num_workers: int = 4,
        prefetch_factor: int = 2,
        pin_memory: bool = True,
        persistent_workers: bool = True,
        task_type: str = "pretrain",
        mlm_probability: float = 0.15,
        segment_mlm_probability: float = 0.0,
        span_shuffle_probability: float = 0.0,
        seed: int = 42
    ):
        """
        Initialize the processed data module.

        Args:
            processed_dir: Directory containing processed dataset and tokenizer (pretrain mode)
            train_dir: Directory containing training data (finetune mode)
            val_dir: Directory containing validation data (finetune mode)
            tokenizer_dir: Directory containing tokenizer (finetune mode)
            batch_size: Batch size for dataloaders
            num_workers: Number of workers for dataloaders
            prefetch_factor: Number of batches to prefetch per worker
            pin_memory: Pin memory for faster GPU transfer
            persistent_workers: Keep workers alive between epochs
            task_type: "pretrain" or "classification"
            mlm_probability: Masking probability for MLM
            seed: Random seed
        """
        super().__init__()
        self.save_hyperparameters()

        # Determine mode based on provided arguments
        self.use_separate_dirs = train_dir is not None

        if self.use_separate_dirs:
            # Finetune mode: separate directories
            self.train_dir = Path(train_dir) if train_dir else None
            self.val_dir = Path(val_dir) if val_dir else None
            self.tokenizer_dir = Path(tokenizer_dir) if tokenizer_dir else None
            self.processed_dir = None

            # Validate paths
            if self.train_dir and not self.train_dir.exists():
                raise ValueError(f"Train directory not found: {self.train_dir}")
            if self.val_dir and not self.val_dir.exists():
                raise ValueError(f"Validation directory not found: {self.val_dir}")
            if self.tokenizer_dir and not self.tokenizer_dir.exists():
                raise ValueError(f"Tokenizer directory not found: {self.tokenizer_dir}")
        else:
            # Pretrain mode: single directory
            if processed_dir is None:
                raise ValueError("Must provide either processed_dir or train_dir/val_dir/tokenizer_dir")
            self.processed_dir = Path(processed_dir)
            self.train_dir = None
            self.val_dir = None
            self.tokenizer_dir = None

            if not self.processed_dir.exists():
                raise ValueError(f"Processed data directory not found: {self.processed_dir}")

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.prefetch_factor = prefetch_factor if num_workers > 0 else None
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers if num_workers > 0 else False
        self.task_type = task_type
        self.mlm_probability = mlm_probability
        self.segment_mlm_probability = segment_mlm_probability
        self.span_shuffle_probability = span_shuffle_probability
        self.seed = seed

        # These will be initialized in setup()
        self.datasets = None
        self.tokenizer = None
        self.train_dataset = None
        self.val_dataset = None
        self.collator = None
        self.val_collator = None
    
    def prepare_data(self):
        """Nothing to prepare - data is already processed."""
        pass
    
    def setup(self, stage: Optional[str] = None):
        """Load processed data and create splits."""

        if self.use_separate_dirs:
            # Finetune mode: load from separate directories
            self._setup_finetune_mode()
        else:
            # Pretrain mode: load from processed_dir
            self._setup_pretrain_mode()

        # Setup collators
        pad_token_id = self.tokenizer.pad_token_id
        mask_token_id = self.tokenizer.mask_token_id if self.task_type == "pretrain" else None

        # Training collator (with MLM masking and span shuffling)
        self.collator = MolecularCollator(
            pad_token_id=pad_token_id,
            mask_token_id=mask_token_id,
            mlm_probability=self.mlm_probability,
            vocab_size=self.tokenizer.vocab_size,
            segment_mlm_probability=self.segment_mlm_probability,
            span_shuffle_probability=self.span_shuffle_probability,
        )

        # Validation collator (NO shuffling, NO MLM masking for deterministic evaluation)
        self.val_collator = MolecularCollator(
            pad_token_id=pad_token_id,
            mask_token_id=mask_token_id,  # Still need masking for MLM loss computation
            mlm_probability=self.mlm_probability,  # Same masking rate for comparable loss
            vocab_size=self.tokenizer.vocab_size,
            segment_mlm_probability=self.segment_mlm_probability,
            span_shuffle_probability=0.0,  # No shuffling during validation
        )

    def _setup_finetune_mode(self):
        """Setup for finetuning: load train/val from separate directories."""
        # Load tokenizer
        print(f"\nLoading tokenizer from {self.tokenizer_dir}")
        self.tokenizer = MolecularTokenizer.from_pretrained(str(self.tokenizer_dir))
        print(f"Tokenizer vocabulary size: {self.tokenizer.vocab_size}")

        # STRICT: Validate tokenizer loaded correctly
        if self.tokenizer.vocab_size == 0:
            raise ValueError(
                f"CRITICAL: Tokenizer vocab_size is 0! This indicates the tokenizer failed to load. "
                f"Check that tokenizer_dir points to a directory containing vocab.json. "
                f"Current tokenizer_dir: {self.tokenizer_dir}. "
                f"If tokenizer files are in a subdirectory (e.g., 'tokenizer/'), update the path accordingly."
            )

        # Load train dataset
        print(f"Loading training data from {self.train_dir}")
        self.train_dataset = load_from_disk(str(self.train_dir))
        print(f"  Train: {len(self.train_dataset):,} molecules")

        # Load validation dataset
        print(f"Loading validation data from {self.val_dir}")
        self.val_dataset = load_from_disk(str(self.val_dir))
        print(f"  Val: {len(self.val_dataset):,} molecules")

        # Test is NOT loaded - kept separate for final evaluation

    def _setup_pretrain_mode(self):
        """Setup for pretraining: load from processed_dir."""
        # Load tokenizer
        print(f"\nLoading tokenizer from {self.processed_dir / 'tokenizer'}")
        self.tokenizer = MolecularTokenizer.from_pretrained(
            str(self.processed_dir / "tokenizer")
        )
        print(f"Tokenizer vocabulary size: {self.tokenizer.vocab_size}")

        # STRICT: Validate tokenizer loaded correctly
        if self.tokenizer.vocab_size == 0:
            raise ValueError(
                f"CRITICAL: Tokenizer vocab_size is 0! This indicates the tokenizer failed to load. "
                f"Check that processed_dir contains a 'tokenizer/' subdirectory with vocab.json. "
                f"Current processed_dir: {self.processed_dir}."
            )

        # Load dataset from dataset/ folder
        dataset_dir = self.processed_dir / "dataset"
        if not dataset_dir.exists():
            raise ValueError(f"Dataset directory not found: {dataset_dir}")

        print(f"Loading processed dataset from {dataset_dir}")
        loaded_data = load_from_disk(str(dataset_dir))

        if isinstance(loaded_data, DatasetDict):
            self.datasets = loaded_data
            print(f"Loaded dataset with splits:")
            for split_name, split_data in self.datasets.items():
                print(f"  {split_name}: {len(split_data):,} molecules")
        else:
            self.datasets = DatasetDict({'train': loaded_data})
            print(f"Loaded {len(loaded_data):,} molecules (no splits)")

        # Setup train/val from DatasetDict
        self._setup_train_val_from_datasets()
    
    def _setup_train_val_from_datasets(self):
        """Setup train/validation datasets from DatasetDict (pretrain mode)."""

        # Use pre-split data if available
        if 'train' in self.datasets:
            self.train_dataset = self.datasets['train']
            print(f"\nUsing pre-split train set: {len(self.train_dataset):,} molecules")
        else:
            raise ValueError("No 'train' split found in processed dataset!")

        if 'validation' in self.datasets:
            self.val_dataset = self.datasets['validation']
            print(f"Using pre-split validation set: {len(self.val_dataset):,} molecules")
        elif 'val' in self.datasets:
            self.val_dataset = self.datasets['val']
            print(f"Using pre-split validation set: {len(self.val_dataset):,} molecules")
        else:
            print("Warning: No validation split found. Using 10% of train for validation...")
            train_size = len(self.train_dataset)
            val_size = int(train_size * 0.1)
            self.val_dataset = self.train_dataset.select(range(val_size))
            self.train_dataset = self.train_dataset.select(range(val_size, train_size))
            print(f"Created validation set: {len(self.val_dataset):,} molecules")
    
    def _process_batch_items(self, batch):
        """Process batch items into format expected by collator."""
        processed_batch = []
        for item in batch:
            processed_item = {
                'input_ids': item['input_ids'],
                'compound_id': item.get('COMPOUND_ID', ''),
                'library_id': item.get('LIBRARY_ID', '')
            }

            # Include segment_ids if present (for segment attention)
            if 'segment_ids' in item:
                processed_item['segment_ids'] = item['segment_ids']

            if self.task_type == "classification":
                # Convert to int for classification (0, 1, 2, ...)
                # STRICT: Require LABEL field - don't silently default to 0
                if 'LABEL' not in item:
                    raise ValueError(
                        f"Classification task requires 'LABEL' field but it's missing from data item. "
                        f"Available keys: {list(item.keys())}. "
                        f"Check that your data was prepared correctly for classification."
                    )
                processed_item['labels'] = int(item['LABEL'])

            processed_batch.append(processed_item)
        return processed_batch

    def _collate_fn(self, batch):
        """Custom collate function for training (with shuffling)."""
        processed_batch = self._process_batch_items(batch)
        return self.collator(processed_batch)

    def _val_collate_fn(self, batch):
        """Custom collate function for validation (no shuffling)."""
        processed_batch = self._process_batch_items(batch)
        return self.val_collator(processed_batch)
    
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            prefetch_factor=self.prefetch_factor,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
            collate_fn=self._collate_fn
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            prefetch_factor=self.prefetch_factor,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
            collate_fn=self._val_collate_fn
        )
    
    def test_dataloader(self):
        # Test data is NOT loaded by this DataModule.
        # Use a separate evaluation script for final test evaluation.
        return None

    def predict_dataloader(self):
        return None
    
    @property
    def vocab_size(self):
        """Get vocabulary size."""
        return self.tokenizer.vocab_size if self.tokenizer else 0
    
    def get_tokenizer(self):
        """Get tokenizer instance."""
        return self.tokenizer