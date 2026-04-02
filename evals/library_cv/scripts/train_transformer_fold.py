#!/usr/bin/env python3
"""
Train a SINGLE fold of cross-validation.

This script is designed to be called by run_cv_orchestrator.py for each fold.
Running each fold as a separate process avoids DDP cleanup issues between folds.

Supports multi-GPU (DDP) training within each fold.

Usage:
    python evals/library_cv/scripts/train_single_fold.py \
        --fold_idx 0 \
        --output_dir /path/to/output \
        --config evals/library_cv/configs/transformer_cv.yaml
"""

import argparse
import os
import sys
import json
from pathlib import Path
from time import time
from typing import Dict, Optional

import numpy as np
import pandas as pd
import yaml
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    TQDMProgressBar
)
from pytorch_lightning.loggers import WandbLogger
from datasets import load_from_disk
from torch.utils.data import DataLoader
from omegaconf import OmegaConf

# Add project root to path

from delbert.data.tokenizer import MolecularTokenizer
from delbert.data.transforms import MolecularCollator
from delbert.data.cv_utils import calculate_ranking_metrics
from delbert.models.classification_model import ClassificationModel


class FoldDataModule(pl.LightningDataModule):
    """DataModule for a single CV fold."""

    def __init__(
        self,
        dataset_path: str,
        tokenizer: MolecularTokenizer,
        train_indices: np.ndarray,
        val_indices: np.ndarray,
        test_indices: np.ndarray,
        batch_size: int = 32,
        num_workers: int = 4,
        prefetch_factor: int = 2,
        pin_memory: bool = True,
        persistent_workers: bool = True
    ):
        super().__init__()
        self.dataset_path = dataset_path
        self.tokenizer = tokenizer
        self.train_indices = train_indices.tolist()
        self.val_indices = val_indices.tolist()
        self.test_indices = test_indices.tolist()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.prefetch_factor = prefetch_factor if num_workers > 0 else None
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers if num_workers > 0 else False

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.collator = None

    def prepare_data(self):
        pass

    def setup(self, stage: Optional[str] = None):
        full_dataset = load_from_disk(self.dataset_path)
        self.train_dataset = full_dataset.select(self.train_indices)
        self.val_dataset = full_dataset.select(self.val_indices)
        self.test_dataset = full_dataset.select(self.test_indices)

        # STRICT VALIDATION: Check that LABEL column exists and has positive samples
        self._validate_labels(full_dataset)

        # mask_token_id=None for classification (no MLM)
        self.collator = MolecularCollator(
            pad_token_id=self.tokenizer.pad_token_id,
            mask_token_id=None,
            mlm_probability=0.0,
            vocab_size=self.tokenizer.vocab_size
        )

    def _validate_labels(self, dataset):
        """Validate that LABEL column exists and has the expected distribution."""
        # Check column exists
        if 'LABEL' not in dataset.column_names and 'labels' not in dataset.column_names:
            raise ValueError(
                f"Neither 'LABEL' nor 'labels' column found in dataset! "
                f"Available columns: {dataset.column_names}. "
                f"This indicates a data preparation or caching issue."
            )

        label_col = 'LABEL' if 'LABEL' in dataset.column_names else 'labels'

        # Check label distribution in training set
        train_labels = [self.train_dataset[i][label_col] for i in range(min(1000, len(self.train_dataset)))]
        pos_count = sum(1 for l in train_labels if l == 1)
        neg_count = sum(1 for l in train_labels if l == 0)

        print(f"Label validation (first {len(train_labels)} train samples):")
        print(f"  Positives: {pos_count} ({100*pos_count/len(train_labels):.2f}%)")
        print(f"  Negatives: {neg_count} ({100*neg_count/len(train_labels):.2f}%)")

        if pos_count == 0:
            raise ValueError(
                f"CRITICAL: No positive labels found in first {len(train_labels)} training samples! "
                f"All labels appear to be 0. This will cause random predictions. "
                f"Check that LABEL column was properly preserved during caching/processing."
            )

        if neg_count == 0:
            raise ValueError(
                f"CRITICAL: No negative labels found in first {len(train_labels)} training samples! "
                f"This suggests a data loading issue."
            )

    def _collate_fn(self, batch):
        processed_batch = []
        for item in batch:
            # STRICT: Require LABEL column - no silent fallback to 0
            label = item.get('LABEL')
            if label is None:
                label = item.get('labels')
            if label is None:
                raise ValueError(
                    f"LABEL/labels column not found in dataset item! "
                    f"Available keys: {list(item.keys())}. "
                    f"This indicates a data preparation or caching issue."
                )
            processed_item = {
                'input_ids': item['input_ids'],
                'labels': int(label)
            }
            # Include segment_ids if present (for segment-based attention)
            if 'segment_ids' in item:
                processed_item['segment_ids'] = item['segment_ids']
            processed_batch.append(processed_item)
        return self.collator(processed_batch)

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
            collate_fn=self._collate_fn
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            prefetch_factor=self.prefetch_factor,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
            collate_fn=self._collate_fn
        )


def train_fold(
    fold_idx: int,
    train_indices: np.ndarray,
    val_indices: np.ndarray,
    test_indices: np.ndarray,
    dataset_path: str,
    tokenizer_path: str,
    cfg: dict,
    output_dir: Path,
    project_root: Path
) -> Dict[str, float]:
    """Train and evaluate a single fold."""
    print(f"\n{'='*60}")
    print(f"FOLD {fold_idx}")
    print(f"{'='*60}")
    print(f"Train: {len(train_indices)}, Val: {len(val_indices)}, Test: {len(test_indices)}")

    # Load tokenizer
    tokenizer = MolecularTokenizer.from_pretrained(tokenizer_path)

    # STRICT: Validate tokenizer loaded correctly
    if tokenizer.vocab_size == 0:
        raise ValueError(
            f"CRITICAL: Tokenizer vocab_size is 0! This indicates the tokenizer failed to load. "
            f"Check that tokenizer_path points to a directory containing vocab.json. "
            f"Current tokenizer_path: {tokenizer_path}. "
            f"If tokenizer files are in a subdirectory (e.g., 'tokenizer/'), update the path accordingly."
        )

    # Create DataModule
    data_module = FoldDataModule(
        dataset_path=dataset_path,
        tokenizer=tokenizer,
        train_indices=train_indices,
        val_indices=val_indices,
        test_indices=test_indices,
        batch_size=cfg['data']['batch_size'],
        num_workers=cfg['data']['num_workers'],
        prefetch_factor=cfg['data'].get('prefetch_factor', 2),
        pin_memory=cfg['data'].get('pin_memory', True),
        persistent_workers=cfg['data'].get('persistent_workers', True)
    )
    data_module.setup("fit")

    # Resolve pretrained checkpoint
    pretrained_checkpoint = cfg.get('pretrained_checkpoint')
    if pretrained_checkpoint and not os.path.isabs(pretrained_checkpoint):
        pretrained_checkpoint = str(project_root / pretrained_checkpoint)

    # Build strategy params
    strategy_params = cfg['model'].get('strategy_params', {})

    # STRICT: Require finetuning_strategy to be explicitly specified
    # Don't silently default to 'full' - user should be explicit about training mode
    if 'finetuning_strategy' not in cfg['model']:
        raise ValueError(
            "finetuning_strategy must be explicitly specified in model config. "
            "Choose from: 'full' (all params trainable), 'frozen' (encoder frozen), 'lora' (LoRA adapters). "
            "This prevents accidentally using full finetuning when LoRA/frozen was intended."
        )
    finetuning_strategy = cfg['model']['finetuning_strategy']

    # Determine if training from scratch (no pretrained checkpoint)
    from_scratch = pretrained_checkpoint is None

    # For from-scratch training, we need vocab_size and model_config
    vocab_size = None
    model_config = None

    if from_scratch:
        # Get vocab_size from config or tokenizer
        vocab_size = cfg.get('vocab_size')
        if vocab_size is None:
            vocab_size = tokenizer.vocab_size
            print(f"Using vocab_size from tokenizer: {vocab_size}")
        else:
            # Validate config vocab_size matches tokenizer
            if vocab_size != tokenizer.vocab_size:
                raise ValueError(
                    f"vocab_size in config ({vocab_size}) does not match tokenizer ({tokenizer.vocab_size}). "
                    f"Remove vocab_size from config to use tokenizer's value, or fix the mismatch."
                )
            print(f"Using vocab_size from config: {vocab_size}")

        # Build model_config from config file's model section
        # These are the architecture parameters needed for from-scratch initialization
        model_config = {
            'hidden_size': cfg['model'].get('hidden_size'),
            'num_hidden_layers': cfg['model'].get('num_hidden_layers'),
            'num_attention_heads': cfg['model'].get('num_attention_heads'),
            'intermediate_size': cfg['model'].get('intermediate_size'),
            'max_position_embeddings': cfg['model'].get('max_position_embeddings'),
            'global_rope_theta': cfg['model'].get('global_rope_theta'),
            'local_attention': cfg['model'].get('local_attention'),
            'hidden_dropout_prob': cfg['model'].get('hidden_dropout_prob'),
            'attention_probs_dropout_prob': cfg['model'].get('attention_probs_dropout_prob'),
            'use_segment_attention': cfg['model'].get('use_segment_attention', False),
            'segment_position_encoding': cfg['model'].get('segment_position_encoding', 'absolute'),
        }

        # Remove None values (will use defaults in ClassificationModel)
        model_config = {k: v for k, v in model_config.items() if v is not None}

        # STRICT: Require essential architecture params for from-scratch
        required_params = ['hidden_size', 'num_hidden_layers', 'num_attention_heads']
        missing = [p for p in required_params if p not in model_config]
        if missing:
            raise ValueError(
                f"From-scratch training requires model architecture parameters. "
                f"Missing: {missing}. "
                f"Add these to the 'model' section of your config."
            )

        print(f"Training from scratch with architecture: {model_config}")

        # Validate finetuning strategy for from-scratch
        if finetuning_strategy != 'full':
            raise ValueError(
                f"From-scratch training requires finetuning_strategy='full', "
                f"got '{finetuning_strategy}'. LoRA and frozen strategies only make sense "
                f"when finetuning from a pretrained checkpoint."
            )

    # Create model
    model = ClassificationModel(
        pretrained_checkpoint=pretrained_checkpoint,
        vocab_size=vocab_size,
        model_config=model_config,
        num_labels=cfg['model']['num_labels'],
        classifier_dropout=cfg['model'].get('classifier_dropout', 0.1),
        classifier_pooling=cfg['model'].get('classifier_pooling', 'mean'),
        learning_rate=cfg['train']['learning_rate'],
        warmup_ratio=cfg['train'].get('warmup_ratio', 0.1),
        weight_decay=cfg['train'].get('weight_decay', 0.01),
        loss_type=cfg['model'].get('loss_type', 'ce'),
        pos_class_weight=cfg['model'].get('pos_class_weight', None),
        focal_gamma=cfg['model'].get('focal_gamma', 2.0),
        focal_alpha=cfg['model'].get('focal_alpha', None),
        finetuning_strategy=finetuning_strategy,
        strategy_params=strategy_params,
    )

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: {total_params:,} total, {trainable_params:,} trainable params")

    # STRICT: Validate segment embedding consistency between pretrained model and data config
    model_uses_segment_embeds = getattr(model.model.config, 'use_segment_embeddings', False)
    model_uses_segment_attention = getattr(model.model.config, 'use_segment_attention', False)
    data_returns_segment_ids = cfg['data'].get('return_segment_ids', False)

    if model_uses_segment_embeds or model_uses_segment_attention:
        if not data_returns_segment_ids:
            raise ValueError(
                f"CRITICAL: Pretrained model requires segment_ids but data config has "
                f"'return_segment_ids: {data_returns_segment_ids}'!\n"
                f"  Model use_segment_embeddings: {model_uses_segment_embeds}\n"
                f"  Model use_segment_attention: {model_uses_segment_attention}\n"
                f"Fix: Set 'return_segment_ids: true' in data config to match pretrained model."
            )
        print(f"Segment config validation PASSED:")
        print(f"  use_segment_embeddings: {model_uses_segment_embeds}")
        print(f"  use_segment_attention: {model_uses_segment_attention}")
        print(f"  return_segment_ids: {data_returns_segment_ids}")

    # STRICT: Validate LoRA actually wrapped layers if configured
    # (finetuning_strategy already extracted and validated above)
    if finetuning_strategy == 'lora':
        trainable_names = [n for n, p in model.named_parameters() if p.requires_grad]
        lora_params = [n for n in trainable_names if 'lora' in n.lower()]

        if len(lora_params) == 0:
            # Check if any parameters are trainable at all
            if trainable_params == 0:
                raise RuntimeError(
                    f"LoRA configured but NO trainable parameters found! "
                    f"target_modules={strategy_params.get('target_modules')} "
                    f"does not match any modules in the model architecture. "
                    f"Check that target_modules matches the pretrained model's layer names."
                )
            else:
                # Some params trainable but none are LoRA - likely just classifier head
                print(f"WARNING: LoRA configured but no 'lora' parameters found in trainable params.")
                print(f"  Trainable params ({len(trainable_names)}): {trainable_names[:5]}...")
                print(f"  target_modules: {strategy_params.get('target_modules')}")
                print(f"  This may indicate LoRA failed to wrap the intended layers!")
        else:
            print(f"LoRA validation PASSED: {len(lora_params)} LoRA parameters found")
            # Show a sample of LoRA param names
            print(f"  Sample LoRA params: {lora_params[:3]}")

    # Setup callbacks
    fold_output_dir = output_dir / f"fold_{fold_idx}"
    fold_output_dir.mkdir(parents=True, exist_ok=True)

    callbacks = [
        TQDMProgressBar(refresh_rate=10),
    ]

    checkpoint_callback = ModelCheckpoint(
        dirpath=fold_output_dir / "checkpoints",
        filename=f"fold{fold_idx}-" + "{epoch:02d}-{val_pr_auc:.4f}",
        monitor=cfg['train'].get('monitor', 'val_pr_auc'),
        mode=cfg['train'].get('mode', 'max'),
        save_top_k=cfg['train'].get('save_top_k', 1),
        save_last=cfg['train'].get('save_last', False),
    )
    callbacks.append(checkpoint_callback)

    if cfg['train'].get('early_stopping', {}).get('enabled', True):
        es_config = cfg['train']['early_stopping']
        early_stop = EarlyStopping(
            monitor=es_config.get('monitor', 'val_pr_auc'),
            patience=es_config.get('patience', 3),
            mode=es_config.get('mode', 'max'),
            min_delta=es_config.get('min_delta', 0.001),
        )
        callbacks.append(early_stop)

    # W&B logger for this fold
    wandb_logger = None
    if cfg['wandb'].get('enabled', False):
        import wandb
        wandb_dir = output_dir / 'wandb'
        wandb_dir.mkdir(exist_ok=True)
        os.environ['WANDB_DIR'] = str(wandb_dir)

        wandb_logger = WandbLogger(
            project=cfg['wandb']['project'],
            entity=cfg['wandb'].get('entity'),
            name=f"{cfg['experiment_name']}_fold{fold_idx}",
            group=cfg['wandb'].get('group', cfg['experiment_name']),
            tags=list(cfg['wandb'].get('tags', [])) + [f'fold_{fold_idx}'],
            config=cfg,
            save_dir=str(wandb_dir)
        )

    # Create trainer - use configured devices and strategy
    trainer = pl.Trainer(
        max_epochs=cfg['train']['num_epochs'],
        accelerator=cfg['train']['accelerator'],
        devices=cfg['train']['devices'],
        strategy=cfg['train'].get('strategy', 'auto'),
        precision=cfg['train']['precision'],
        accumulate_grad_batches=cfg['train'].get('gradient_accumulation_steps', 1),
        gradient_clip_val=cfg['train'].get('gradient_clip_val', 1.0),
        callbacks=callbacks,
        logger=wandb_logger,
        log_every_n_steps=cfg['train'].get('log_every_n_steps', 10),
        val_check_interval=cfg['train'].get('val_check_interval', 1.0),
        num_sanity_val_steps=0,
        enable_checkpointing=True,
        default_root_dir=str(fold_output_dir),
    )

    # Train
    start_time = time()
    trainer.fit(model, data_module)
    train_time = time() - start_time
    print(f"Training completed in {train_time:.1f}s")

    # Only run evaluation on rank 0 (main process) to avoid duplicate work and file race conditions
    if not trainer.is_global_zero:
        return {'train_time': train_time}

    # Load best checkpoint for evaluation - STRICT: require best checkpoint
    best_ckpt = checkpoint_callback.best_model_path
    if not best_ckpt or not os.path.exists(best_ckpt):
        raise RuntimeError(
            f"CRITICAL: No best checkpoint found! best_model_path='{best_ckpt}', "
            f"save_top_k={cfg['train'].get('save_top_k')}. "
            f"Cannot evaluate without best checkpoint - using last epoch model could give "
            f"unreliable results. Ensure save_top_k >= 1 in config."
        )
    print(f"Loading best checkpoint: {best_ckpt}")
    model = ClassificationModel.load_from_checkpoint(best_ckpt)

    # Evaluate on test set
    model.eval()
    model.to('cuda' if torch.cuda.is_available() else 'cpu')

    all_probs = []
    all_labels = []

    data_module.setup("test")
    test_loader = data_module.test_dataloader()

    # Extract compound IDs from test dataset for saving with predictions
    test_compound_ids = data_module.test_dataset['COMPOUND_ID']

    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(model.device)
            attention_mask = batch['attention_mask'].to(model.device)
            segment_ids = batch.get('segment_ids')
            if segment_ids is not None:
                segment_ids = segment_ids.to(model.device)
            labels = batch['labels']

            outputs = model.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                segment_ids=segment_ids
            )
            probs = torch.softmax(outputs['logits'], dim=-1)[:, 1].float().cpu().numpy()

            all_probs.extend(probs)
            all_labels.extend(labels.numpy())

    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)

    # Calculate metrics
    metrics = calculate_ranking_metrics(all_labels, all_probs)
    metrics['train_time'] = train_time

    print(f"\nFold {fold_idx} Test Results:")
    for name, value in metrics.items():
        print(f"  {name}: {value:.4f}")

    # Save predictions
    if cfg['output'].get('save_predictions', True):
        pred_df = pd.DataFrame({
            'compound_id': test_compound_ids,
            'label': all_labels,
            'prob': all_probs,
            'fold': fold_idx
        })
        pred_df.to_csv(fold_output_dir / 'test_predictions.csv', index=False)

    # Save metrics for this fold
    with open(fold_output_dir / 'metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)

    # Log TEST metrics to W&B
    if wandb_logger:
        import wandb
        if wandb.run is not None:
            test_metrics = {f"test/{k}": v for k, v in metrics.items()}
            wandb.log(test_metrics)
            wandb.finish()

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Train single CV fold")
    parser.add_argument('--fold_idx', type=int, required=True, help='Fold index (0-based)')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory')
    parser.add_argument('--config', type=str, required=True, help='Config YAML path')
    parser.add_argument('--indices_file', type=str, required=True, help='Path to fold indices JSON')
    parser.add_argument('--dataset_path', type=str, required=True, help='Path to processed dataset')
    parser.add_argument('--tokenizer_path', type=str, required=True, help='Path to tokenizer')
    args = parser.parse_args()

    # Load config
    cfg = OmegaConf.load(args.config)
    cfg = OmegaConf.to_container(cfg, resolve=True)

    # Set PyTorch optimizations
    if cfg['train'].get('matmul_precision'):
        torch.set_float32_matmul_precision(cfg['train']['matmul_precision'])
    if cfg['train'].get('enable_tf32', False):
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cuda.matmul.allow_tf32 = True

    # Set seed
    seed = cfg.get('seed', 42)
    pl.seed_everything(seed + args.fold_idx)

    # Load fold indices
    with open(args.indices_file, 'r') as f:
        indices_data = json.load(f)

    train_indices = np.array(indices_data['train_indices'])
    val_indices = np.array(indices_data['val_indices'])
    test_indices = np.array(indices_data['test_indices'])

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Train
    metrics = train_fold(
        fold_idx=args.fold_idx,
        train_indices=train_indices,
        val_indices=val_indices,
        test_indices=test_indices,
        dataset_path=args.dataset_path,
        tokenizer_path=args.tokenizer_path,
        cfg=cfg,
        output_dir=output_dir,
        project_root=project_root
    )

    print(f"\nFold {args.fold_idx} complete!")
    print(f"Metrics: {metrics}")


if __name__ == "__main__":
    main()
