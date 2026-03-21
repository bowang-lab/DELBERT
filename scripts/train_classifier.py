#!/usr/bin/env python3
"""
Classification training script for DELBERT using PyTorch Lightning and Hydra.

Supports two training modes:
1. From scratch: Train with randomly initialized weights (no pretrained checkpoint)
2. Finetuning: Initialize from a pretrained MLM checkpoint

Usage:
    # Train from pretrained checkpoint (finetuning)
    python scripts/train_classifier.py pretrained_checkpoint=path/to/checkpoint

    # Train from scratch (random initialization)
    python scripts/train_classifier.py pretrained_checkpoint=null

    # Multi-run experiments
    python scripts/train_classifier.py --multirun model.finetuning_strategy=frozen,full
"""

import os
# Fix for OpenMP conflict on macOS
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'


import hydra
from omegaconf import DictConfig, OmegaConf
import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    LearningRateMonitor,
    TQDMProgressBar
)
from pytorch_lightning.loggers import WandbLogger
import torch
import wandb
from pathlib import Path

from delbert.data.processed_data_module import ProcessedDataModule
from delbert.models.classification_model import ClassificationModel


def setup_callbacks(cfg: DictConfig):
    """Setup training callbacks."""
    callbacks = []

    # Model checkpointing
    # Use the monitored metric in the filename
    monitor_metric = cfg.train.monitor
    checkpoint_callback = ModelCheckpoint(
        dirpath=cfg.train.checkpoint_dir,
        filename=f'epoch={{epoch:02d}}-{monitor_metric}={{{monitor_metric}:.3f}}',
        monitor=monitor_metric,
        mode=cfg.train.mode,
        save_top_k=cfg.train.save_top_k,
        save_last=cfg.train.save_last,
        verbose=True,
        auto_insert_metric_name=False  # Prevent double metric names
    )
    callbacks.append(checkpoint_callback)

    # Early stopping
    if cfg.train.early_stopping and cfg.train.early_stopping.get('enabled', True):
        early_stop_callback = EarlyStopping(
            monitor=cfg.train.early_stopping.monitor,
            patience=cfg.train.early_stopping.patience,
            mode=cfg.train.early_stopping.mode,
            min_delta=cfg.train.early_stopping.min_delta,
            verbose=True
        )
        callbacks.append(early_stop_callback)

    # Learning rate monitor
    callbacks.append(LearningRateMonitor(logging_interval='step'))

    # Progress bar
    callbacks.append(TQDMProgressBar(refresh_rate=10))

    return callbacks


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    """Main classification training function."""

    # Get project root directory (parent of scripts/)
    # This ensures paths work regardless of where the script is run from
    project_root = Path(__file__).parent.parent.resolve()

    # Print config
    print(OmegaConf.to_yaml(cfg))

    # Set PyTorch optimizations
    if cfg.train.get('matmul_precision'):
        torch.set_float32_matmul_precision(cfg.train.matmul_precision)
        print(f"Set float32 matmul precision to: {cfg.train.matmul_precision}")

    if cfg.train.get('enable_tf32', False):
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cuda.matmul.allow_tf32 = True
        print("Enabled TF32 for faster training")

    # Determine training mode: pretrained or from scratch
    pretrained_checkpoint = cfg.get('pretrained_checkpoint')

    # Handle null/None values from config
    if pretrained_checkpoint in [None, 'null', 'None', '']:
        pretrained_checkpoint = None
        training_mode = "from_scratch"
        print("\n" + "=" * 60)
        print("Training Mode: FROM SCRATCH (random initialization)")
        print("=" * 60)
    else:
        training_mode = "finetuning"
        # Resolve pretrained checkpoint path (relative to original cwd)
        if not os.path.isabs(pretrained_checkpoint):
            pretrained_checkpoint = str(project_root / pretrained_checkpoint)

        if not os.path.exists(pretrained_checkpoint):
            raise ValueError(f"Pretrained checkpoint not found: {pretrained_checkpoint}")

        print("\n" + "=" * 60)
        print("Training Mode: FINETUNING (from pretrained checkpoint)")
        print("=" * 60)
        print(f"Pretrained checkpoint: {pretrained_checkpoint}")

    # Set seed for reproducibility
    pl.seed_everything(cfg.seed)

    # Get Hydra output directory
    output_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir

    # Initialize logger
    logger = None
    if cfg.wandb.enabled:
        # Configure W&B to use experiment output directory
        wandb_dir = os.path.join(output_dir, 'wandb')
        os.makedirs(wandb_dir, exist_ok=True)
        os.environ['WANDB_DIR'] = wandb_dir

        # Set offline mode if requested
        if cfg.wandb.offline:
            os.environ['WANDB_MODE'] = 'offline'
            print("W&B running in offline mode")

        # Initialize W&B
        wandb_logger = WandbLogger(
            project=cfg.wandb.project,
            entity=cfg.wandb.entity,
            name=cfg.wandb.name,
            group=cfg.wandb.group,
            tags=cfg.wandb.tags + [training_mode],
            config=OmegaConf.to_container(cfg, resolve=True),
            save_dir=wandb_dir  # Save W&B data in experiment directory
        )
        logger = wandb_logger
        print(f"W&B logging enabled: {cfg.wandb.project}/{cfg.wandb.name}")
    else:
        print("W&B logging disabled")

    # Create data module
    print("\n=== Setting up data module ===")

    # Validate data paths
    if not cfg.data.get('train_dir') or not cfg.data.get('val_dir') or not cfg.data.get('tokenizer_dir'):
        raise ValueError(
            "Data paths not specified in config!\n"
            "Required: data.train_dir, data.val_dir, data.tokenizer_dir\n"
            "Run prepare_supervised_data.py first to create processed data."
        )

    # Resolve data paths (relative to original cwd)
    train_dir = cfg.data.train_dir
    val_dir = cfg.data.val_dir
    tokenizer_dir = cfg.data.tokenizer_dir

    if not os.path.isabs(train_dir):
        train_dir = str(project_root / train_dir)
    if not os.path.isabs(val_dir):
        val_dir = str(project_root / val_dir)
    if not os.path.isabs(tokenizer_dir):
        tokenizer_dir = str(project_root / tokenizer_dir)

    print(f"Train data: {train_dir}")
    print(f"Val data: {val_dir}")
    print(f"Tokenizer: {tokenizer_dir}")

    data_module = ProcessedDataModule(
        train_dir=train_dir,
        val_dir=val_dir,
        tokenizer_dir=tokenizer_dir,
        batch_size=cfg.data.batch_size,
        num_workers=cfg.data.num_workers,
        prefetch_factor=cfg.data.get('prefetch_factor', 2),
        pin_memory=cfg.data.get('pin_memory', True),
        persistent_workers=cfg.data.get('persistent_workers', True),
        task_type="classification",  # Always classification
        seed=cfg.seed
    )

    # Setup data
    data_module.setup("fit")
    vocab_size = data_module.tokenizer.vocab_size
    print(f"Vocabulary size: {vocab_size}")
    print(f"Train samples: {len(data_module.train_dataset)}")
    print(f"Val samples: {len(data_module.val_dataset)}")

    # Create model
    print("\n=== Creating model ===")

    # Build strategy params from config
    strategy_params = {}
    if hasattr(cfg, 'model') and cfg.model.get('strategy_params'):
        strategy_params = dict(cfg.model.strategy_params)
    elif hasattr(cfg, 'finetune') and cfg.finetune.get('strategy_params'):
        # Backwards compatibility with old config structure
        strategy_params = dict(cfg.finetune.strategy_params)

    # Get model config for training from scratch
    model_config = None
    if training_mode == "from_scratch" and hasattr(cfg, 'model') and cfg.model.get('architecture'):
        model_config = OmegaConf.to_container(cfg.model.architecture, resolve=True)

    # Get model settings (support both 'model' and 'finetune' config keys for backwards compatibility)
    model_cfg = cfg.get('model', cfg.get('finetune', {}))

    model = ClassificationModel(
        pretrained_checkpoint=pretrained_checkpoint,
        vocab_size=vocab_size if training_mode == "from_scratch" else None,
        model_config=model_config,
        num_labels=model_cfg.get('num_labels', 2),
        classifier_dropout=model_cfg.get('classifier_dropout', 0.1),
        classifier_pooling=model_cfg.get('classifier_pooling', 'cls'),
        learning_rate=cfg.train.learning_rate,
        warmup_steps=cfg.train.get('warmup_steps'),
        warmup_ratio=cfg.train.get('warmup_ratio', 0.1),
        weight_decay=cfg.train.weight_decay,
        loss_type=model_cfg.get('loss_type', 'ce'),
        pos_class_weight=model_cfg.get('pos_class_weight', None),
        focal_gamma=model_cfg.get('focal_gamma', 2.0),
        focal_alpha=model_cfg.get('focal_alpha', None),
        finetuning_strategy=model_cfg.get('finetuning_strategy', 'full'),
        strategy_params=strategy_params,
    )

    # Log model summary
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {total_params:,} total, {trainable_params:,} trainable")

    # Setup callbacks
    callbacks = setup_callbacks(cfg)

    # Create trainer
    trainer = pl.Trainer(
        max_epochs=cfg.train.num_epochs,
        accelerator=cfg.train.accelerator,
        devices=cfg.train.devices,
        num_nodes=cfg.train.num_nodes,
        strategy=cfg.train.strategy,
        precision=cfg.train.precision,
        accumulate_grad_batches=cfg.train.gradient_accumulation_steps,
        gradient_clip_val=cfg.train.gradient_clip_val,
        callbacks=callbacks,
        logger=logger,
        log_every_n_steps=cfg.train.get('log_every_n_steps', 10),
        val_check_interval=cfg.train.get('val_check_interval', 1.0),
        limit_val_batches=cfg.train.get('limit_val_batches', 1.0),
        num_sanity_val_steps=cfg.train.get('num_sanity_val_steps', 0),
        enable_checkpointing=cfg.train.get('enable_checkpointing', True),
        deterministic=cfg.train.get('deterministic', False),
        default_root_dir=output_dir,  # Save Lightning logs in experiment dir
    )

    # Check for resume checkpoint
    ckpt_path = None
    if cfg.train.get('resume_from_checkpoint'):
        ckpt_path = cfg.train.resume_from_checkpoint
        if not os.path.exists(ckpt_path):
            raise ValueError(f"Checkpoint not found: {ckpt_path}")
        print(f"\n=== Resuming from checkpoint: {ckpt_path} ===")

    # Train model
    print(f"\n=== Starting {training_mode} training ===")
    trainer.fit(model, data_module, ckpt_path=ckpt_path)

    # Note: Test evaluation is NOT done here - test set is kept separate
    # Use a separate evaluation script after training is complete

    # Save final model
    final_model_path = os.path.join(output_dir, "final_model")
    os.makedirs(final_model_path, exist_ok=True)

    # Load best checkpoint for saving
    best_model = ClassificationModel.load_from_checkpoint(
        trainer.checkpoint_callback.best_model_path
    )

    # Save model weights
    torch.save(best_model.model.state_dict(), os.path.join(final_model_path, "pytorch_model.bin"))

    # Save config
    best_model.model.config.save_pretrained(final_model_path)

    # Save tokenizer
    data_module.tokenizer.save_pretrained(final_model_path)

    print(f"\nTraining completed! Model saved to {final_model_path}")
    print(f"\nTo evaluate on test set, use a separate evaluation script.")

    # Log metrics summary
    print("\n=== Final Validation Metrics ===")
    if hasattr(trainer, 'logged_metrics'):
        for key, value in trainer.logged_metrics.items():
            if 'val_' in key:
                print(f"{key}: {value:.4f}")

    # Close W&B run if enabled
    if cfg.wandb.enabled:
        wandb.finish()


if __name__ == "__main__":
    main()
