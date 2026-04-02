#!/usr/bin/env python3
"""
Pretraining script for DELBERT using PyTorch Lightning and Hydra.

Usage:
    python scripts/pretrain.py experiment=pretrain_wdr91
    python scripts/pretrain.py experiment=pretrain_wdr91 training.batch_size=64
    python scripts/pretrain.py --multirun experiment=pretrain_wdr91 training.learning_rate=1e-4,5e-4,1e-3
"""

import os
# Fix for OpenMP conflict on macOS
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import sys

import hydra
from omegaconf import DictConfig, OmegaConf
import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    TQDMProgressBar
)
from pytorch_lightning.loggers import WandbLogger
import torch
import wandb

from delbert.data.processed_data_module import ProcessedDataModule
from delbert.models.pretrain_model import PretrainModel


def setup_callbacks(cfg: DictConfig):
    """Setup training callbacks from config."""
    callbacks = []

    # Model checkpointing - support multiple checkpoint strategies
    for ckpt_cfg in cfg.checkpoints:
        checkpoint_callback = ModelCheckpoint(
            dirpath=ckpt_cfg.dirpath,
            filename=ckpt_cfg.filename,
            monitor=ckpt_cfg.monitor,
            mode=ckpt_cfg.get('mode', 'min'),
            save_top_k=ckpt_cfg.get('save_top_k', 1),
            save_last=ckpt_cfg.get('save_last', False),
            every_n_epochs=ckpt_cfg.get('every_n_epochs', 1),
            save_on_train_epoch_end=ckpt_cfg.get('save_on_train_epoch_end', False),
            verbose=True
        )
        callbacks.append(checkpoint_callback)

    # Early stopping
    if cfg.early_stopping.enabled:
        early_stop_callback = EarlyStopping(
            monitor=cfg.early_stopping.monitor,
            patience=cfg.early_stopping.patience,
            mode=cfg.early_stopping.mode,
            min_delta=cfg.early_stopping.min_delta,
            verbose=True
        )
        callbacks.append(early_stop_callback)

    # Progress bar
    callbacks.append(TQDMProgressBar(refresh_rate=10))

    return callbacks


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    """Main training function."""

    # Print config
    print(OmegaConf.to_yaml(cfg))

    # Set PyTorch optimizations
    if cfg.get('pytorch'):
        if cfg.pytorch.get('matmul_precision'):
            torch.set_float32_matmul_precision(cfg.pytorch.matmul_precision)
            print(f"Set float32 matmul precision to: {cfg.pytorch.matmul_precision}")

        if cfg.pytorch.get('enable_tf32', False):
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cuda.matmul.allow_tf32 = True
            print("Enabled TF32 for faster training")

    # Set seed for reproducibility
    pl.seed_everything(cfg.seed)

    # Initialize logger
    logger = None
    if cfg.wandb.enabled:
        # Configure W&B to use experiment output directory
        wandb_dir = cfg.wandb.get('save_dir', 'wandb')
        os.makedirs(wandb_dir, exist_ok=True)
        os.environ['WANDB_DIR'] = wandb_dir

        # Set offline mode if requested
        if cfg.wandb.offline:
            os.environ['WANDB_MODE'] = 'offline'
            print("W&B running in offline mode")

        # Initialize W&B
        wandb_logger = WandbLogger(
            project=cfg.wandb.project,
            entity=cfg.wandb.get('entity'),
            name=cfg.wandb.name,
            group=cfg.wandb.group,
            tags=cfg.wandb.get('tags', []),
            config=OmegaConf.to_container(cfg, resolve=True),
            save_dir=wandb_dir
        )
        logger = wandb_logger
        print(f"W&B logging enabled: {cfg.wandb.project}/{cfg.wandb.name}")
    else:
        print("W&B logging disabled")

    # Create data module
    print("\n=== Setting up data module ===")

    # Require pre-processed data
    if not cfg.data.get('processed_dir') or not cfg.data.processed_dir:
        raise ValueError(
            "processed_dir not specified in config!\n"
            "Please run prepare_data.py first to preprocess your data:\n"
            "  python scripts/prepare_data.py --config-name data_prep/your_config"
        )

    # Use processed directory from config
    processed_dir = cfg.data.processed_dir
    # Resolve relative paths from project root (before Hydra's chdir)
    if not os.path.isabs(processed_dir):
        processed_dir = os.path.join(hydra.utils.get_original_cwd(), processed_dir)

    print(f"Using pre-processed data from: {processed_dir}")
    data_module = ProcessedDataModule(
        processed_dir=processed_dir,
        batch_size=cfg.training.batch_size,
        num_workers=cfg.training.num_workers,
        prefetch_factor=cfg.training.get('prefetch_factor', 2),
        pin_memory=cfg.training.get('pin_memory', True),
        persistent_workers=cfg.training.get('persistent_workers', True),
        task_type=cfg.data.task_type,
        mlm_probability=cfg.data.mlm_probability,
        segment_mlm_probability=cfg.data.get('segment_mlm_probability', 0.0),
        span_shuffle_probability=cfg.data.get('span_shuffle_probability', 0.0),
        seed=cfg.seed
    )

    # Setup data (builds vocabulary)
    data_module.setup("fit")
    vocab_size = data_module.tokenizer.vocab_size
    print(f"Vocabulary size: {vocab_size}")

    # Create model
    print("\n=== Creating model ===")
    model_kwargs = {
        "config": OmegaConf.to_container(cfg.model),
        "vocab_size": vocab_size,
        "learning_rate": cfg.training.learning_rate,
        "weight_decay": cfg.training.weight_decay,
        "warmup_ratio": cfg.training.get('warmup_ratio', 0.1),
    }

    # Add warmup_steps if specified (overrides warmup_ratio)
    if cfg.training.get('warmup_steps') is not None:
        model_kwargs["warmup_steps"] = cfg.training.warmup_steps

    model = PretrainModel(**model_kwargs)

    # Log model summary
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Setup callbacks
    callbacks = setup_callbacks(cfg)

    # Create trainer
    devices = cfg.training.devices
    accumulate_grad_batches = cfg.training.get('accumulate_grad_batches', 1)

    trainer = pl.Trainer(
        max_epochs=cfg.training.num_epochs,
        accelerator=cfg.training.accelerator,
        devices=devices,
        num_nodes=cfg.training.num_nodes,
        strategy=cfg.training.strategy,
        precision=cfg.training.precision,
        accumulate_grad_batches=accumulate_grad_batches,
        gradient_clip_val=cfg.training.gradient_clip_val,
        callbacks=callbacks,
        logger=logger,
        log_every_n_steps=cfg.training.log_every_n_steps,
        val_check_interval=cfg.training.val_check_interval,
        limit_val_batches=cfg.training.get('limit_val_batches', 1.0),
        num_sanity_val_steps=cfg.training.num_sanity_val_steps,
        enable_checkpointing=True,
        deterministic=cfg.training.get('deterministic', False),
        enable_progress_bar=cfg.training.get('enable_progress_bar', True),
    )

    # Check for resume checkpoint
    ckpt_path = None
    if cfg.resume.get('checkpoint_path'):
        ckpt_path = cfg.resume.checkpoint_path
        # Resolve relative paths from project root (before Hydra's chdir)
        if not os.path.isabs(ckpt_path):
            ckpt_path = os.path.join(hydra.utils.get_original_cwd(), ckpt_path)
        if not os.path.exists(ckpt_path):
            raise ValueError(f"Checkpoint not found: {ckpt_path}")
        print(f"\n=== Resuming from checkpoint: {ckpt_path} ===")

    # Train model
    print("\n=== Starting training ===")
    trainer.fit(model, data_module, ckpt_path=ckpt_path)

    # Save final model
    final_model_path = "final_model"
    os.makedirs(final_model_path, exist_ok=True)

    # Save model weights
    torch.save(model.model.state_dict(), os.path.join(final_model_path, "pytorch_model.bin"))

    # Save config
    model.model.config.save_pretrained(final_model_path)

    # Save tokenizer
    data_module.tokenizer.save_pretrained(final_model_path)

    print(f"\nTraining completed! Model saved to {final_model_path}")

    # Log best checkpoint path
    best_model_path = trainer.checkpoint_callback.best_model_path
    print(f"Best checkpoint: {best_model_path}")

    # Close W&B run if enabled
    if cfg.wandb.enabled:
        wandb.finish()


if __name__ == "__main__":
    main()
