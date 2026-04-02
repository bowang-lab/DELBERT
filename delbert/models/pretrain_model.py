"""
PyTorch Lightning module for DELBERT pretraining (MLM).
"""

import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch.optim import AdamW
from transformers import get_cosine_schedule_with_warmup
from typing import Dict, Any, Optional
import wandb

from .delbert_model import DELBERTConfig, DELBERTForMLM, apply_layer_types


class PretrainModel(pl.LightningModule):
    """
    Lightning module for DELBERT MLM pretraining.
    """
    
    def __init__(
        self,
        config: Dict[str, Any],
        vocab_size: int,
        learning_rate: float = 5e-4,
        warmup_ratio: float = 0.1,
        warmup_steps: Optional[int] = None,
        weight_decay: float = 0.01
    ):
        """
        Initialize pretrain model.

        Args:
            config: Model configuration dictionary
            vocab_size: Vocabulary size from tokenizer
            learning_rate: Learning rate
            warmup_ratio: Ratio of total steps for warmup (default: 0.1)
            warmup_steps: Fixed number of warmup steps (overrides warmup_ratio if set)
            weight_decay: Weight decay for AdamW
        """
        super().__init__()
        self.save_hyperparameters()
        
        # Create model config
        model_config = DELBERTConfig(
            vocab_size=vocab_size,
            hidden_size=config.get('hidden_size', 768),
            num_hidden_layers=config.get('num_hidden_layers', 12),
            num_attention_heads=config.get('num_attention_heads', 12),
            intermediate_size=config.get('intermediate_size', 1152),
            max_position_embeddings=config.get('max_position_embeddings', 1024),
            global_rope_theta=config.get('global_rope_theta', 160000.0),
            local_attention=config.get('local_attention', 128),
            global_attn_every_n_layers=config.get('global_attn_every_n_layers', 3),
            hidden_dropout_prob=config.get('hidden_dropout_prob', 0.1),
            attention_probs_dropout_prob=config.get('attention_probs_dropout_prob', 0.1),
            use_segment_attention=config.get('use_segment_attention', False),
            segment_position_encoding=config.get('segment_position_encoding', 'absolute'),
            use_segment_embeddings=config.get('use_segment_embeddings', False),
            num_segment_types=config.get('num_segment_types', 8),
        )
        
        # Initialize model
        self.model = DELBERTForMLM(model_config)

        # Apply custom layer_types if specified
        layer_types = config.get('layer_types')
        if layer_types:
            apply_layer_types(self.model.encoder, layer_types, config.get('local_attention', 128))
            # Store layer_types in model config for checkpoint saving
            self.model.config.layer_types = layer_types

        # Training params
        self.learning_rate = learning_rate
        self.warmup_ratio = warmup_ratio
        self.warmup_steps = warmup_steps
        self.weight_decay = weight_decay

    def forward(self, input_ids, attention_mask=None, segment_ids=None, labels=None):
        """Forward pass."""
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            segment_ids=segment_ids,
            labels=labels
        )

    def training_step(self, batch, batch_idx):
        """Training step."""
        # Validate segment_ids if model expects segment attention
        if getattr(self.model, '_use_segment_attention', False) and batch.get('segment_ids') is None:
            raise ValueError(
                "Model configured for segment attention (use_segment_attention=True) but batch missing 'segment_ids'. "
                "Check that data prep config has 'return_segment_ids: true' and "
                "ProcessedDataModule is passing segment_ids through."
            )
        outputs = self.forward(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask'],
            segment_ids=batch.get('segment_ids'),
            labels=batch.get('labels')  # Labels should be masked, not raw input_ids
        )
        
        loss = outputs['loss']
        
        # Get current learning rate
        lr = self.optimizers().param_groups[0]['lr']
        
        # Log metrics (detach to prevent memory leak with on_epoch=True)
        # self.log() automatically logs to W&B when WandbLogger is configured
        self.log('train_loss', loss.detach(), on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('lr', lr, on_step=True, on_epoch=False, prog_bar=True, sync_dist=True)

        return loss
    
    def validation_step(self, batch, batch_idx):
        """Validation step."""
        # Validate segment_ids if model expects segment attention
        if getattr(self.model, '_use_segment_attention', False) and batch.get('segment_ids') is None:
            raise ValueError(
                "Model configured for segment attention (use_segment_attention=True) but batch missing 'segment_ids'. "
                "Check that data prep config has 'return_segment_ids: true' and "
                "ProcessedDataModule is passing segment_ids through."
            )
        outputs = self.forward(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask'],
            segment_ids=batch.get('segment_ids'),
            labels=batch.get('labels')
        )
        
        loss = outputs['loss']
        
        # Log metrics
        self.log('val_loss', loss.detach(), on_epoch=True, prog_bar=True, sync_dist=True)
        
        return loss
    
    def configure_optimizers(self):
        """Configure optimizer and scheduler."""
        # Separate parameters for weight decay
        no_decay = ["bias", "LayerNorm.weight", "layer_norm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters()
                          if not any(nd in n for nd in no_decay)],
                "weight_decay": self.weight_decay,
            },
            {
                "params": [p for n, p in self.model.named_parameters()
                          if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]

        optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=self.learning_rate,
            betas=(0.9, 0.98),
            eps=1e-6
        )

        # Get total training steps from Lightning (accounts for multi-GPU, accumulation, etc.)
        num_training_steps = self.trainer.estimated_stepping_batches

        # Calculate warmup steps: use fixed value if provided, otherwise use ratio
        warmup_steps = self.warmup_steps if self.warmup_steps is not None else int(self.warmup_ratio * num_training_steps)
        print(f"Training steps: {num_training_steps}, Warmup steps: {warmup_steps}")

        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=num_training_steps,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step", "frequency": 1}
        }
    
    def on_train_epoch_end(self):
        """Log epoch-level metrics."""
        if self.logger:
            metrics = {
                'epoch': self.current_epoch,
                'global_step': self.global_step,
            }
            
            # Get average metrics from logged values
            if hasattr(self.trainer, 'logged_metrics'):
                for key in ['train_loss']:
                    if key in self.trainer.logged_metrics:
                        metrics[f'epoch/{key}'] = self.trainer.logged_metrics[key]
            
            self.logger.log_metrics(metrics)
    
    def on_save_checkpoint(self, checkpoint):
        """Add model config to checkpoint."""
        checkpoint['model_config'] = self.model.config.to_dict()
        return checkpoint