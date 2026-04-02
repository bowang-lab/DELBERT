"""
PyTorch Lightning module for DELBERT classification.

Supports both:
- Training from scratch (random initialization)
- Finetuning from a pretrained MLM checkpoint
"""

import math
import os
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from transformers import get_cosine_schedule_with_warmup
from typing import Dict, Any, Optional, List, Union
from sklearn.metrics import roc_auc_score, average_precision_score
import numpy as np
import wandb

try:
    from rdkit.ML.Scoring.Scoring import CalcBEDROC
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False

from .delbert_model import DELBERTConfig, DELBERTForSequenceClassification
from .finetuning_strategies import get_finetuning_strategy, FinetuningStrategy


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance.

    From "Focal Loss for Dense Object Detection" (https://arxiv.org/abs/1708.02002)

    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    Args:
        alpha: Weighting factor for each class. Can be a float (for positive class weight)
               or a list of weights for each class. Default: None (no weighting)
        gamma: Focusing parameter to reduce loss for well-classified examples. Default: 2.0
        reduction: 'none', 'mean', or 'sum'. Default: 'mean'
    """

    def __init__(
        self,
        alpha: Optional[float] = None,
        gamma: float = 2.0,
        reduction: str = 'mean'
    ):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs: Logits of shape (N, C) where C is number of classes
            targets: Ground truth labels of shape (N,)
        """
        # Compute softmax probabilities
        p = F.softmax(inputs, dim=-1)

        # Get probability of true class
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        p_t = p.gather(1, targets.unsqueeze(1)).squeeze(1)

        # Compute focal weight: (1 - p_t)^gamma
        focal_weight = (1 - p_t) ** self.gamma

        # Apply focal weight
        focal_loss = focal_weight * ce_loss

        # Apply alpha weighting if specified
        if self.alpha is not None:
            # alpha for positive class, (1-alpha) for negative class
            alpha_t = torch.where(targets == 1, self.alpha, 1 - self.alpha)
            focal_loss = alpha_t * focal_loss

        # Apply reduction
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class ClassificationModel(pl.LightningModule):
    """
    Lightning module for DELBERT classification.

    Supports two initialization modes:
    1. From pretrained checkpoint: Load encoder weights from MLM pretraining
    2. From scratch: Random initialization with specified architecture

    When training from scratch, you must provide vocab_size and optionally model_config.
    """

    def __init__(
        self,
        # Model initialization - one of these approaches:
        pretrained_checkpoint: Optional[str] = None,  # Path to pretrained MLM checkpoint
        vocab_size: Optional[int] = None,  # Required if no pretrained checkpoint
        model_config: Optional[Dict[str, Any]] = None,  # Architecture config for random init
        # Classification settings
        num_labels: int = 2,
        classifier_dropout: float = 0.1,
        classifier_pooling: str = 'cls',
        # Training settings
        learning_rate: float = 1e-5,
        warmup_steps: Optional[int] = None,
        warmup_ratio: float = 0.1,
        weight_decay: float = 0.01,
        # Loss settings
        loss_type: str = 'ce',
        pos_class_weight: Optional[float] = None,
        focal_gamma: float = 2.0,
        focal_alpha: Optional[float] = None,
        # Finetuning strategy (only applies when using pretrained checkpoint)
        finetuning_strategy: str = 'full',
        strategy_params: Optional[Dict[str, Any]] = None,
        # Internal flags for checkpoint loading - DO NOT SET MANUALLY
        _skip_pretrained_mlm_loading: bool = False,
        _model_config_dict: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize classification model.

        Args:
            pretrained_checkpoint: Path to pretrained MLM checkpoint. If None, trains from scratch.
            vocab_size: Vocabulary size (required if pretrained_checkpoint is None)
            model_config: Model architecture config dict for random initialization. Keys:
                - hidden_size (default: 768)
                - num_hidden_layers (default: 12)
                - num_attention_heads (default: 12)
                - intermediate_size (default: 1152)
                - max_position_embeddings (default: 1024)
                - global_rope_theta (default: 160000.0)
                - local_attention (default: 128)
                - hidden_dropout_prob (default: 0.1)
                - attention_probs_dropout_prob (default: 0.1)
            num_labels: Number of classification labels
            classifier_dropout: Dropout rate for classification head
            classifier_pooling: Pooling method for classification ('cls' or 'mean')
            learning_rate: Learning rate
            warmup_steps: Fixed number of warmup steps (overrides warmup_ratio if set)
            warmup_ratio: Ratio of total steps for warmup (default: 0.1)
            weight_decay: Weight decay for AdamW
            loss_type: Loss function type - 'ce' (default), 'weighted_ce', or 'focal'
            pos_class_weight: Weight for positive class (for weighted_ce: num_neg/num_pos)
            focal_gamma: Focusing parameter for focal loss (default: 2.0)
            focal_alpha: Alpha for focal loss (default: computed from pos_class_weight if set)
            finetuning_strategy: Strategy name - 'full', 'frozen', 'partial', or 'lora'
                Only applies when pretrained_checkpoint is provided.
            strategy_params: Dict of strategy-specific parameters
            _skip_pretrained_mlm_loading: Internal flag - when True, skip loading MLM checkpoint
                (used by load_from_checkpoint to avoid reloading MLM weights)
            _model_config_dict: Internal - model config dict for architecture reconstruction
                (used by load_from_checkpoint when MLM checkpoint is unavailable)
        """
        super().__init__()
        self.save_hyperparameters(ignore=['_skip_pretrained_mlm_loading', '_model_config_dict'])

        # Determine initialization mode
        self.from_pretrained = pretrained_checkpoint is not None

        if self.from_pretrained:
            if _skip_pretrained_mlm_loading:
                # Loading from a saved classifier checkpoint - create model architecture only
                # The actual weights will be loaded by Lightning after __init__
                if _model_config_dict is not None:
                    # Use config dict passed from load_from_checkpoint
                    config = DELBERTConfig(**_model_config_dict)
                    config.classifier_dropout = classifier_dropout
                    config.classifier_pooling = classifier_pooling
                    self.model = DELBERTForSequenceClassification(config, num_labels=num_labels)
                    print(f"Loading classifier checkpoint (skipping MLM weight loading)")
                else:
                    raise ValueError(
                        "Cannot load classifier checkpoint: model config not found. "
                        "This may be an old checkpoint that doesn't include model_config. "
                        "Try using the original MLM checkpoint path if available."
                    )
            else:
                # Normal training: Load encoder from pretrained MLM checkpoint
                self.model = DELBERTForSequenceClassification.from_pretrained_mlm(
                    pretrained_checkpoint,
                    num_labels=num_labels,
                    config=None  # Load config from checkpoint
                )
                print(f"Loaded pretrained model from: {pretrained_checkpoint}")
        else:
            # Initialize from scratch
            if vocab_size is None:
                raise ValueError(
                    "vocab_size is required when training from scratch (no pretrained_checkpoint). "
                    "You can get this from data_module.vocab_size after setup()."
                )

            # Build model config
            # STRICT: Require model_config when training from scratch
            # Don't silently use defaults - user should explicitly specify architecture
            if model_config is None or len(model_config) == 0:
                raise ValueError(
                    "model_config is required when training from scratch (no pretrained_checkpoint). "
                    "Please provide model architecture settings (hidden_size, num_hidden_layers, etc.) "
                    "in the config. Without this, the model would silently use default values "
                    "which may not match your intended architecture."
                )
            config = DELBERTConfig(
                vocab_size=vocab_size,
                hidden_size=model_config.get('hidden_size', 768),
                num_hidden_layers=model_config.get('num_hidden_layers', 12),
                num_attention_heads=model_config.get('num_attention_heads', 12),
                intermediate_size=model_config.get('intermediate_size', 1152),
                max_position_embeddings=model_config.get('max_position_embeddings', 1024),
                global_rope_theta=model_config.get('global_rope_theta', 160000.0),
                local_attention=model_config.get('local_attention', 128),
                hidden_dropout_prob=model_config.get('hidden_dropout_prob', 0.1),
                attention_probs_dropout_prob=model_config.get('attention_probs_dropout_prob', 0.1),
                classifier_dropout=classifier_dropout,
                classifier_pooling=classifier_pooling,
                use_segment_attention=model_config.get('use_segment_attention', False),
                segment_position_encoding=model_config.get('segment_position_encoding', 'absolute'),
                use_segment_embeddings=model_config.get('use_segment_embeddings', False),
                num_segment_types=model_config.get('num_segment_types', 8),
            )

            self.model = DELBERTForSequenceClassification(config, num_labels=num_labels)

        # Override classifier settings if specified (for pretrained models)
        if self.from_pretrained:
            config_changed = False
            if classifier_dropout != self.model.config.classifier_dropout:
                self.model.config.classifier_dropout = classifier_dropout
                config_changed = True
            if classifier_pooling != self.model.config.classifier_pooling:
                self.model.config.classifier_pooling = classifier_pooling
                config_changed = True

            # Recreate classifier head if config changed
            if config_changed:
                self.model.classifier = nn.Linear(
                    self.model.config.hidden_size,
                    num_labels
                )
                self.model.dropout = nn.Dropout(classifier_dropout)

        # Print model info
        print(f"  Hidden size: {self.model.config.hidden_size}")
        print(f"  Num layers: {self.model.config.num_hidden_layers}")
        print(f"  Num labels: {num_labels}")
        print(f"  Classifier pooling: {classifier_pooling}")

        # Training params
        self.learning_rate = learning_rate
        self.warmup_steps = warmup_steps
        self.warmup_ratio = warmup_ratio
        self.weight_decay = weight_decay
        self.num_labels = num_labels

        # Setup loss function based on loss_type
        self.loss_type = loss_type
        self.loss_fn = self._setup_loss(loss_type, pos_class_weight, focal_gamma, focal_alpha)

        # Setup and apply finetuning strategy (only relevant for pretrained models)
        strategy_params = strategy_params or {}
        self._finetuning_strategy = get_finetuning_strategy(finetuning_strategy, **strategy_params)

        if self.from_pretrained:
            self._finetuning_strategy.apply(self.model)
        else:
            # For random init, always use full training (all params trainable)
            if finetuning_strategy != 'full':
                print(f"  Note: finetuning_strategy='{finetuning_strategy}' ignored for random init (using 'full')")
            full_strategy = get_finetuning_strategy('full')
            full_strategy.apply(self.model)

        # Metrics storage for epoch-level computation
        self.validation_outputs = []
        self.test_outputs = []

    def _setup_loss(
        self,
        loss_type: str,
        pos_class_weight: Optional[float],
        focal_gamma: float,
        focal_alpha: Optional[float]
    ) -> Optional[nn.Module]:
        """
        Setup the loss function based on configuration.

        Args:
            loss_type: 'ce', 'weighted_ce', or 'focal'
            pos_class_weight: Weight for positive class
            focal_gamma: Gamma for focal loss
            focal_alpha: Alpha for focal loss

        Returns:
            Loss function module or None (to use model's built-in loss)
        """
        if loss_type == 'ce':
            print("  Loss: CrossEntropy (default)")
            return None  # Use model's built-in loss

        elif loss_type == 'weighted_ce':
            if pos_class_weight is None:
                raise ValueError("pos_class_weight required for weighted_ce loss")
            class_weights = torch.tensor([1.0, pos_class_weight])
            self.register_buffer('class_weights', class_weights)
            print(f"  Loss: Weighted CrossEntropy [1.0, {pos_class_weight}]")
            return nn.CrossEntropyLoss(weight=class_weights)

        elif loss_type == 'focal':
            # Compute alpha from pos_class_weight if not explicitly set
            # alpha = probability of positive class in balanced scenario (must be 0-1)
            if focal_alpha is not None:
                alpha = focal_alpha
                if not 0 < alpha < 1:
                    raise ValueError(f"focal_alpha must be between 0 and 1, got {alpha}. "
                                     "Use pos_class_weight for class weight ratios.")
            elif pos_class_weight is not None:
                # Convert weight ratio to alpha: alpha = weight / (1 + weight)
                # e.g., weight=15 -> alpha=0.94 (positive class gets 94% weight)
                alpha = pos_class_weight / (1.0 + pos_class_weight)
            else:
                alpha = 0.25  # Default from RetinaNet paper
            print(f"  Loss: Focal Loss (gamma={focal_gamma}, alpha={alpha:.3f})")
            return FocalLoss(alpha=alpha, gamma=focal_gamma)

        else:
            raise ValueError(f"Unknown loss_type: {loss_type}. Choose from 'ce', 'weighted_ce', 'focal'")

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
        labels = batch.get('labels')

        if self.loss_fn is not None:
            # Use custom weighted loss
            outputs = self.forward(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'],
                segment_ids=batch.get('segment_ids'),
                labels=None  # Don't compute built-in loss
            )
            loss = self.loss_fn(outputs['logits'], labels)
        else:
            # Use model's built-in loss
            outputs = self.forward(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'],
                segment_ids=batch.get('segment_ids'),
                labels=labels
            )
            loss = outputs['loss']

        # Log metrics (detach to prevent memory leak with on_epoch=True)
        self.log('train_loss', loss.detach(), on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)

        # Log to W&B (only rank 0 to avoid duplicate/noisy logs in DDP)
        if self.logger and self.trainer.is_global_zero:
            self.logger.log_metrics({
                'train/loss': loss.item(),
                'train/learning_rate': self.optimizers().param_groups[0]['lr']
            })

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
        labels = batch['labels']

        if self.loss_fn is not None:
            # Use custom weighted loss
            outputs = self.forward(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'],
                segment_ids=batch.get('segment_ids'),
                labels=None
            )
            loss = self.loss_fn(outputs['logits'], labels)
        else:
            # Use model's built-in loss
            outputs = self.forward(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'],
                segment_ids=batch.get('segment_ids'),
                labels=labels
            )
            loss = outputs['loss']

        logits = outputs['logits']

        # Store for epoch-level metrics
        self.validation_outputs.append({
            'loss': loss,
            'logits': logits,
            'labels': labels
        })

        # Log loss
        self.log('val_loss', loss.detach(), on_epoch=True, prog_bar=True, sync_dist=True)

        return loss

    def on_validation_epoch_end(self):
        """Compute epoch-level validation metrics.

        IMPORTANT: In DDP, each GPU only sees a shard of the validation data.
        We must gather all predictions across GPUs before computing rank-based
        metrics like ROC-AUC and PR-AUC, because:
            mean(local_AUCs) ≠ global_AUC

        The global metric is computed on rank 0 and broadcast to all ranks.
        """
        if not self.validation_outputs:
            return

        # Concatenate local outputs from this GPU's shard
        local_logits = torch.cat([x['logits'] for x in self.validation_outputs], dim=0)
        local_labels = torch.cat([x['labels'] for x in self.validation_outputs], dim=0)

        # Get true validation dataset size for DistributedSampler padding removal
        val_dataset_size = self._get_dataloader_dataset_size(self.trainer.val_dataloaders)

        # Gather from all GPUs with proper handling of DistributedSampler padding
        all_logits, all_labels = self._gather_ddp_outputs(
            local_logits, local_labels, dataset_size=val_dataset_size
        )

        # Move to CPU for sklearn metrics
        all_probs = F.softmax(all_logits, dim=-1).float().cpu().numpy()
        all_preds = torch.argmax(all_logits, dim=-1).cpu().numpy()
        all_labels_np = all_labels.cpu().numpy()

        # Calculate metrics on the GLOBAL validation set
        metrics = self._calculate_metrics(all_labels_np, all_preds, all_probs)

        # Log metrics - use rank_zero_only since we computed global metrics
        # No sync_dist needed because all ranks have the same global values
        for key, value in metrics.items():
            self.log(f'val_{key}', value, on_epoch=True, rank_zero_only=True)

        # Log to W&B (only on rank 0)
        if self.logger and self.trainer.is_global_zero:
            wandb_metrics = {f'val/{k}': v for k, v in metrics.items()}
            wandb_metrics['epoch'] = self.current_epoch
            self.logger.log_metrics(wandb_metrics)

        # Clear stored outputs to free memory
        self.validation_outputs.clear()
        del local_logits, local_labels, all_logits, all_labels, all_probs, all_preds

    def test_step(self, batch, batch_idx):
        """Test step."""
        labels = batch['labels']

        if self.loss_fn is not None:
            # Use custom weighted loss
            outputs = self.forward(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'],
                segment_ids=batch.get('segment_ids'),
                labels=None
            )
            loss = self.loss_fn(outputs['logits'], labels)
        else:
            # Use model's built-in loss
            outputs = self.forward(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'],
                segment_ids=batch.get('segment_ids'),
                labels=labels
            )
            loss = outputs['loss']

        logits = outputs['logits']

        # Store for epoch-level metrics
        self.test_outputs.append({
            'loss': loss,
            'logits': logits,
            'labels': labels
        })

        return loss

    def on_test_epoch_end(self):
        """Compute epoch-level test metrics.

        Same DDP-aware gathering as validation - see on_validation_epoch_end docstring.
        """
        if not self.test_outputs:
            return

        # Concatenate local outputs from this GPU's shard
        local_logits = torch.cat([x['logits'] for x in self.test_outputs], dim=0)
        local_labels = torch.cat([x['labels'] for x in self.test_outputs], dim=0)

        # Get true test dataset size for DistributedSampler padding removal
        test_dataset_size = self._get_dataloader_dataset_size(self.trainer.test_dataloaders)

        # Gather from all GPUs with proper handling of DistributedSampler padding
        all_logits, all_labels = self._gather_ddp_outputs(
            local_logits, local_labels, dataset_size=test_dataset_size
        )

        # Move to CPU for sklearn metrics
        all_probs = F.softmax(all_logits, dim=-1).float().cpu().numpy()
        all_preds = torch.argmax(all_logits, dim=-1).cpu().numpy()
        all_labels_np = all_labels.cpu().numpy()

        # Calculate metrics on the GLOBAL test set
        metrics = self._calculate_metrics(all_labels_np, all_preds, all_probs)

        # Log metrics - use rank_zero_only since we computed global metrics
        for key, value in metrics.items():
            self.log(f'test_{key}', value, on_epoch=True, rank_zero_only=True)

        # Log to W&B (only on rank 0)
        if self.logger and self.trainer.is_global_zero:
            wandb_metrics = {f'test/{k}': v for k, v in metrics.items()}
            self.logger.log_metrics(wandb_metrics)

        # Clear stored outputs to free memory
        self.test_outputs.clear()
        del local_logits, local_labels, all_logits, all_labels, all_probs, all_preds

    def _calculate_metrics(self, labels, preds, probs):
        """Calculate ranking-based classification metrics (no threshold dependency)."""
        metrics = {}

        if self.num_labels == 2:
            # Binary classification - ranking metrics only
            pos_probs = probs[:, 1]  # Probability of positive class

            # Top-K Precision and Enrichment Factor metrics
            base_rate = np.sum(labels) / len(labels) if len(labels) > 0 else 0.0

            if len(labels) >= 100:
                sorted_indices = np.argsort(pos_probs)[::-1]
                top_100_labels = labels[sorted_indices[:100]]
                precision_100 = float(np.mean(top_100_labels))
                metrics['precision_at_100'] = precision_100
                metrics['enrich_at_100'] = precision_100 / base_rate if base_rate > 0 else 0.0

                if len(labels) >= 500:
                    top_500_labels = labels[sorted_indices[:500]]
                    precision_500 = float(np.mean(top_500_labels))
                    metrics['precision_at_500'] = precision_500
                    metrics['enrich_at_500'] = precision_500 / base_rate if base_rate > 0 else 0.0

            # ROC-AUC and PR-AUC
            try:
                if len(np.unique(labels)) > 1:
                    metrics['roc_auc'] = roc_auc_score(labels, pos_probs)
                    metrics['pr_auc'] = average_precision_score(labels, pos_probs)
                else:
                    # Single class in validation
                    metrics['roc_auc'] = 0.5
                    metrics['pr_auc'] = float(np.mean(labels))
            except Exception as e:
                print(f"Warning: Error calculating metrics: {e}")
                metrics['roc_auc'] = 0.0
                metrics['pr_auc'] = 0.0

            # BEDROC (requires RDKit)
            if RDKIT_AVAILABLE:
                try:
                    if len(np.unique(labels)) > 1 and np.sum(labels) > 0:
                        # Sort by predicted probability descending
                        order = np.argsort(-pos_probs)
                        sorted_labels = labels[order]
                        sorted_probs = pos_probs[order]
                        # Format for RDKit: list of (score, activity) tuples
                        scores = list(zip(sorted_probs, sorted_labels))
                        # alpha=85.0 emphasizes early recognition (standard for virtual screening)
                        metrics['bedroc'] = float(CalcBEDROC(scores, col=1, alpha=85.0))
                    else:
                        metrics['bedroc'] = 0.0
                except Exception:
                    metrics['bedroc'] = 0.0

        else:
            # Multi-class classification - ROC-AUC only
            try:
                if len(np.unique(labels)) > 1:
                    metrics['roc_auc'] = roc_auc_score(
                        labels, probs, multi_class='ovr', average='weighted'
                    )
                else:
                    metrics['roc_auc'] = 0.5
            except (ValueError, TypeError):
                metrics['roc_auc'] = 0.0

        return metrics

    @staticmethod
    def _get_dataloader_dataset_size(dataloaders) -> Optional[int]:
        """Extract dataset size from a dataloader or list of dataloaders."""
        if dataloaders is None:
            return None
        # Lightning may wrap dataloaders in a list or CombinedLoader
        dl = dataloaders[0] if isinstance(dataloaders, (list, tuple)) else dataloaders
        if hasattr(dl, 'dataset'):
            return len(dl.dataset)
        return None

    def _gather_ddp_outputs(
        self,
        local_logits: torch.Tensor,
        local_labels: torch.Tensor,
        dataset_size: Optional[int] = None
    ) -> tuple:
        """
        Gather outputs from all DDP ranks, removing DistributedSampler padding.

        When dataset size is not evenly divisible by world_size, DistributedSampler
        pads with duplicate samples so each rank gets ceil(N/W) samples. The last
        (ceil(N/W)*W - N) ranks have 1 padding sample at the end of their shard.

        Args:
            local_logits: Logits from this rank, shape (local_size, num_classes)
            local_labels: Labels from this rank, shape (local_size,)
            dataset_size: True dataset size (before padding). Required for proper
                         deduplication in multi-GPU. If None, no trimming is done.

        Returns:
            Tuple of (all_logits, all_labels) with padding removed
        """
        if self.trainer.world_size == 1:
            return local_logits, local_labels

        # Gather logits and labels from all ranks
        # gathered shape: (world_size, local_size, ...)
        gathered_logits = self.all_gather(local_logits)
        gathered_labels = self.all_gather(local_labels)

        world_size = self.trainer.world_size
        samples_per_rank = local_logits.shape[0]  # = ceil(N / W), same on all ranks

        if dataset_size is not None:
            # Compute how many ranks have padding.
            # DistributedSampler pads the dataset to total_size = ceil(N/W) * W.
            # It then assigns indices round-robin: rank r gets global[r::W].
            # The padding indices are at the END of the global list, so the
            # last P ranks (by index) each have 1 extra (duplicate) sample.
            padding = samples_per_rank * world_size - dataset_size
        else:
            padding = 0

        valid_logits = []
        valid_labels = []

        for rank_idx in range(world_size):
            if rank_idx >= world_size - padding:
                # This rank has 1 padding sample at the end — trim it
                valid = samples_per_rank - 1
            else:
                valid = samples_per_rank
            valid_logits.append(gathered_logits[rank_idx, :valid, :])
            valid_labels.append(gathered_labels[rank_idx, :valid])

        all_logits = torch.cat(valid_logits, dim=0)
        all_labels = torch.cat(valid_labels, dim=0)

        return all_logits, all_labels

    def configure_optimizers(self):
        """Configure optimizer and scheduler."""
        # Separate parameters for weight decay (only include trainable params)
        no_decay = ["bias", "LayerNorm.weight", "layer_norm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters()
                          if p.requires_grad and not any(nd in n for nd in no_decay)],
                "weight_decay": self.weight_decay,
            },
            {
                "params": [p for n, p in self.model.named_parameters()
                          if p.requires_grad and any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]

        # Filter out empty param groups
        optimizer_grouped_parameters = [g for g in optimizer_grouped_parameters if len(g['params']) > 0]

        optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=self.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-8
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

    def predict_step(self, batch, batch_idx):
        """Prediction step for inference."""
        outputs = self.forward(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask'],
            segment_ids=batch.get('segment_ids')
        )

        logits = outputs['logits']
        probs = F.softmax(logits, dim=-1)
        preds = torch.argmax(logits, dim=-1)

        return {
            'predictions': preds,
            'probabilities': probs,
            'compound_ids': batch.get('compound_id', None)
        }

    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        """Save model config to checkpoint for architecture reconstruction.

        This allows load_from_checkpoint to reconstruct the model architecture
        without needing access to the original MLM checkpoint.
        """
        checkpoint['model_config'] = self.model.config.to_dict()

    @classmethod
    def load_from_checkpoint(
        cls,
        checkpoint_path: str,
        map_location: Optional[Any] = None,
        hparams_file: Optional[str] = None,
        strict: bool = True,
        **kwargs
    ):
        """Load classifier checkpoint without re-loading MLM weights.

        This override ensures that when loading a saved classifier checkpoint,
        we don't try to reload the original MLM checkpoint (which would overwrite
        the trained classifier weights and fail if the MLM file was moved/deleted).

        Args:
            checkpoint_path: Path to the classifier checkpoint file
            map_location: Device mapping for loading
            hparams_file: Optional path to hyperparameters file
            strict: Whether to strictly enforce state_dict key matching
            **kwargs: Additional keyword arguments passed to parent

        Returns:
            Loaded ClassificationModel with trained weights
        """
        # Load checkpoint to extract model config
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        model_config_dict = checkpoint.get('model_config')

        if model_config_dict is None:
            # Backward compatibility: try to load config from MLM checkpoint
            hparams = checkpoint.get('hyper_parameters', {})
            mlm_path = hparams.get('pretrained_checkpoint')
            if mlm_path and os.path.exists(mlm_path):
                print(f"Warning: Checkpoint missing model_config, falling back to MLM checkpoint: {mlm_path}")
                mlm_ckpt = torch.load(mlm_path, map_location='cpu')
                if 'hyper_parameters' in mlm_ckpt:
                    # It's a Lightning checkpoint
                    model_config_dict = mlm_ckpt.get('model_config')
                elif 'config' in mlm_ckpt:
                    # Direct model config
                    model_config_dict = mlm_ckpt['config']

            if model_config_dict is None:
                raise ValueError(
                    f"Cannot load checkpoint: model_config not found in checkpoint "
                    f"and original MLM checkpoint is unavailable or doesn't contain config. "
                    f"Checkpoint path: {checkpoint_path}"
                )

        # Pass skip flag and model config to __init__
        return super().load_from_checkpoint(
            checkpoint_path,
            map_location=map_location,
            hparams_file=hparams_file,
            strict=strict,
            _skip_pretrained_mlm_loading=True,
            _model_config_dict=model_config_dict,
            **kwargs
        )
