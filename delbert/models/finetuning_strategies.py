"""
Finetuning strategies for DELBERT.

This module provides modular finetuning strategies that can be applied to the model:

Implemented:
- FullFinetuning: Train all parameters (default)
- FrozenEncoder: Freeze encoder, only train classifier head (linear probing)
- LoRAFinetuning: Use LoRA adapters for parameter-efficient finetuning (requires peft)

Placeholders (not yet implemented):
- PartialUnfreezing: Freeze lower layers, unfreeze top N layers

Usage:
    strategy = get_finetuning_strategy('frozen', **kwargs)
    strategy.apply(model)
"""

from abc import ABC, abstractmethod
from typing import Optional, List, Dict, Any
import torch.nn as nn


class FinetuningStrategy(ABC):
    """
    Abstract base class for finetuning strategies.

    Each strategy defines how to configure model parameters for training:
    which parameters to freeze/unfreeze, and any additional setup needed.
    """

    @abstractmethod
    def apply(self, model: nn.Module) -> None:
        """
        Apply this finetuning strategy to the model.

        Args:
            model: The model to configure (DELBERTForSequenceClassification)
        """
        pass

    @abstractmethod
    def get_param_groups(self, model: nn.Module, base_lr: float) -> List[Dict[str, Any]]:
        """
        Get optimizer parameter groups with appropriate learning rates.

        Args:
            model: The model
            base_lr: Base learning rate

        Returns:
            List of parameter group dicts for optimizer
        """
        pass

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


class FullFinetuning(FinetuningStrategy):
    """
    Full finetuning - train all model parameters.

    This is the standard finetuning approach where all parameters
    (encoder + classifier) are updated during training.
    """

    def __init__(self, encoder_lr_multiplier: float = 1.0):
        """
        Args:
            encoder_lr_multiplier: Multiplier for encoder learning rate relative to classifier.
                                   Use < 1.0 for discriminative fine-tuning (e.g., 0.1)
        """
        self.encoder_lr_multiplier = encoder_lr_multiplier

    def apply(self, model: nn.Module) -> None:
        """Enable gradients for all parameters."""
        for param in model.parameters():
            param.requires_grad = True

        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        print(f"  Strategy: Full Finetuning")
        print(f"  Trainable parameters: {trainable:,} / {total:,} ({100*trainable/total:.1f}%)")
        if self.encoder_lr_multiplier != 1.0:
            print(f"  Encoder LR multiplier: {self.encoder_lr_multiplier}")

    def get_param_groups(self, model: nn.Module, base_lr: float) -> List[Dict[str, Any]]:
        """Get param groups, optionally with different LR for encoder."""
        if self.encoder_lr_multiplier == 1.0:
            # Single learning rate for all parameters
            return [{'params': model.parameters(), 'lr': base_lr}]

        # Discriminative fine-tuning: lower LR for encoder
        encoder_params = []
        classifier_params = []

        for name, param in model.named_parameters():
            if 'classifier' in name or 'pooler' in name:
                classifier_params.append(param)
            else:
                encoder_params.append(param)

        return [
            {'params': encoder_params, 'lr': base_lr * self.encoder_lr_multiplier},
            {'params': classifier_params, 'lr': base_lr}
        ]

    def __repr__(self) -> str:
        return f"FullFinetuning(encoder_lr_multiplier={self.encoder_lr_multiplier})"


class FrozenEncoder(FinetuningStrategy):
    """
    Frozen encoder finetuning (linear probing).

    Freezes all encoder parameters and only trains the classification head.
    This is useful for:
    - Quick evaluation of pretrained representations
    - When you have limited training data
    - As a baseline before full finetuning
    """

    def apply(self, model: nn.Module) -> None:
        """Freeze encoder, unfreeze classifier."""
        # First freeze everything
        for param in model.parameters():
            param.requires_grad = False

        # Unfreeze classifier head
        # Our custom wrapper models have: model.classifier
        inner_model = model.model if hasattr(model, 'model') else model

        # Check for our custom wrapper models (have .model encoder and .classifier)
        is_custom_wrapper = (
            hasattr(inner_model, 'model') and
            hasattr(inner_model, 'classifier') and
            hasattr(inner_model, 'dropout')
        )

        if is_custom_wrapper:
            # Unfreeze the classifier for our custom wrapper models
            for param in inner_model.classifier.parameters():
                param.requires_grad = True
        else:
            # Standard HF model - look for classifier/head/pooler in names
            for name, param in inner_model.named_parameters():
                if 'classifier' in name or 'head' in name or 'pooler' in name:
                    param.requires_grad = True

        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        print(f"  Strategy: Frozen Encoder (Linear Probing)")
        print(f"  Trainable parameters: {trainable:,} / {total:,} ({100*trainable/total:.1f}%)")

    def get_param_groups(self, model: nn.Module, base_lr: float) -> List[Dict[str, Any]]:
        """Get only trainable (classifier) parameters."""
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        return [{'params': trainable_params, 'lr': base_lr}]


class PartialUnfreezing(FinetuningStrategy):
    """
    Partial unfreezing - freeze lower layers, unfreeze top N layers.

    NOT YET IMPLEMENTED - placeholder for future development.
    """

    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def apply(self, model: nn.Module) -> None:
        raise NotImplementedError(
            "PartialUnfreezing strategy is not yet implemented. "
            "Use 'full' or 'frozen' strategy instead."
        )

    def get_param_groups(self, model: nn.Module, base_lr: float) -> List[Dict[str, Any]]:
        raise NotImplementedError("PartialUnfreezing not implemented")

    def __repr__(self) -> str:
        return "PartialUnfreezing(NOT IMPLEMENTED)"


class LoRAFinetuning(FinetuningStrategy):
    """
    LoRA (Low-Rank Adaptation) finetuning.

    Adds low-rank adapters to specific modules while freezing the base model.
    Requires: peft library (pip install peft)
    """

    def __init__(
        self,
        r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.1,
        target_modules: Optional[List[str]] = None,
        **kwargs
    ):
        """
        Args:
            r: LoRA rank (lower = fewer params, higher = more expressive)
            lora_alpha: LoRA scaling factor (typically 2*r)
            lora_dropout: Dropout for LoRA layers
            target_modules: List of module names to apply LoRA to.
                           Default: ["Wqkv"] for ModernBERT attention.
        """
        self.r = r
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.target_modules = target_modules or ["Wqkv"]  # Default for ModernBERT

    def apply(self, model: nn.Module) -> None:
        """Apply LoRA adapters to the model."""
        try:
            from peft import get_peft_model, LoraConfig, TaskType
        except ImportError:
            raise ImportError(
                "LoRA finetuning requires the 'peft' library. "
                "Install with: pip install peft"
            )

        # Get the underlying model
        # model is DELBERTForSequenceClassification
        # model.model is one of our custom wrapper classes
        inner_model = model.model

        # Check if this is one of our custom wrapper models
        # Both SegmentAttentionModernBertForSequenceClassification and
        # ModernBertForSequenceClassificationWithSegmentEmbeds have .model (encoder) and .classifier
        is_custom_wrapper = (
            hasattr(inner_model, 'model') and
            hasattr(inner_model, 'classifier') and
            hasattr(inner_model, 'dropout')
        )

        if is_custom_wrapper:
            # For our custom wrapper models, apply LoRA to just the encoder
            # inner_model.model is the encoder (SegmentAttentionModernBertModel or ModernBertModelWithSegmentEmbeds)
            encoder = inner_model.model

            # Configure LoRA for the encoder (FEATURE_EXTRACTION since it's just the encoder)
            lora_config = LoraConfig(
                task_type=TaskType.FEATURE_EXTRACTION,
                r=self.r,
                lora_alpha=self.lora_alpha,
                lora_dropout=self.lora_dropout,
                target_modules=self.target_modules,
                bias="none",
            )

            # Apply LoRA to encoder only
            peft_encoder = get_peft_model(encoder, lora_config)
            inner_model.model = peft_encoder

            # Ensure classification head is trainable
            for param in inner_model.classifier.parameters():
                param.requires_grad = True

            print(f"  Strategy: LoRA Finetuning")
        else:
            # Standard HuggingFace model - apply LoRA to the whole classification model
            lora_config = LoraConfig(
                task_type=TaskType.SEQ_CLS,
                r=self.r,
                lora_alpha=self.lora_alpha,
                lora_dropout=self.lora_dropout,
                target_modules=self.target_modules,
                bias="none",
            )

            # Apply LoRA - this wraps the model
            peft_model = get_peft_model(inner_model, lora_config)
            model.model = peft_model

            print(f"  Strategy: LoRA Finetuning")

        # Log trainable parameters
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        print(f"  LoRA config: r={self.r}, alpha={self.lora_alpha}, dropout={self.lora_dropout}")
        print(f"  Target modules: {self.target_modules}")
        print(f"  Trainable parameters: {trainable_params:,} / {total_params:,} ({100*trainable_params/total_params:.2f}%)")

    def get_param_groups(self, model: nn.Module, base_lr: float) -> List[Dict[str, Any]]:
        """Get parameter groups for LoRA - only trainable adapter params."""
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        return [{'params': trainable_params, 'lr': base_lr}]

    def __repr__(self) -> str:
        return f"LoRAFinetuning(r={self.r}, alpha={self.lora_alpha}, targets={self.target_modules})"


# Strategy registry for easy instantiation
STRATEGY_REGISTRY = {
    'full': FullFinetuning,
    'frozen': FrozenEncoder,
    'partial': PartialUnfreezing,
    'lora': LoRAFinetuning,
}


def get_finetuning_strategy(
    strategy_name: str,
    **kwargs
) -> FinetuningStrategy:
    """
    Factory function to create a finetuning strategy.

    Args:
        strategy_name: Name of the strategy ('full', 'frozen', 'partial', 'lora')
        **kwargs: Strategy-specific arguments

    Returns:
        Configured FinetuningStrategy instance

    Example:
        >>> strategy = get_finetuning_strategy('partial', unfreeze_layers=4)
        >>> strategy.apply(model)
    """
    if strategy_name not in STRATEGY_REGISTRY:
        available = ', '.join(STRATEGY_REGISTRY.keys())
        raise ValueError(
            f"Unknown finetuning strategy: '{strategy_name}'. "
            f"Available strategies: {available}"
        )

    strategy_class = STRATEGY_REGISTRY[strategy_name]
    return strategy_class(**kwargs)


def list_strategies() -> List[str]:
    """Return list of available strategy names."""
    return list(STRATEGY_REGISTRY.keys())
