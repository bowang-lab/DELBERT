"""
DELBERT: ModernBERT-based model for molecular fingerprint sequences.

This module provides a base transformer model using ModernBERT architecture
for both pretraining (MLM) and finetuning (classification) tasks.
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Any, Tuple
from transformers import (
    ModernBertConfig,
    ModernBertModel,
    ModernBertForMaskedLM,
    ModernBertForSequenceClassification,
)
from transformers.modeling_outputs import BaseModelOutput


def create_segment_attention_mask(
    segment_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    dtype: torch.dtype
) -> torch.Tensor:
    """
    Create 4D attention mask for segment-restricted (Intra-FP) attention.

    Tokens can only attend to other tokens with the same segment_id.
    Padding tokens (segment_id=0) cannot attend or be attended to.

    Args:
        segment_ids: [batch, seq_len] - segment ID for each token (0 = padding)
        attention_mask: [batch, seq_len] - 1 for valid tokens, 0 for padding
        dtype: torch dtype for output (determines min value for masking)

    Returns:
        [batch, 1, seq_len, seq_len] attention mask
        Values: 0.0 where attention allowed, -inf where blocked
    """
    batch_size, seq_len = segment_ids.shape
    device = segment_ids.device

    # Same segment check: [batch, seq_len, seq_len]
    # segment_ids[:, :, None] == segment_ids[:, None, :]
    same_segment = segment_ids.unsqueeze(2) == segment_ids.unsqueeze(1)

    # Exclude padding: padding tokens (segment_id=0) should not match anything
    # This prevents padding tokens from attending to each other
    not_padding_query = (segment_ids != 0).unsqueeze(2)  # [batch, seq_len, 1]
    not_padding_key = (segment_ids != 0).unsqueeze(1)    # [batch, 1, seq_len]

    # Can attend iff: same segment AND query not padding AND key not padding
    can_attend = same_segment & not_padding_query & not_padding_key

    # Also apply the original attention_mask (handles actual padding positions)
    valid_key = attention_mask.unsqueeze(1).bool()  # [batch, 1, seq_len]
    can_attend = can_attend & valid_key

    # Convert to float mask: 0 = attend, min_value = block
    min_value = torch.finfo(dtype).min
    attn_mask = torch.where(
        can_attend,
        torch.tensor(0.0, dtype=dtype, device=device),
        torch.tensor(min_value, dtype=dtype, device=device)
    )

    # Add head dimension: [batch, 1, seq_len, seq_len]
    return attn_mask.unsqueeze(1)


def compute_position_ids(
    segment_ids: torch.Tensor,
    seq_len: int,
    mode: str,
    device: torch.device
) -> torch.Tensor:
    """
    Compute position_ids based on segment structure and encoding mode.

    Args:
        segment_ids: [batch, seq_len] segment IDs (0 = padding)
        seq_len: sequence length
        mode: 'absolute', 'reset', or 'none'
        device: torch device

    Returns:
        position_ids: [batch, seq_len]
    """
    batch_size = segment_ids.shape[0]

    if mode == 'absolute':
        # Standard: [0, 1, 2, ..., seq_len-1]
        return torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)

    elif mode == 'none':
        # All zeros: no positional information
        return torch.zeros(batch_size, seq_len, dtype=torch.long, device=device)

    elif mode == 'reset':
        # Reset at each segment boundary
        # Example: segment_ids = [1,1,1,2,2,3,3,3]
        #          position_ids = [0,1,2,0,1,0,1,2]

        # Detect segment boundaries (where segment changes)
        segment_changed = torch.cat([
            torch.ones(batch_size, 1, dtype=torch.bool, device=device),
            segment_ids[:, 1:] != segment_ids[:, :-1]
        ], dim=1)

        # Vectorized position computation using cumsum trick
        # Create a running count that resets at segment boundaries
        ones = torch.ones_like(segment_ids, dtype=torch.long)
        # cumsum gives running count, but we need to reset at boundaries
        cumsum = torch.cumsum(ones, dim=1)  # [1, 2, 3, 4, 5, ...]

        # At each boundary, we need to subtract the cumsum value at that point
        # to reset to 0
        boundary_cumsum = cumsum * segment_changed.long()
        # cummax gives the "floor" value to subtract
        reset_values, _ = torch.cummax(boundary_cumsum, dim=1)
        position_ids = cumsum - reset_values

        return position_ids

    else:
        raise ValueError(f"Unknown segment_position_encoding: {mode}")


class DELBERTConfig(ModernBertConfig):
    """
    Configuration class for DELBERT model.
    Extends ModernBertConfig with molecular-specific parameters.
    """

    def __init__(
        self,
        vocab_size: int = 5000,
        hidden_size: int = 768,
        num_hidden_layers: int = 12,
        num_attention_heads: int = 12,
        intermediate_size: int = 1152,
        max_position_embeddings: int = 1024,
        global_rope_theta: float = 160000.0,
        local_attention: int = 128,
        hidden_dropout_prob: float = 0.1,
        attention_probs_dropout_prob: float = 0.1,
        classifier_dropout: float = 0.1,
        classifier_pooling: str = 'cls',  # 'cls' or 'mean'
        pad_token_id: int = 0,
        use_segment_attention: bool = False,  # Enable intra-FP attention
        segment_position_encoding: str = 'absolute',  # 'absolute', 'reset', or 'none'
        use_segment_embeddings: bool = False,  # Add learnable segment type embeddings
        num_segment_types: int = 8,  # Max segment types (7 FP types + 1 padding)
        **kwargs
    ):
        super().__init__(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            intermediate_size=intermediate_size,
            max_position_embeddings=max_position_embeddings,
            global_rope_theta=global_rope_theta,
            local_attention=local_attention,
            hidden_dropout_prob=hidden_dropout_prob,
            attention_probs_dropout_prob=attention_probs_dropout_prob,
            classifier_dropout=classifier_dropout,
            classifier_pooling=classifier_pooling,
            pad_token_id=pad_token_id,
            **kwargs
        )
        self.use_segment_attention = use_segment_attention
        self.segment_position_encoding = segment_position_encoding
        self.use_segment_embeddings = use_segment_embeddings
        self.num_segment_types = num_segment_types


class SegmentEmbeddingLayer(nn.Module):
    """Learnable embeddings for fingerprint segment types."""

    def __init__(self, num_segment_types: int, hidden_size: int):
        super().__init__()
        # padding_idx=0 ensures segment 0 (padding) always gets zeros
        self.segment_embeddings = nn.Embedding(
            num_embeddings=num_segment_types,
            embedding_dim=hidden_size,
            padding_idx=0
        )
        # Initialize with small values (like BERT segment embeddings)
        nn.init.normal_(self.segment_embeddings.weight, mean=0.0, std=0.02)
        with torch.no_grad():
            self.segment_embeddings.weight[0].zero_()

    def forward(self, segment_ids: torch.Tensor) -> torch.Tensor:
        return self.segment_embeddings(segment_ids)


class SegmentAttentionModernBertModel(ModernBertModel):
    """
    ModernBertModel with segment-based attention instead of sliding window.

    When segment_ids is provided, tokens can only attend within their segment.
    When segment_ids is None, falls back to standard behavior.
    """

    def __init__(self, config):
        super().__init__(config)
        # Apply layer_types if present in config
        # This controls which layers use sliding window vs full attention
        layer_types = getattr(config, 'layer_types', None)
        if layer_types:
            self._apply_layer_types(layer_types, config.local_attention)

        # Create segment embedding layer if enabled
        if getattr(config, 'use_segment_embeddings', False):
            self.segment_embedding_layer = SegmentEmbeddingLayer(
                num_segment_types=getattr(config, 'num_segment_types', 8),
                hidden_size=config.hidden_size
            )
            self._use_segment_embeddings = True
        else:
            self.segment_embedding_layer = None
            self._use_segment_embeddings = False

    def _apply_layer_types(self, layer_types: list, local_attention_window: int):
        """
        Apply custom attention patterns to model layers.

        Args:
            layer_types: List of 'full_attention' or 'sliding_attention' per layer
            local_attention_window: Window size for sliding attention
        """
        if len(layer_types) != len(self.layers):
            raise ValueError(
                f"layer_types has {len(layer_types)} entries but model has {len(self.layers)} layers"
            )

        half_window = local_attention_window // 2
        for i, (layer, layer_type) in enumerate(zip(self.layers, layer_types)):
            if layer_type == 'full_attention':
                layer.attn.local_attention = (-1, -1)
            elif layer_type == 'sliding_attention':
                layer.attn.local_attention = (half_window, half_window)
            else:
                raise ValueError(f"Unknown layer_type '{layer_type}' at index {i}. Use 'full_attention' or 'sliding_attention'")

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        segment_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs
    ):
        """
        Forward pass with optional segment-based attention.

        Args:
            segment_ids: [batch, seq_len] segment ID for each token (0 = padding).
                If provided, uses segment-based attention mask.
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if inputs_embeds is not None:
            batch_size, seq_len = inputs_embeds.shape[:2]
        else:
            batch_size, seq_len = input_ids.shape[:2]
        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if attention_mask is None:
            attention_mask = torch.ones((batch_size, seq_len), device=device, dtype=torch.long)

        # Compute position_ids based on segment structure and config
        if position_ids is None:
            if segment_ids is not None:
                position_ids = compute_position_ids(
                    segment_ids=segment_ids,
                    seq_len=seq_len,
                    mode=getattr(self.config, 'segment_position_encoding', 'absolute'),
                    device=device
                )
            else:
                position_ids = torch.arange(seq_len, device=device).unsqueeze(0)

        # Create attention masks
        # STRICT VALIDATION: SegmentAttentionModernBertModel REQUIRES segment_ids
        # Without this check, the model would silently fall back to standard sliding window
        # attention, which is NOT what the user configured and defeats the purpose.
        if segment_ids is None:
            raise ValueError(
                "SegmentAttentionModernBertModel requires segment_ids but received None. "
                "This model is configured for segment-based attention (use_segment_attention=True). "
                "Without segment_ids, the model would silently fall back to standard sliding window attention. "
                "Fix: Ensure data was prepared with 'return_segment_ids: true' and that segment_ids "
                "is passed through the data module and collator."
            )

        # Get the standard masks first (for global attention layers)
        global_attn_mask, _ = self._update_attention_mask(
            attention_mask, output_attentions
        )

        # Create segment-based mask for sliding window layers only
        # Global attention layers will use unrestricted attention
        # Sliding window layers will use segment-restricted attention
        segment_mask = create_segment_attention_mask(
            segment_ids=segment_ids,
            attention_mask=attention_mask,
            dtype=self.dtype
        )

        # attn_mask = global (unrestricted), sliding_window_mask = segment-restricted
        attn_mask = global_attn_mask
        sliding_window_mask = segment_mask

        hidden_states = self.embeddings(input_ids=input_ids, inputs_embeds=inputs_embeds)

        # Add segment embeddings if enabled
        if self._use_segment_embeddings:
            if segment_ids is None:
                raise ValueError(
                    "use_segment_embeddings=True but segment_ids is None. "
                    "Ensure data prep has 'return_segment_ids: true'."
                )
            segment_emb = self.segment_embedding_layer(segment_ids)
            hidden_states = hidden_states + segment_emb

        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None

        for encoder_layer in self.layers:
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_outputs = encoder_layer(
                hidden_states,
                attention_mask=attn_mask,
                sliding_window_mask=sliding_window_mask,
                position_ids=position_ids,
                output_attentions=output_attentions,
            )
            hidden_states = layer_outputs[0]
            if output_attentions and len(layer_outputs) > 1:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        hidden_states = self.final_norm(hidden_states)

        if not return_dict:
            return tuple(v for v in [hidden_states, all_hidden_states, all_self_attentions] if v is not None)

        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )


class SegmentAttentionModernBertForMaskedLM(nn.Module):
    """ModernBertForMaskedLM with segment-based attention support."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model = SegmentAttentionModernBertModel(config)
        # Get the prediction head components from ModernBert
        # We need to create these manually since we're using a custom model
        from transformers.models.modernbert.modeling_modernbert import ModernBertPredictionHead
        self.head = ModernBertPredictionHead(config)
        self.decoder = nn.Linear(config.hidden_size, config.vocab_size, bias=config.decoder_bias)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        segment_ids: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        **kwargs
    ):
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            segment_ids=segment_ids,
            **kwargs
        )
        hidden_states = outputs.last_hidden_state
        predictions = self.decoder(self.head(hidden_states))

        loss = None
        if labels is not None:
            loss = nn.functional.cross_entropy(
                predictions.view(-1, self.config.vocab_size),
                labels.view(-1),
                ignore_index=-100
            )

        return {'loss': loss, 'logits': predictions, 'hidden_states': outputs.hidden_states}


class SegmentAttentionModernBertForSequenceClassification(nn.Module):
    """ModernBertForSequenceClassification with segment-based attention support."""

    def __init__(self, config, num_labels: int = 2):
        super().__init__()
        self.config = config
        self.num_labels = num_labels
        self.model = SegmentAttentionModernBertModel(config)
        self.classifier = nn.Linear(config.hidden_size, num_labels)
        self.dropout = nn.Dropout(config.classifier_dropout)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        segment_ids: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        **kwargs
    ):
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            segment_ids=segment_ids,
            **kwargs
        )
        hidden_states = outputs.last_hidden_state

        # Pooling: use CLS token (first token) or mean pooling
        pooling = getattr(self.config, 'classifier_pooling', 'cls')
        if pooling == 'cls':
            pooled = hidden_states[:, 0]
        else:  # mean pooling
            mask = attention_mask.unsqueeze(-1).float()
            pooled = (hidden_states * mask).sum(dim=1) / mask.sum(dim=1)

        pooled = self.dropout(pooled)
        logits = self.classifier(pooled)

        loss = None
        if labels is not None:
            loss = nn.functional.cross_entropy(logits, labels)

        return {'loss': loss, 'logits': logits, 'hidden_states': outputs.hidden_states}


class ModernBertModelWithSegmentEmbeds(nn.Module):
    """
    ModernBertModel with optional segment embeddings.

    Uses standard attention (sliding window + global) but enriches
    embeddings with learnable segment type embeddings if configured.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model = ModernBertModel(config)

        # Optional segment embeddings (independent of attention type)
        if getattr(config, 'use_segment_embeddings', False):
            self.segment_embedding_layer = SegmentEmbeddingLayer(
                num_segment_types=getattr(config, 'num_segment_types', 8),
                hidden_size=config.hidden_size
            )
            self._use_segment_embeddings = True
        else:
            self.segment_embedding_layer = None
            self._use_segment_embeddings = False

    @property
    def dtype(self):
        return self.model.embeddings.tok_embeddings.weight.dtype

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        segment_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs
    ):
        # Compute position_ids based on segment_position_encoding if needed
        if position_ids is None and segment_ids is not None:
            mode = getattr(self.config, 'segment_position_encoding', 'absolute')
            if mode in ('reset', 'none'):
                seq_len = segment_ids.shape[1]
                position_ids = compute_position_ids(
                    segment_ids=segment_ids,
                    seq_len=seq_len,
                    mode=mode,
                    device=segment_ids.device
                )

        # If segment embeddings enabled, we need to handle embeddings ourselves
        if self._use_segment_embeddings:
            if segment_ids is None:
                raise ValueError(
                    "use_segment_embeddings=True but segment_ids is None. "
                    "Ensure data prep has 'return_segment_ids: true'."
                )

            # Get token embeddings
            hidden_states = self.model.embeddings(
                input_ids=input_ids,
                inputs_embeds=inputs_embeds
            )

            # Add segment embeddings
            segment_emb = self.segment_embedding_layer(segment_ids)
            hidden_states = hidden_states + segment_emb

            # Pass enriched embeddings through model (skip embedding layer)
            return self.model(
                input_ids=None,  # Don't use - we're passing inputs_embeds
                attention_mask=attention_mask,
                position_ids=position_ids,
                inputs_embeds=hidden_states,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                **kwargs
            )
        else:
            # Standard path - no segment embeddings
            return self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                **kwargs
            )


class ModernBertForMaskedLMWithSegmentEmbeds(nn.Module):
    """ModernBertForMaskedLM with optional segment embeddings."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model = ModernBertModelWithSegmentEmbeds(config)

        # MLM head (same as ModernBert)
        from transformers.models.modernbert.modeling_modernbert import ModernBertPredictionHead
        self.head = ModernBertPredictionHead(config)
        self.decoder = nn.Linear(config.hidden_size, config.vocab_size, bias=config.decoder_bias)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        segment_ids: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        **kwargs
    ):
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            segment_ids=segment_ids,
            **kwargs
        )
        hidden_states = outputs.last_hidden_state
        predictions = self.decoder(self.head(hidden_states))

        loss = None
        if labels is not None:
            loss = nn.functional.cross_entropy(
                predictions.view(-1, self.config.vocab_size),
                labels.view(-1),
                ignore_index=-100
            )

        return {'loss': loss, 'logits': predictions, 'hidden_states': outputs.hidden_states}


class ModernBertForSequenceClassificationWithSegmentEmbeds(nn.Module):
    """ModernBertForSequenceClassification with optional segment embeddings."""

    def __init__(self, config, num_labels: int = 2):
        super().__init__()
        self.config = config
        self.num_labels = num_labels
        self.model = ModernBertModelWithSegmentEmbeds(config)
        self.classifier = nn.Linear(config.hidden_size, num_labels)
        self.dropout = nn.Dropout(config.classifier_dropout)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        segment_ids: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        **kwargs
    ):
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            segment_ids=segment_ids,
            **kwargs
        )
        hidden_states = outputs.last_hidden_state

        # Pooling
        pooling = getattr(self.config, 'classifier_pooling', 'cls')
        if pooling == 'cls':
            pooled = hidden_states[:, 0]
        else:
            mask = attention_mask.unsqueeze(-1).float()
            pooled = (hidden_states * mask).sum(dim=1) / mask.sum(dim=1)

        pooled = self.dropout(pooled)
        logits = self.classifier(pooled)

        loss = None
        if labels is not None:
            loss = nn.functional.cross_entropy(logits, labels)

        return {'loss': loss, 'logits': logits, 'hidden_states': outputs.hidden_states}


class DELBERTForMLM(nn.Module):
    """
    DELBERT model for Masked Language Modeling (pretraining).
    """

    def __init__(self, config: DELBERTConfig):
        super().__init__()
        self.config = config

        # Use segment attention model if configured
        if getattr(config, 'use_segment_attention', False):
            # Segment-restricted attention (+ optional segment embeddings)
            self.model = SegmentAttentionModernBertForMaskedLM(config)
            self._use_segment_attention = True
        else:
            # Standard attention (+ optional segment embeddings)
            self.model = ModernBertForMaskedLMWithSegmentEmbeds(config)
            self._use_segment_attention = False

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        segment_ids: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for MLM.

        Args:
            input_ids: Token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            segment_ids: Segment IDs for intra-FP attention [batch_size, seq_len]
            labels: MLM labels (-100 for non-masked tokens) [batch_size, seq_len]

        Returns:
            Dictionary with 'loss' and 'logits'
        """
        # Validate segment_ids is provided when segment attention is enabled
        if self._use_segment_attention and segment_ids is None:
            raise ValueError(
                "DELBERTForMLM has use_segment_attention=True but segment_ids is None. "
                "Ensure data prep config has 'return_segment_ids: true' and that the data module "
                "and collator are passing segment_ids through to the model."
            )

        # Both model types (SegmentAttention and WithSegmentEmbeds) accept segment_ids
        # The WithSegmentEmbeds wrapper handles segment_ids internally (adds embeddings if enabled)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            segment_ids=segment_ids,
            labels=labels,
            **kwargs
        )
        return outputs
    
    def get_encoder(self):
        """Get the underlying ModernBERT encoder."""
        return self.model.model
    
    @classmethod
    def from_pretrained(cls, checkpoint_path: str, config: Optional[DELBERTConfig] = None):
        """
        Load model from Lightning .ckpt checkpoint.

        Args:
            checkpoint_path: Path to .ckpt file
            config: Optional config (will load from checkpoint if not provided)

        Returns:
            DELBERTForMLM model
        """
        checkpoint = torch.load(checkpoint_path, map_location='cpu')

        if config is None:
            if 'model_config' not in checkpoint:
                raise ValueError("No model_config found in checkpoint and no config provided")
            config = DELBERTConfig(**checkpoint['model_config'])

        model = cls(config)

        # Lightning stores model state under 'state_dict' with 'model.' prefix
        state_dict = {k.replace('model.', '', 1): v for k, v in checkpoint['state_dict'].items()}
        model.load_state_dict(state_dict)

        # Apply layer_types if present in config
        layer_types = getattr(config, 'layer_types', None)
        if layer_types:
            half_window = config.local_attention // 2
            encoder = model.model.model
            for i, (layer, layer_type) in enumerate(zip(encoder.layers, layer_types)):
                if layer_type == 'full_attention':
                    layer.attn.local_attention = (-1, -1)
                elif layer_type == 'sliding_attention':
                    layer.attn.local_attention = (half_window, half_window)

        return model


class DELBERTForSequenceClassification(nn.Module):
    """
    DELBERT model for sequence classification (finetuning).
    """

    def __init__(
        self,
        config: DELBERTConfig,
        num_labels: int = 2,
        pretrained_encoder: Optional[nn.Module] = None
    ):
        super().__init__()
        self.config = config
        self.num_labels = num_labels

        # Use segment attention model if configured
        if getattr(config, 'use_segment_attention', False):
            # Segment-restricted attention (+ optional segment embeddings)
            self.model = SegmentAttentionModernBertForSequenceClassification(config, num_labels)
            self._use_segment_attention = True
        else:
            # Standard attention (+ optional segment embeddings)
            self.model = ModernBertForSequenceClassificationWithSegmentEmbeds(config, num_labels)
            self._use_segment_attention = False

        # Load pretrained encoder if provided
        if pretrained_encoder is not None:
            self.load_pretrained_encoder(pretrained_encoder)

    def load_pretrained_encoder(self, encoder: nn.Module):
        """
        Load weights from pretrained encoder (from MLM model).

        Args:
            encoder: Pretrained ModernBERT encoder
        """
        # Copy encoder weights
        self.model.model.load_state_dict(encoder.state_dict())

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        segment_ids: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for classification.

        Args:
            input_ids: Token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            segment_ids: Segment IDs for intra-FP attention [batch_size, seq_len]
            labels: Classification labels [batch_size]

        Returns:
            Dictionary with 'loss', 'logits', and 'hidden_states'
        """
        # Validate segment_ids is provided when segment attention is enabled
        if self._use_segment_attention and segment_ids is None:
            raise ValueError(
                "DELBERTForSequenceClassification has use_segment_attention=True but segment_ids is None. "
                "Ensure data prep config has 'return_segment_ids: true' and that the data module "
                "and collator are passing segment_ids through to the model."
            )

        # Both model types (SegmentAttention and WithSegmentEmbeds) accept segment_ids
        # The WithSegmentEmbeds wrapper handles segment_ids internally (adds embeddings if enabled)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            segment_ids=segment_ids,
            labels=labels,
            **kwargs
        )
        return outputs

    def get_encoder(self):
        """Get the underlying ModernBERT encoder."""
        return self.model.model
    
    @classmethod
    def from_pretrained_mlm(
        cls,
        mlm_checkpoint: str,
        num_labels: int = 2,
        config: Optional[DELBERTConfig] = None
    ):
        """
        Initialize classification model from pretrained MLM checkpoint.
        
        Args:
            mlm_checkpoint: Path to MLM checkpoint
            num_labels: Number of classification labels
            config: Optional config (will load from checkpoint if not provided)
            
        Returns:
            DELBERTForSequenceClassification model
        """
        # Load MLM model
        mlm_model = DELBERTForMLM.from_pretrained(mlm_checkpoint, config)
        
        # Create classification model with pretrained encoder
        if config is None:
            config = mlm_model.config
        
        model = cls(config, num_labels=num_labels)
        model.load_pretrained_encoder(mlm_model.get_encoder())
        
        return model